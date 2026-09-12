# NDL v1.0 — Language Reference (Neural Description Language)

NDL — компилируемый предметно-ориентированный язык для моделирования спайковых
нейроморфных сетей (SNN). Компилируется в нативный машинный код через LLVM IR и
в CUDA PTX для GPU-ускорения. Пользователь работает только с файлами `.ndl`,
манифестом `ndl.toml` и CLI-инструментом `ndlc`.

## Программа

Программа состоит из последовательности элементов верхнего уровня:

```ndl
import "stdlib/math.ndl";       // импорт (файл-относительный → корень проекта → std)

node_group Input[1000] : Excitatory(tau=20.0, threshold=-55.0, rest=-70.0, reset=-75.0);
node_group Inhib[200]  : Inhibitory(tau=10.0, threshold=-50.0, rest=-70.0, reset=-70.0);

dense_connect(Input, Hidden, weight_func=random_gaussian(0.5, 0.1), plastic=true);
configure_stdp(Input -> Hidden, lr_pot=0.01, lr_dep=0.005, window_ms=20.0);

let epochs = 10;
for (e in 1..epochs) { run (duration = 100ms, dt = 1.0ms, device = GPU) { ... } }
```

### node_group

```ndl
node_group <Имя>[<Размер>] : <Тип>(tau=<f>, threshold=<f>, rest=<f>, reset=<f>);
```
* `<Тип>` ∈ `Excitatory | Inhibitory` (метаданные; знак веса задаётся явно).
* Все четыре параметра обязательны. LIF-нейрон (явный Эйлер, R=1):
  `v' = v + dt·(rest − v + I)/tau`; спайк при `v' ≥ threshold`, затем `v = reset`.
* Размер — константное выражение (литералы, `+ - * /`, константные `let`).

### Связи

```ndl
dense_connect(Src, Dst, weight=0.4, plastic=false);
dense_connect(Src, Dst, weight_func=random_gaussian(0.5, 0.1), plastic=true);
dense_connect(Src, Dst, weight_func=random_uniform(-0.2, 0.8));
sparse_connect(Src, Dst, density=0.2, weight=0.8);
one_to_one_connect(Src, Dst, weight=1.0);
```
* `weight_func` вычисляется **для каждого синапса** при создании связи
  (row-major обход, детерминированный RNG).
* Повторная связь той же пары заменяет предыдущую.
* Связь без `plastic=true` не участвует в STDP.

### Пластичность

```ndl
configure_stdp(Src -> Dst, lr_pot=0.01, lr_dep=0.005, window_ms=20.0);
```
Парная STDP на каждом тике:
`ΔW[j][i] = lr_pot·tr_pre[i]·s_post[j] − lr_dep·tr_post[j]·s_pre[i]`,
следы: `tr[i] = tr[i]·exp(−dt/window) + s[i]`.
Требует ранее объявленной пластичной связи `Src -> Dst`.

### Симуляция во времени

```ndl
run (duration = 100ms, dt = 1.0ms, device = GPU) {
    at 5ms  emit Input[0..100](current = 25.0);
    at 20ms emit Input[100..300](current = 30.0);
}
```
* `duration` обязателен (`Duration` или `float`, мс); `dt` по умолчанию 1.0 мс;
  `device` по умолчанию `CPU` (при недоступном CUDA — честное предупреждение и CPU-fallback).
* Тело `run` содержит только `at ... emit ... ;`.
* **`at T emit G[срез](current = C)`** — ступенчатый ток: с момента `T` до конца
  прогона (или до перезаписи поздним `emit` на тех же нейронах). Срезы:
  `G` (вся группа), `G[i]`, `G[a..b]` — полуинтервал `[a, b)`.
* Синаптический ток затухает экспоненциально: `I_syn ← I_syn·exp(−dt/5мс) + Σ W·s`.
* `run` разрешён внутри `if`/`for`/`while`; вложенные `run` запрещены.

### События

```ndl
on_spike(OutputLayer) {
    print("spike at", spike_time, "from neuron", neuron_index);
}
```
Только верхний уровень; внутри доступны `neuron_index: int` и `spike_time: float`
(время детекции в мс), переменные верхнего уровня и группы. Несколько обработчиков
на одну группу — срабатывают в порядке объявления.

### Ввод-вывод и служебные

```ndl
print("Epoch:", e);                          // значения через пробел, перевод строки
save_checkpoint("models/snn.ndlbin");        // граф + матрицы весов + состояния
load_checkpoint("models/snn.ndlbin");
export_raster(OutputLayer, "spikes.csv");    // CSV: t_ms,neuron_id
```

### Управление

```ndl
if (x > 0.5) { ... } else if (...) { ... } else { ... }
for (i in 1..N) { ... }        // полуинтервал [1, N)
while (cond) { ... }
let y: float = x * 2.0;
y = y + 1.0;
```

## Типы

| Тип       | Описание                                    | Литерал            |
|-----------|---------------------------------------------|--------------------|
| `int`     | 64-битное целое                              | `42`, `0x1F`       |
| `float`   | 64-битное вещественное                       | `3.14`, `1e-3`     |
| `bool`    | логический                                   | `true`/`false`     |
| `string`  | строка (только печать/пути/конкатенация `+`) | `"hello"`          |
| `duration`| время в мс (результат `5ms`, `t*2.0`)        | `100ms`            |
| `tensor<float, Dims>` | тензор (экспериментально)        | —                  |

`int → float` повышается неявно. `Duration ± число` — ошибка типизации
(безопасность единиц). `&&`, `||`, `!` — только для `bool`; условия `if`/`while`
строго `bool`. Диапазоны `a..b` — полуинтервалы.

## Встроенные функции

`sin cos exp log sqrt` · `random_uniform(a, b)` · `random_gaussian(mu, sigma)` ·
`get_weight(A, B, i, j)` · `tensor_new{,2,3} / tensor_get{,2,3} / tensor_set{,2,3} / dump_tensor`

## Комментарии

`// строка` и `/* блочный */`.

## ndl.toml

```toml
[project]
name = "snn-demo"
version = "0.1.0"
ndl_version = "1.0"
main = "main.ndl"

[build]
target = "auto"      # auto | cpu | gpu
opt_level = 2        # уровень оптимизаций LLVM (0..3)
fastmath = true      # fast-math флаги LLVM

[simulation]
seed = 42            # детерминизм: xoshiro256**

[dependencies]
mylib = { path = "vendor/mylib" }   # локальные пакеты; подключение: use mylib;
```

## Семантика исполнения (сводка)

1. Импорты резолвятся DFS (зависимости раньше, циклы — ошибка), модули исполняются по порядку.
2. Верхний уровень — императивный сценарий: группы/связи/STDP создаются в порядке объявления.
3. `run` — фиксированный шаг dt: затухание синапсов → распространение спайков →
   инжекции → LIF → STDP → следы → лог/обработчики.
4. Детерминизм: фиксированный seed, последовательная инициализация весов,
   параллелизм без влияния на результат.
