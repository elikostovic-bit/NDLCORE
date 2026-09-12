![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg)
![Target](https://img.shields.io/badge/Domain-Neuromorphic%20%2F%20Edge%20AI-success.svg)
# ndlc — NDL Toolchain v1.0

**NDL (Neural Description Language)** — компилируемый язык для моделирования,
обучения и запуска спайковых нейроморфных систем (SNN). Тулчейн `ndlc`
компилирует `.ndl` в **нативный машинный код (через LLVM IR)** и генерирует
**CUDA PTX** для GPU-ускорения. Пользователь работает только с `.ndl`,
`ndl.toml` и CLI `ndlc` — промежуточные языки полностью скрыты.

```
  main.ndl ──▶ ndlc ──▶ LLVM IR (.ll) ──▶ x86_64 / AArch64 native binary ──▶ exec
                 │                └▶ (clang / llc + системный линкер)
                 ├──▶ CUDA PTX (.ptx) ──▶ JIT через CUDA Driver API (cuModuleLoadData)
                 └──▶ встроенная VM ──▶ исполнение поверх libndl_rt (паритет с native)
```

## Состав тулчейна

| Модуль | Файлы | Назначение |
|---|---|---|
| Lexer | `src/lexer.cpp` | Ручной лексер, точные Source Location, duration-литералы `100ms` |
| Parser | `src/parser.cpp` | Рекурсивный спуск без генераторов, panic-mode восстановление |
| Sema | `src/sema.cpp` | Двухпроходный анализ: таблицы символов, type checker, scopes, const-eval |
| LLVM IR Backend | `src/irgen_llvm.cpp` | Текстовый LLVM IR (opaque pointers), fastmath, обработчики on_spike |
| CUDA PTX Backend | `src/irgen_ptx.cpp` | 8 кернелов (LIF, propagate dense/CSR, STDP, traces, инжекции) |
| Runtime `libndl_rt` | `runtime/*` | Сетка LIF+STDP, планировщик-пул, часы, RNG, чекпоинты, raster, GPU JIT |
| VM | `src/vm.cpp` | Интерпретатор AST поверх C ABI libndl_rt — идентичная семантика |
| Package Manager | `src/toml.cpp`, `src/manifest.cpp` | `ndl.toml`, импорты, `use`-пакеты, детект циклов |
| Toolchain Driver | `src/linker.cpp`, `src/main.cpp` | clang/llc/ld/nvcc/ptxas, команды init/check/build/run/clean |

Диагностика — в стиле rustc: `error[E0201]`, сниппет с caret, `help: did you mean …`.

## Быстрый старт

```bash
make -j$(nproc)                      # dist/bin/ndlc + dist/lib/libndl_rt.a

./dist/bin/ndlc check main.ndl       # семантическая проверка
./dist/bin/ndlc run main.ndl         # исполнение (VM; GPU с CUDA-драйвером)
./dist/bin/ndlc build main.ndl       # артефакты: build/<name>.ll + .ptx (+ нативный бинарник при наличии clang/llc)
./dist/bin/ndlc init my-project      # каркас нового проекта
```

Эталонная программа `main.ndl` (сеть 1000→5000→1000→100 с STDP, GPU-режим):
9 эпох × 100 мс, чекпоинт `models/snn_trained.ndlbin` (~96 МБ) и растр
`output_spikes.csv` (~6.9k спайков). Полный прогон — секунды.

## Верификация (выполнена в этом репозитории)

* `ndlc check/run` — эталонная программа и примеры: OK.
* **VM ⇄ Native паритет**: тот же растр (6901 спайк, первый спайк t=47 мс) в обоих режимах.
* **Native путь**: `main.ndl → LLVM IR → llvm.verify() OK (LLVM 20) → x86-64 ELF → запуск`.
* **Физика сверена аналитически**: LIF (порог из rest при I=30, tau=20 → спайк через
  ~14 мс; межспайковый интервал ~17 мс), STDP (0.05 → 0.99 за 6 прогонов).
* Чекпоинт `.ndlbin` — бинарный формат разобран и проверен внешним парсером; roundtrip
  save/restore весов — OK.
* Диагностика: батч из 6 ошибок за проход с предложениями исправлений и bounds-check срезов.

## Примеры

* `main.ndl` — эталон: 4 группы, dense/sparse/one-to-one, STDP, GPU-прогоны, чекпоинт.
* `examples/minimal.ndl` — минимальное реле LIF 64→64, экспорт растра.
* `examples/stdp_demo.ndl` — рост веса через get_weight до/после обучения.
* `examples/use_demo/` — пакеты: `use mylib;` + локальная зависимость в `ndl.toml`.

## Структура проекта

```
ndl/
├── main.ndl, ndl.toml, stdlib/   — эталонный проект
├── include/ndl/                  — контракты (AST, sema, кодогены, VM, …)
├── src/                          — компилятор (~7.5k строк C++20 без внешних зависимостей)
├── runtime/                      — libndl_rt (C ABI; статически вшивается в бинарники)
├── docs/INTERNALS.md             — внутренние контракты всех модулей
├── docs/LANGUAGE.md              — справочник языка NDL v1.0
└── Makefile, CMakeLists.txt
```
## Бенчмарк рантайма и потокового инференса (PoC Edge AI)

В репозитории представлен двухфазный стресс-тест рантайма `libndl_rt` (`poc_benchmark.cpp`), демонстрирующий потоковую классификацию событийного DVS-сенсора (N-MNIST, сетка 34x34, 2 канала полярности, 10 классов) в непрерывном времени[cite: 4, 5].

**Архитектура:** 2312 входов -> 320 скрытых возбуждающих (E) / 64 тормозных (I) нейрона -> 10 считывающих нейронов с активным 3-факторным STDP[cite: 4, 5].

### Результаты измерений (10 000 событий, 5000 train / 5000 eval)

| Метрика | Значение | Порог / Бюджет | Статус |
| :--- | :--- | :--- | :--- |
| **Латентность p99 (Инференс)** | **711.28 мкс** | 1000 мкс (1 кГц) | **28.9% запас по времени** (PASS)[cite: 5] |
| **Латентность средняя** | **635.79 мкс** | 1000 мкс | Высокая скорость отклика[cite: 5] |
| **Джиттер (RMS)** | **± 38.84 мкс** | < 50 мкс | Жесткий детерминизм[cite: 5] |
| **Базовая точность (случайная)** | **10.00 %** | — | Уровень шума[cite: 5] |
| **Точность Eval (без учителя, STDP заморожен)** | **41.60 %** | > 30.0 % | **+31.60 п.п. прирост точности**[cite: 5] |
| **Коэффициент селективности (Own/Other W)** | **3.98** | > 2.0 | Селективное обучение связей[cite: 5] |
| **Максимальный вес синапса ($W_{max}$)** | **1.52** | < 5.0 | Стабильность без взрыва весов[cite: 5] |
| **Потребление RAM (RSS steady-state)** | **10.6 МБ** | < 32 МБ | Пригодно для микроконтроллеров[cite: 5] |

```bash
# Сборка и прогон бенчмарка:
g++ -std=c++20 -O3 -I runtime poc_benchmark.cpp dist/lib/libndl_rt.a -lpthread -o poc_benchmark
./poc_benchmark
```

## Требования

Сборка: любой C++20 компилятор (g++ ≥ 11, clang++ ≥ 12), POSIX.
Нативный бэкенд: clang или llc (опционально; иначе VM-режим).
GPU: драйвер CUDA (PTX JIT через Driver API, тулкит не нужен), иначе CPU-fallback.

Лицензия: MIT (см. LICENSE).

---

### Поддержка проекта
* **Solana:** `13e382pwdcEbZGNbSbakbSFWVKp976sSTPYEgZzLo5eG`
* **Bitcoin:** `bc1qk3c7dr4rhkc97n9gycqd3ymkag4kjhlpcgxxcp`
* **Ethereum:** `0x8E2421eC40559e1b2924095535A6dF0a5297a870`
