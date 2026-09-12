# СБОРКА И ЗАПУСК — нейросеть «Аги» (standalone-пакет)

Этот пакет **полностью самодостаточен**: рантайм NDL (`runtime/`) лежит внутри
и компилируется вместе с хостом. Установленный NDL / `ndlc` больше НЕ нужны —
рассинхрон версий (ошибка `ndl_rt_normalize_incoming was not declared`)
теперь невозможен в принципе.

---

## Структура

```
agi-standalone/
├── BUILD.md               ← вы здесь (инструкция)
├── build_windows.bat      ← сборка в один клик (Windows)
├── run_windows.bat        ← запуск с правильной кодировкой
├── build.sh               ← сборка (Linux / macOS)
├── agi/
│   ├── agi.ndl            ← описание мозга на NDL (для чтения и ndlc, не для запуска)
│   ├── ndl.toml
│   ├── GUIDE.md           ← ПОЛНОЕ руководство по общению и обучению
│   ├── README.md
│   ├── host/agi_chat.cpp  ← весь хост: embedder, контекст, руки, скриншот, рост
│   ├── host/WINDOWS.md    ← таблица MinGW-ошибок «символ → заголовок → -lбиблиотека»
│   ├── models/            ← здесь живёт её мозг (создаётся при :save)
│   └── testdata/          ← примеры датасетов для :autotrain
└── runtime/               ← рантайм NDL v2.1 (GPU: OpenCL), той же версии, что и хост
```

Важно: запускать exe нужно из корня пакета (там, где `agi/models`) — модели
сохраняются в `agi/models` относительно текущего каталога.

---

## 1. Windows — сборка

Нужен только **MinGW-w64** (g++ 12+): MSYS2 (`pacman -S
mingw-w64-ucrt-x86_64-gcc`), w64devkit, или любой другой комплект.

```bat
:: открыть cmd В КАТАЛОГЕ ПАКЕТА (где лежит этот BUILD.md)
build_windows.bat
```

Скрипт сам проверит наличие g++ и соберёт `agi_chat.exe`. Вручную то же самое:

```bat
g++ -std=c++20 -O2 -I runtime ^
    agi\host\agi_chat.cpp ^
    runtime\ndl_rt.cpp runtime\ndl_rt_network.cpp runtime\ndl_rt_io.cpp ^
    runtime\ndl_rt_gpu.cpp runtime\ndl_rt_backend.cpp runtime\ndl_rt_ocl.cpp ^
    runtime\ndl_rt_sched.cpp runtime\ndl_rt_v2.cpp ^
    -lwinpthread -static -o agi_chat.exe
```

- `-static` вшивает MinGW-DLL внутрь exe — запускается на любой машине.
- Если хотите clang — замените `g++` на `clang++`, остальное то же.

## 2. Windows — запуск

```bat
run_windows.bat
:: или вручную:
chcp 65001
agi_chat.exe
```

`chcp 65001` обязателен (UTF-8 в консоли) + шрифт TrueType (Consolas/Cascadia)
в свойствах окна, иначе кириллица будет кашей.

## 3. Linux / macOS — сборка и запуск

```bash
cd agi-standalone
./build.sh            # или вручную:
# g++ -std=c++20 -O2 -I runtime agi/host/agi_chat.cpp \
#     runtime/ndl_rt.cpp runtime/ndl_rt_network.cpp runtime/ndl_rt_io.cpp \
#     runtime/ndl_rt_gpu.cpp runtime/ndl_rt_backend.cpp runtime/ndl_rt_ocl.cpp \
#     runtime/ndl_rt_sched.cpp runtime/ndl_rt_v2.cpp \
#     (и держите ndl_ocl_kernels.cl рядом с agi_chat — его читает OpenCL-бэкенд)
#     -lpthread -ldl -o agi_chat
./agi_chat
```

---

## 4. Первая минута с ней

```
привет ||| привет! я тебя слушаю     :: teach-пара (услышал X → ответил Y)
:exam                                :: самопроверка (должна быть 100%)
:autotrain agi/testdata              :: залить примеры датасетов без ручной работы
как тебя зовут?                      :: спросить
+                                    :: похвалить последний ответ
:save                                :: сохранить мозг в agi/models
```

Дальше — `agi/GUIDE.md`: все команды (§4), датасеты без ручной работы (§5),
контекст диалога и три источника ответов (§6), курс обучения языку (§7),
руки `!команды` (§9), память об ошибках (§11), самостоятельный рост мозга
(§12).

---

## 5. Частые ошибки

| Ошибка                                                       | Решение |
|--------------------------------------------------------------|---------|
| `ndl_rt_normalize_incoming was not declared`                 | вы собираете со СТАРЫМ рантаймом (заголовок/библиотека установленного NDL). Собирайте из этого пакета — `runtime/` здесь той же версии, что хост |
| `'_open_osfhandle' / '_O_RDONLY' was not declared` (senses/vbm.cpp) | этот файл в сборку НЕ нужен — все «руки» уже внутри agi_chat.cpp. Собирайте ровно команду из §1. Если всё же нужен свой vbm.cpp: `#include <io.h>` + `#include <fcntl.h>` (см. agi/host/WINDOWS.md) |
| `undefined reference to std::thread...`                      | добавить `-lwinpthread` |
| При запуске `libwinpthread-1.dll not found`                  | пересобрать с `-static` |
| Кириллица — каша                                             | `chcp 65001` + TrueType-шрифт |
| `!screen` не снимает экран                                   | на Windows работает через PowerShell из коробки; проверьте, что exe запущен не из службы |
| `!run` пишет «not whitelisted»                               | это защита: `:allow <префикс>` или `:allowall` |

## 6. Что где хранится у неё «в голове»

`agi/models/`: чекпоинт мозга, словарь и семантика (`chat_memory.txt`),
языковая модель генератора (`chat_lm.txt`), уроки об ошибках
(`chat_lessons.txt`), приоры декодера (`chat_prior.bin`), файлы «рук»
(`agi/models/files` — песочница). Удалите каталог — получите новорождённую.

## Демон и клиент (Фаза 0)

```bat
agi_chat.exe --daemon            rem она живёт без консоли, лог: agi\models\log.txt
agi-ctl.exe                      rem подключиться и говорить (другое окно)
```
Порт по умолчанию 42777 (`--port 5000`). Ctrl+C — мягкое завершение
с сохранением мозга. `agi-ctl` доступен и в Linux/macOS (build.sh собирает).
