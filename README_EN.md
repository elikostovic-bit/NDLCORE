![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg)
![Target](https://img.shields.io/badge/Domain-Neuromorphic%20%2F%20Edge%20AI-success.svg)

# ndlc — NDL Toolchain v1.0

**NDL (Neural Description Language)** is a compiled domain-specific language for modeling, training, and deploying Spiking Neural Networks (SNNs)[cite: 2]. The `ndlc` toolchain compiles `.ndl` source files into **native machine code (via LLVM IR)** and generates **CUDA PTX** for GPU acceleration[cite: 2]. Developers interact exclusively with `.ndl`, `ndl.toml`, and the `ndlc` CLI—all intermediate representations remain completely abstracted[cite: 2].

```
  main.ndl ──▶ ndlc ──▶ LLVM IR (.ll) ──▶ x86_64 / AArch64 native binary ──▶ exec
                 │                └▶ (clang / llc + system linker)
                 ├──▶ CUDA PTX (.ptx) ──▶ JIT via CUDA Driver API (cuModuleLoadData)
                 └──▶ Built-in VM ──▶ execution over libndl_rt (native parity)
```

## Toolchain Components

| Component | Source Files | Purpose |
|---|---|---|
| Lexer | `src/lexer.cpp` | Hand-written lexer, precise source locations, duration literals like `100ms`[cite: 2] |
| Parser | `src/parser.cpp` | Recursive descent parser without external generators, panic-mode recovery[cite: 2] |
| Sema | `src/sema.cpp` | Two-pass analysis: symbol tables, type checking, lexical scopes, const-eval[cite: 2] |
| LLVM IR Backend | `src/irgen_llvm.cpp` | Textual LLVM IR emission (opaque pointers), fast-math, `on_spike` event handlers[cite: 2] |
| CUDA PTX Backend | `src/irgen_ptx.cpp` | 8 compute kernels (LIF step, dense/CSR propagation, STDP, traces, current injection)[cite: 2] |
| Runtime `libndl_rt` | `runtime/*` | LIF + STDP simulation, thread-pool scheduler, clock, RNG, checkpoints, spike raster, GPU JIT[cite: 2] |
| VM | `src/vm.cpp` | AST interpreter layered over the `libndl_rt` C-ABI—semantically identical to native execution[cite: 2] |
| Package Manager | `src/toml.cpp`, `src/manifest.cpp` | `ndl.toml` parsing, module imports, `use` package resolution, cycle detection[cite: 2] |
| Toolchain Driver | `src/linker.cpp`, `src/main.cpp` | clang/llc/ld/nvcc/ptxas orchestration, `init`/`check`/`build`/`run`/`clean` CLI verbs[cite: 2] |

Diagnostics follow `rustc`-style conventions: `error[E0201]`, code snippets with caret indicators, and `help: did you mean …` suggestions[cite: 2].

## Quick Start

```bash
make -j$(nproc)                      # Builds dist/bin/ndlc + dist/lib/libndl_rt.a

./dist/bin/ndlc check main.ndl       # Semantic verification
./dist/bin/ndlc run main.ndl         # Execution (VM mode; GPU via CUDA driver)
./dist/bin/ndlc build main.ndl       # Artifacts: build/<name>.ll + .ptx (+ native binary if clang/llc present)
./dist/bin/ndlc init my-project      # Scaffolds a new project
```

The canonical reference program `main.ndl` runs a 1000→5000→1000→100 topology with STDP in GPU mode: 9 epochs × 100 ms, producing a binary checkpoint `models/snn_trained.ndlbin` (~96 MB) and an output raster `output_spikes.csv` (~6.9k spikes) in seconds[cite: 2].

## Verification (Validated in this Repository)

* `ndlc check/run`: Passed across the reference architecture and all test cases[cite: 2].
* **VM ⇄ Native Parity**: Identical spike raster output (6901 spikes, first spike at t = 47 ms) across both execution modes[cite: 2].
* **Native Pipeline**: Verified end-to-end: `main.ndl → LLVM IR → llvm.verify() OK (LLVM 20) → x86-64 ELF → execution`[cite: 2].
* **Analytical Biophysics Verification**: LIF dynamics verified (threshold reached from resting potential at I = 30, tau = 20 yields a spike at ~14 ms; inter-spike interval ~17 ms)[cite: 2]; STDP potentiation verified (0.05 → 0.99 over 6 runs)[cite: 2].
* **Checkpoint Roundtrip**: Binary `.ndlbin` format parsed and verified with an external inspector; weight save/restore cycle validated[cite: 2].
* **Compiler Diagnostics**: Multi-error recovery verified with up to 6 diagnostic errors per pass, fix suggestions, and slice bounds checks[cite: 2].

## Examples

* `main.ndl` — Reference pipeline: 4 neuron groups, dense/sparse/one-to-one projections, STDP, GPU execution, checkpointing[cite: 2].
* `examples/minimal.ndl` — Minimal 64→64 LIF relay with raster export[cite: 2].
* `examples/stdp_demo.ndl` — Synaptic weight tracking via `get_weight` before and after learning[cite: 2].
* `examples/use_demo/` — Package system demonstration: `use mylib;` with local dependencies defined in `ndl.toml`[cite: 2].

## Repository Layout

```
ndl/
├── main.ndl, ndl.toml, stdlib/   — Reference project and standard library
├── include/ndl/                  — Public headers (AST, sema, codegen, VM contracts)
├── src/                          — Compiler implementation (~7.5k lines of C++20, zero dependencies)
├── runtime/                      — libndl_rt engine (C-ABI; statically linked into binaries)
├── docs/INTERNALS.md             — Internal specification and architectural contracts
├── docs/LANGUAGE.md              — NDL v1.0 Language Reference Manual
└── Makefile, CMakeLists.txt
```

## Runtime & Streaming Inference Benchmark (Edge AI PoC)

This repository includes a standalone, two-phase stress benchmark (`poc_benchmark.cpp`) testing `libndl_rt` on a neuromorphic DVS event stream classification task (N-MNIST-like, 34x34 sensor grid, 2 polarity channels, 10 digit classes) operating in continuous time[cite: 3].

**Topology:** 2312 Input LIF → 320 Excitatory (10x32 class-banded) / 64 Inhibitory → 10 Readout neurons with active 3-factor STDP[cite: 3].

### Benchmark Results (10,000 events: 5000 train / 5000 eval)

| Metric | Measured Value | Target / Budget | Status |
| :--- | :--- | :--- | :--- |
| **Inference Latency (p99)** | **711.28 µs** | 1000 µs (1 kHz) | **28.9% real-time slack** (PASS) |
| **Mean Latency** | **635.79 µs** | 1000 µs | Fast deterministic response |
| **Jitter (RMS)** | **± 38.84 µs** | < 50 µs | High determinism |
| **Baseline Accuracy (Untrained Chance)** | **10.00%** | — | Noise floor |
| **Eval Accuracy (Frozen STDP, No Teacher)** | **41.60%** | > 30.0% | **+31.60 pp accuracy lift** |
| **Synaptic Selectivity Ratio (Own/Other W)** | **3.98** | > 2.0 | Class-selective projection learned |
| **Peak Synaptic Weight (W_max)** | **1.52** | < 5.0 | Stable (no runaway potentiation) |
| **RAM Footprint (RSS steady-state)** | **10.6 MB** | < 32 MB | Embedded / MCU compatible |

```bash
# Build and run the benchmark:
g++ -std=c++20 -O3 -I runtime poc_benchmark.cpp dist/lib/libndl_rt.a -lpthread -o poc_benchmark
./poc_benchmark
```

## Requirements

* **Compilation:** Any standard C++20 compiler (`g++` >= 11, `clang++` >= 12), POSIX-compliant environment[cite: 2].
* **Native Backend:** `clang` or `llc` (optional; falls back to internal VM if absent)[cite: 2].
* **GPU Acceleration:** NVIDIA CUDA driver (PTX JIT via Driver API; full CUDA Toolkit not required) with automatic CPU fallback[cite: 2].

## License

MIT (see [LICENSE](LICENSE))[cite: 2].

---

### Project Support
* **Solana:** `13e382pwdcEbZGNbSbakbSFWVKp976sSTPYEgZzLo5eG`
* **Bitcoin:** `bc1qk3c7dr4rhkc97n9gycqd3ymkag4kjhlpcgxxcp`
* **Ethereum:** `0x8E2421eC40559e1b2924095535A6dF0a5297a870`
