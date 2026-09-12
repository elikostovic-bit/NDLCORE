# NDL v1.0 — Internal Contracts (INTERNALS)

This document is the SINGLE SOURCE OF TRUTH for all toolchain modules.

Repository root: `/home/z/my-project/ndl/`

```
ndl/
├── include/ndl/     headers (CONTRACTS)
├── src/             compiler sources
├── runtime/         libndl_rt (C ABI + C++ impl, statically linked into every binary)
├── stdlib/          reference stdlib of the example project
├── examples/        example .ndl programs
├── docs/            this file, LANGUAGE.md
└── Makefile         builds dist/bin/ndlc + dist/lib/libndl_rt.a
```

Pipeline: `.ndl files → ImportResolver → Lexer → Parser → AST (Program) → Sema →
{ LLVMCodeGen (.ll), PTXCodeGen (.ptx), VM (interpreter calling libndl_rt) }`.

Execution modes:
* **VM mode** (`ndlc run`): tree-walking interpreter executes the checked AST; every
  heavy operation (simulation, connections, STDP, IO) is a call into `libndl_rt` —
  the same library a native binary links. VM and native semantics are identical.
* **Native mode** (`ndlc build`): emits LLVM IR text → external clang/llc → object →
  system linker with `libndl_rt.a` → native executable. Verified in-repo: the IR
  parses & passes `llvm.verify()` under LLVM 20 (opaque pointers) and lowers to a
  working x86-64 ELF.
* **GPU**: `libndl_rt` loads the emitted PTX via the CUDA driver API
  (`cuModuleLoadData` → in-driver JIT). No CUDA driver → single warning, CPU fallback.

## 1. Error codes

| code  | producer          | meaning                        |
|-------|-------------------|--------------------------------|
| E0001 | lexer             | lexical error                  |
| E0100 | parser            | syntax error                   |
| E02xx | sema              | semantic errors (200..299)     |
| E03xx | codegen           | internal codegen errors        |
| E04xx | resolver/manifest | import/package errors          |
| E05xx | driver            | CLI/project errors             |
| W02xx | sema              | warnings                       |

Diagnostics render rustc-style: `error[E0201]: message`, `--> file:line:col`,
source line, caret underline, `help:` suggestions.

## 2. Lexical contract

* Comments: `// ...` to EOL; `/* ... */` block (not nested).
* Identifiers: `[A-Za-z_][A-Za-z0-9_]*`.
* Int literal: decimal, hex `0x1F`.
* Float literal: digit required before `.`; exponents allowed.
* Duration literal: `<int|float>ms` (`100ms`, `1.0ms`) → one token, value in ms.
  `..` disambiguation: number ends before `..` (`1..epochs`).
* Strings: `"..."` with `\" \\ \n \t \r \0`.
* Keywords: node_group dense_connect sparse_connect one_to_one_connect configure_stdp
  run at emit on_spike if else for in while let set print save_checkpoint
  load_checkpoint export_raster import use true false int float bool string tensor
  Excitatory Inhibitory CPU GPU ms
* Operators: `+ - * / % == != < > <= >= && || ! = -> .. , ; : ( ) [ ] { }`.

## 3. Grammar

```ebnf
program       = { top_item } EOF ;
top_item      = import_decl | use_decl | node_group_decl | stmt ;
import_decl   = "import" STRING ";" ;
use_decl      = "use" IDENT ";" ;
node_group_decl = "node_group" IDENT "[" expr "]" ":" group_type
                  "(" [ IDENT "=" expr { "," IDENT "=" expr } ] ")" ";" ;
group_type    = "Excitatory" | "Inhibitory" ;

stmt          = connect_stmt | stdp_stmt | let_stmt | assign_stmt | print_stmt
              | save_stmt | load_stmt | raster_stmt | if_stmt | for_stmt | while_stmt
              | run_block | at_emit | on_spike | expr_stmt ;
connect_stmt  = ("dense_connect" | "sparse_connect" | "one_to_one_connect")
                "(" [ arg { "," arg } ] ")" ";" ;
arg           = IDENT "=" expr | expr ;
stdp_stmt     = "configure_stdp" "(" IDENT "->" IDENT { "," IDENT "=" expr } ")" ";" ;
let_stmt      = "let" IDENT [ ":" type ] "=" expr ";" ;
type          = ( "int" | "float" | "bool" | "string" )
              | "tensor" "<" "float" "," INT { "," INT } ">" ;
assign_stmt   = IDENT "=" expr ";" ;
print_stmt    = "print" "(" [ expr { "," expr } ] ")" ";" ;
save_stmt     = "save_checkpoint" "(" expr ")" ";" ;
load_stmt     = "load_checkpoint" "(" expr ")" ";" ;
raster_stmt   = "export_raster" "(" IDENT "," expr ")" ";" ;
if_stmt       = "if" "(" expr ")" block [ "else" ( if_stmt | block ) ] ;
for_stmt      = "for" "(" IDENT "in" expr ".." expr ")" block ;
while_stmt    = "while" "(" expr ")" block ;
block         = "{" { stmt } "}" ;
run_block     = "run" "(" run_param { "," run_param } ")" block ;
run_param     = ( "duration" | "dt" ) "=" expr | "device" "=" ( "CPU" | "GPU" ) ;
at_emit       = "at" expr "emit" IDENT [ "[" ( expr | expr ".." expr ) "]" ]
                "(" "current" "=" expr ")" ";" ;
on_spike      = "on_spike" "(" IDENT ")" block ;

expr          = logic_or { handled by precedence climbing } ;
precedence    = || < && < == != < < > <= >= < + - < * / % < unary(!,-) < primary ;
primary       = INT | FLOAT | DURATION | STRING | "true" | "false" | "CPU" | "GPU"
              | IDENT | call | "(" expr ")" ;
call          = IDENT "(" [ arg { "," arg } ] ")" ;
```

`Group[i]` / `Group[a..b]` indexing is parsed ONLY in the `at_emit` target.

## 4. Semantic contract

Two passes over the merged Program (modules in import order, main module LAST).

Pass 1: collect direct top-level groups and lets (const-evaluated sizes).
Pass 2: full walk with scopes:

* Placement: NodeGroup/Connect/Stdp/Import/Use top-level only (E0207);
  `at` only directly inside a run body (E0212); on_spike top-level only (E0213);
  nested run (E0211); run body holds only AtEmit (E0210); nested handlers (E0214).
* node_group requires exactly tau, threshold, rest, reset (E0205).
* dense: `weight` OR `weight_func` (= random_gaussian(mu,σ) | random_uniform(a,b));
  sparse: `density` (0<d≤1) + `weight`; one_to_one: `weight`; optional `plastic`.
* configure_stdp requires a plastic connection src→dst declared EARLIER (E0209).
* Types: Int(i64) / Float(f64) / Bool / Str / Duration(f64 ms) / Group / Tensor / Void.
  Int→Float implicit; everything else strict. Duration unit-safety: Duration±numeric → error.
* Ranges are HALF-OPEN `[lo, hi)` (`1..epochs` → 9 iterations for epochs=10).
* for bounds Int; if/while conditions strictly Bool; `&&`/`||` short-circuit.
* Const-eval (sizes, tensor dims, static bounds checks): int literals + `+ - * /`
  + const top-level int lets. Static slice bounds checked against group size (E0203).
* on_spike body sees `neuron_index: Int`, `spike_time: Float`, top-level lets, groups.
* Builtins: sin cos exp log sqrt (Float); random_uniform(a,b), random_gaussian(μ,σ)
  (Float, immediate draw in expression position; per-synapse draw inside weight_func);
  get_weight(Group,Group,Int,Int)→Float; tensor_new{,2,3}, tensor_get{,2,3},
  tensor_set{,2,3}, dump_tensor (experimental tensors).
* SemaResult fills: groups (order), globals (order), varTypes (per binding site),
  exprTypes (EVERY expression node, post-promotion) — codegen depends on these.

## 5. Network model & runtime semantics (libndl_rt)

LIF neuron (explicit Euler, R=1 normalization):

```
v' = v + dt * (rest - v + I_eff) / tau
if v' >= threshold: spike; v' = reset
I_eff = I_syn + hold
```

Synaptic current (current-based synapse with exponential decay, tau_syn = 5 ms):

```
I_syn ← I_syn * exp(−dt / tau_syn) + Σ_connections Σ_i W[j][i] * s_prev[i]
hold  = sustained step currents set by `at T emit G[..](current=C)`
        (from T until the end of the run or until overwritten by a later emit
        on the same neurons; zeroed at the start of every run)
```

Tick order per dt step (t = tick*dt):
1. `I_syn *= exp(−dt/tau_syn)` (synaptic decay)
2. propagate per connection into `I_syn` (impulses)
3. injections with `T ∈ [t, t+dt)`: `hold[idx] = current` (assign, step)
4. LIF step using `I_eff = I_syn + hold` → `s_new`
5. STDP per plastic connection with a config (A=src, B=dst):
   `ΔW[j][i] = lr_pot * tr_A[i] * s_new_B[j] − lr_dep * tr_B[j] * s_prev_A[i]`
6. traces: `tr_g[i] = tr_g[i] * decay_g + s_new_g[i]`,
   `decay_g = exp(−dt / window_ms)` (window of the first STDP config mentioning g;
   default 20 ms)
7. spike log `(t, i)` per group; handlers fire in declaration order, index order
8. `s_prev := s_new`

`ticks = llround(duration / dt)`, ≥ 1. Each run starts electrically clean
(`I_syn = hold = 0`). Injections are consumed or dropped at the end of a run.
Storage: dense = dst-major `W[j*n_src + i]`; sparse = CSR by dst (u64 rowptr,
u32 cols, f64 vals); one-to-one = per-index vector (sizes must match).
Connections between the same pair replace each other (same rule as checkpoint load).

Determinism: xoshiro256** RNG, seed via `ndl_rt_set_seed` (default 42 from
ndl.toml). Weight init at connect time is sequential row-major — identical in VM
and native. Thread pool parallelism is data-disjoint and rate-independent.

Checkpoint `.ndlbin` (little-endian):
```
"NDLB" | u32 version=1 | u32 n_groups | u32 n_conns | u32 n_stdp | u32 reserved
group:  u32 name_len, name, i64 size, i32 ntype, f64 tau, threshold, rest, reset,
        window_ms, f64 v[size]
conn:   i32 kind, u8 plastic, u32 src_len, src, u32 dst_len, dst,
        dense: u64 n, f64 w[n] | sparse: u64 n_rowptr, u64 rowptr[], u64 nnz,
               u32 cols[], f64 vals[] | one2one: u64 n, f64 w[n]
stdp:   src name, dst name, f64 lr_pot, lr_dep, window_ms
```
Load matches groups by name (size mismatch → panic), replaces connections by pair,
clears spike logs. `export_raster` writes CSV `t_ms,neuron_id`.

print protocol: `print_begin` … values separated by single spaces … `print_end`
(newline + flush). f64 printed as `%.6g`.

## 6. Runtime C ABI

See `runtime/ndl_rt.h` (frozen). Dist kinds: 0 constant(p0), 1 gaussian(p0,p1),
2 uniform(p0,p1). Device: 0 CPU, 1 GPU. Neuron type: 0 Excitatory, 1 Inhibitory.
`ndl_rt_panic` → `runtime error: …` + exit(70). GPU fallback warning printed once.

## 7. LLVM IR conventions

* LLVM 15+ opaque pointers (`ptr`); validated under LLVM 20 (`verify()` OK).
* Naming: `@g_<var>` (top-level lets), `@grp_<Name>` (group handles),
  `@.str.N` (interned strings), `@ndl_on_spike.N` (handlers),
  `@ndl_user_main`, `@main`.
* Types: Int→i64, Float/Duration→double, Bool→i1 (regs) / i8 (memory),
  Str/Tensor/Group→ptr. Every variable = alloca hoisted to function entry +
  load/store. `fast` flags when fastmath. `&&`/`||` short-circuit via i8 temporaries.
* All libndl_rt functions are declared; group handles are global slots — ALWAYS
  `load ptr` before passing a group to the runtime (never pass `@grp_X` itself).
* run block: schedule_inject per at-emit (source order) → `@ndl_rt_run`.
* Handler signature: `void @ndl_on_spike.N(i64 %neuron_param, double %time_param, ptr %user_param)`.
* PTX embed: `@ndl_ptx_source` byte-escaped constant + `@ndl_rt_gpu_load_ptx` call
  at the top of `@ndl_user_main`.

## 8. PTX contract

PTX 7.0, `.target sm_70`, `.address_size 64`, f64 math. Eight kernels:

```
ndl_gpu_lif_step(u64 v, u64 i_syn, u64 spike, f64 tau, f64 threshold, f64 rest,
                 f64 reset, f64 dt, u32 n)
ndl_gpu_propagate_dense(u64 w, u64 s_in, u64 i_syn, u32 n_src, u32 n_dst)
ndl_gpu_propagate_csr(u64 rowptr, u64 cols, u64 vals, u64 s_in, u64 i_syn, u32 n_dst)
ndl_gpu_inject_apply(u64 idx, u64 cur, u32 count, u64 hold)      // hold[idx]=cur
ndl_gpu_hold_apply(u64 syn, u64 hold, u64 i_eff, u32 n)          // i_eff = syn+hold
ndl_gpu_syn_decay(u64 i_syn, f64 decay, u32 n)                   // i_syn *= decay
ndl_gpu_stdp_update(u64 w, u64 tr_src, u64 tr_dst, u64 s_prev, u64 s_new,
                    f64 lr_pot, f64 lr_dep, u32 n_src, u32 n_dst)
ndl_gpu_trace_update(u64 tr, u64 s_new, f64 decay, u32 n)
```
GPU tick order mirrors the CPU loop exactly: syn_decay → propagate → inject →
hold_apply → lif → stdp → trace → spike download (host logs/handlers) → swap.
Pointers are cvta'd to global space before ld/st; spike arrays are u8.

## 9. VM contract

Value = variant<Void, Int(i64), Float(f64), Bool, Str, Duration(ms), Group, Tensor>.
Scope stack with fresh scopes per if/while/for-iteration/handler. Handler trampoline:
`extern "C" void ndl_vm_spike_trampoline(int64_t, double, void*)` dispatching to
`VM::invokeHandler(slot, neuron, t)`. Duration arithmetic per §4. VM calls
`ndl_rt_set_seed(seed)` then (if PTX present) `ndl_rt_gpu_load_ptx`, then executes
modules in order.

## 10. CLI contract

Commands: init | check | build | run | clean | help | --version.
Flags: --emit-llvm --emit-ptx --emit-tokens --emit-ast --opt=<0..3> --triple=<t>
--seed=<n> --device=<cpu|gpu> --native --vm --project=<dir> --no-color --verbose.
Project resolution: input .ndl → walk up for ndl.toml; dir → manifest inside;
defaults otherwise. Artifacts → `<root>/build/`, name from manifest.
Import search order: dir(importer)/P → projectRoot/P → $NDLC_STD/P → <exe>/../std/P.
Cargo-style status output on stderr; program stdout stays clean.
Exit codes: 0 ok, 1 diagnostics, 64 usage, 70 runtime panic.

## 11. Build

Makefile: `make -j$(nproc)` → `dist/bin/ndlc` + `dist/lib/libndl_rt.a` +
`dist/std/stdlib/math.ndl`. Flags: `-std=c++20 -O2 -Wall -Wextra -pthread`;
ndlc links `-lpthread -ldl -lm`. NOTE: runtime object files depend on BOTH
`runtime/ndl_rt.h` and `runtime/ndl_rt_internal.h` (struct layout drift guard).
`make test` runs `ndlc check` over the reference program and examples.
CMakeLists.txt provided for IDE integration.

---

# NDL v2.0 — Asynchronous continuous-time environment (contract addendum)

v2.0 extends v1.0 additively. A program is EITHER discrete-time (`run`) OR
continuous-time (`run_continuous`) — mixing is E0215. New error codes:
E0215 (run/run_continuous mix), E0216 (structural op on a missing connection).

## 2.1 New tokens
`signal oscillator external stream run_continuous stop_continuous wait_continuous
set_plasticity prune_weights bind_input_stream` keywords; `FreqLit`
(`<num>Hz`, value in hertz). `set` (v1-reserved) is now assignment sugar:
`set IDENT = expr;` ≡ `IDENT = expr;`.

## 3.1 Grammar additions (top level unless noted)
```
signal          := 'signal' IDENT [':' 'float'] '=' expr ';'
oscillator      := 'oscillator' IDENT '(' (frequency=expr, amplitude=expr,
                   target=IDENT [, phase=expr]) ')' ';'
ext_stream      := 'external' 'stream' IDENT '[' expr ']' ';'
bind_stream     := 'bind_input_stream' '(' IDENT ',' IDENT
                   [',' ('encoding' '=' Poisson | 'max_freq' '=' expr
                         | 'kick' '=' expr)] ')' ';'        (top / cont body)
run_continuous  := 'run_continuous' '(' [dt '=' expr] [',' device '=' CPU|GPU] ')'
                   block            (body: any stmts except run/run_continuous/at_emit)
stop_cont       := 'stop_continuous' ['(' ')'] ';'    (top / cont / handler / block)
wait_cont       := 'wait_continuous' ['(' ')'] ';'    (main thread only)
set_plasticity  := 'set_plasticity' '(' IDENT '->' IDENT ',' 'enabled' '=' bool ')' ';'
prune_weights   := 'prune_weights' '(' IDENT '->' IDENT ',' 'threshold' '=' expr ')' ';'
```
`configure_stdp` gains `modulator=<signal>` (3-factor rule). New builtins:
`get_membrane_potentials(Group)->Tensor[N]`,
`predict_linear(Tensor[N], Tensor[M,N])->Tensor[M]`,
`vector_l2_norm(Tensor)->Float`, `stream_set(streamName, Int, Float)->Void`.
Signals are Float values; `set <Signal> = expr;` updates the runtime M(t).

## 5.1 CPU tick order (v2.0; replaces §5 order)
(1) i_syn decay (2) propagate (3) injections (run only; times are shifted to
the global time base) **(3b) oscillators** I_osc = A·sin(2πf·t+φ)
**(3c) streams**: drain SPSC ring → Poisson-encode: neuron i spikes with
p = clamp(latest[i],0,1)·max_freq·dt contributing `kick` current
(4) LIF (5) **modulated STDP** ΔW_ji = (lr_pot·tr_pre[i]·s_post[j] −
lr_dep·tr_post[j]·s_pre[i]) · M(t) **(5b) structural commands** queued while
running are applied at tick boundaries (6) traces (7) logs + handlers
(8) prev = cur (9) `sim_time_ms += dt` (global time base).

## 6.1 New C ABI
```c
signal_create/signal_set/signal_get; stdp_set_modulator(src,dst,name);
oscillator_add(name,target,freq,amp,phase);
stream_create(name,size)->id; stream_find(name)->id;
stream_push(id,data,n)        // host thread, SPSC ring, drops on overrun
stream_set(id,index,value)    // mailbox write (NDL-side)
bind_input_stream(id,target,encoding /*0=Poisson*/, max_freq, kick);
set_plasticity(src,dst,enabled); prune_weights(src,dst,threshold);
run_continuous(dt,device); stop_continuous(); wait_continuous(); is_continuous();
get_membrane_potentials(g); predict_linear(x,w); vector_l2_norm(t);
```
Thread-safety: the continuous loop thread owns neuron/connection state;
host pushes are ring/atomic-safe; structural ops issued while running are
queued (`cmd_mx`) and applied between ticks; `stop_continuous` from a handler
(flags + detaches) never self-joins. Continuous mode runs the CPU event loop;
`device=GPU` with v2.0 simulation-time features falls back (one-time warning).

## 8.1 PTX additions
`ndl_gpu_stdp_update` gains a `.param .f64 p9` (M(t)) IFF the program uses a
modulator; `ndl_gpu_oscillator_apply(i_syn, i_osc, n)`,
`ndl_gpu_prune_dense(w, threshold, n)`, `ndl_gpu_prune_csr(vals, threshold, nnz)`
are emitted when the corresponding features exist. The v2.0 CPU runtime does
not launch them yet (see §5.1 note); they keep the blob contract complete.
