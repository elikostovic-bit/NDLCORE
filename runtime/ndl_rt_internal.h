// NDL v1.0 — libndl_rt internal shared declarations.
// NOT part of the public ABI (ndl_rt.h). Included only by runtime/*.cpp TUs
// to share simulation state, internal structs and cross-TU helpers.
#pragma once

#include "ndl_rt.h"

#include <atomic>
#include <shared_mutex>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// --- internal network model -------------------------------------------------

struct NdlGroup {
  std::string name;
  int64_t size = 0;
  int32_t ntype = 0;  // 0 = Excitatory, 1 = Inhibitory
  double tau = 0.0;
  double threshold = 0.0;
  double rest = 0.0;
  double reset = 0.0;
  double window_ms = 20.0;  // trace window; updated by configure_stdp
  std::vector<double> v;    // membrane potentials (init = rest)
  std::vector<double> i_syn;  // synaptic + injected current per tick (decays with tau_syn)
  std::vector<double> hold;   // sustained step currents from `at .. emit` (per run)
  std::vector<double> trace;  // eligibility traces
  std::vector<uint8_t> spikes;        // current-tick spikes
  std::vector<uint8_t> prev_spikes;   // previous-tick spikes
  // Per-neuron pcg32 states for membrane noise (ndl_ocl_kernels.cl uses the
  // SAME algorithm: with equal seeds CPU and GPU noise match exactly).
  // Not part of the checkpoint; re-seeded from the global RNG on (re)creation.
  std::vector<uint64_t> noise_rng;
  std::vector<std::pair<double, int64_t>> spike_log;  // (t_ms, neuron)
  struct Handler {
    NdlSpikeHandler fn;
    void* user;
  };
  std::vector<Handler> handlers;
};

enum ConnKind : int32_t {
  CONN_DENSE = 0,
  CONN_SPARSE = 1,
  CONN_ONE2ONE = 2,
};

struct Connection {
  int32_t kind = CONN_DENSE;
  NdlGroup* src = nullptr;
  NdlGroup* dst = nullptr;
  bool plastic = false;
  // dense: w[j * n_src + i] (dst-major); one2one: w[i]
  std::vector<double> w;
  // sparse CSR over dst rows j (only for CONN_SPARSE)
  std::vector<uint64_t> rowptr;  // size n_dst + 1
  std::vector<uint32_t> cols;
  std::vector<double> vals;
  // Ф9.4b grow-on-demand: how many synapses were added by grow_synapses
  // (runtime-only diagnostic, not part of the checkpoint; counts since the
  // connection was created/loaded).
  std::atomic<long long> grownTotal{0};
};

struct StdpCfg {
  NdlGroup* src = nullptr;
  NdlGroup* dst = nullptr;
  double lr_pot = 0.0;
  double lr_dep = 0.0;
  double window_ms = 20.0;
  // v2.0 — 3-factor rule: the whole update is multiplied by M(t), the current
  // value of the named signal (empty = unmodulated, M = 1).
  std::string modulator;
};

struct Injection {
  NdlGroup* g = nullptr;
  int64_t lo = 0;
  int64_t hi = 0;
  double t = 0.0;    // ms, relative to the start of the enclosing run()
  double cur = 0.0;  // injected current
};

// --- v2.0 state ---------------------------------------------------------------

struct NdlSignal {
  std::string name;
  std::atomic<double> value{0.0};
};

struct NdlOscillator {
  std::string name;
  NdlGroup* target = nullptr;
  double freq_hz = 0.0;
  double amplitude = 0.0;
  double phase = 0.0;
};

// External stream: a fixed-size vector register. Two write paths converge on
// `latest` (guarded by `mx`): the NDL-side mailbox write (ndl_rt_stream_set)
// and the host-side SPSC ring (ndl_rt_stream_push) drained by the tick loop.
struct NdlStream {
  std::string name;
  int64_t size = 0;
  std::vector<double> latest;         // decoded snapshot fed to the bound group
  mutable std::mutex mx;
  // lock-free SPSC ring of pushed samples (host thread = producer)
  std::vector<double> ring;
  static constexpr int64_t kRingCap = 64;  // samples
  std::atomic<int64_t> wpos{0};
  std::atomic<int64_t> rpos{0};
  // binding
  NdlGroup* target = nullptr;
  int32_t encoding = 0;  // 0 = Poisson
  double max_freq = 0.0;
  double kick = 1.0;     // synaptic current injected per Poisson spike
};

// Structural command issued while the continuous loop runs; applied between
// ticks on the simulation thread (avoids data races without long-held locks).
struct StructCmd {
  enum class K { SetPlasticity, Prune, Normalize, Grow } k = K::SetPlasticity;
  NdlGroup* src = nullptr;
  NdlGroup* dst = nullptr;
  bool enabled = false;   // SetPlasticity
  double threshold = 0.0; // Prune
  double target_sum = 0.0; // Normalize (homeostatic synaptic scaling)
  // Grow (Ф9.4b): grow every (pre x post) synapse that is not wired yet.
  // Shared pointers keep the queued copies cheap and immutable.
  std::shared_ptr<std::vector<int64_t>> pre_idx, post_idx;
  double weight = 0.0;
};

struct SimState {
  std::vector<NdlGroup*> groups;
  std::vector<Connection*> conns;
  std::vector<StdpCfg*> stdps;
  std::vector<Injection> injections;  // pending; cleared at the end of run()
  std::string gpu_ptx;  // PTX source registered via ndl_rt_gpu_load_ptx
  bool warned_no_cuda = false;  // CPU-fallback warning printed at most once
  int64_t run_index = 0;        // number of completed ndl_rt_run calls
  int64_t topology_version = 0;  // bumped on every structural change (GPU resync)
  bool gpu_requested = false;   // device==1 on the current episode (run_continuous)

  // v2.0 — asynchronous continuous-time environment
  std::vector<NdlSignal*> signals;
  std::vector<NdlOscillator*> oscillators;
  std::vector<NdlStream*> streams;
  std::atomic<double> sim_time_ms{0.0};  // global time base (oscillator phase);
                                         // atomic: the continuous loop writes it
                                         // every tick while host threads read it
                                         // (ingestSleep SIMTIME polls it — torn
                                         // non-atomic reads gave garbage windows)
  std::atomic<bool> cont_running{false};
  std::thread cont_thread;
  std::mutex cmd_mx;               // guards cmds
  std::vector<StructCmd> cmds;
  // Ф9.4f (fix): CSR-buffer guard. Tick bodies (prop/LIF/STDP/trace, incl. the
  // pool workers they fan out to) hold SHARED for the whole tick; structural
  // mutations that swap/rebuild CSR buffers (grow/prune/normalize/plasticity/
  // load_checkpoint — applied either between ticks by drain_commands or
  // directly when the loop is off) hold UNIQUE. Without this, apply_grow_direct
  // swapped cols/vals while pool workers were still dereferencing them →
  // garbage cols[k] → spikes[cols[k]] segfault at 12800 (observed twice).
  // Readers of the live CSR on other threads (row_hits, save_checkpoint) hold
  // SHARED so a swap cannot free a buffer under them.
  std::shared_mutex struct_mx;

  // Ф9.4f-debug: эпоха CSR (bump на каждом структурном свапе) + снимок на
  // границах тика. NDL_CSRDEBUG=1 печатает события и ловит свап посреди тика.
  std::atomic<uint64_t> csr_epoch{0};
  uint64_t csr_tick_epoch = 0;
};

// NDL_CSRDEBUG=1 — диагностика гонки CSR (временная, Ф9.4f).
bool ndl_rt_csr_debug();
void ndl_rt_csr_bump(const char* op, long long a, long long b);
uint64_t ndl_rt_csr_epoch_now();
void ndl_rt_csr_diag();

// Defined in ndl_rt_network.cpp.
SimState& ndl_rt_sim();

// --- shared helpers ---------------------------------------------------------

// printf-style formatter (defined in ndl_rt.cpp).
std::string ndl_rt_sfmt(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

// Bounded spike-log append (defined in ndl_rt_network.cpp). Every spike of
// every group goes through here; when the log exceeds NDL_SPIKELOG_MAX
// entries (default 4M per group) the OLDEST HALF is dropped — the log stays
// a diagnostic ring, not an unbounded leak on long-lived daemons. The raster
// export (ndl_rt_export_raster) therefore shows the most recent window.
void ndl_rt_log_spike(NdlGroup* g, double t_ms, int64_t neuron);

// RNG access (defined in ndl_rt.cpp; xoshiro256**).
uint64_t ndl_rt_rng_next_u64();
double ndl_rt_rng_uniform01();

// Effective trace window for a group: window of the FIRST StdpCfg mentioning
// the group, otherwise the group's own window (default 20 ms).
inline double ndl_rt_effective_window(const SimState& S, const NdlGroup* g) {
  for (const StdpCfg* c : S.stdps)
    if (c->src == g || c->dst == g) return c->window_ms;
  return g->window_ms;
}

inline const StdpCfg* ndl_rt_find_stdp(const SimState& S, const NdlGroup* src,
                                       const NdlGroup* dst) {
  for (const StdpCfg* c : S.stdps)
    if (c->src == src && c->dst == dst) return c;
  return nullptr;
}

// True when the program uses v2.0 simulation-time features that the GPU
// executor does not implement (signals, oscillators, streams, modulated STDP).
inline bool ndl_rt_has_v2_sim_features(const SimState& S) {
  if (!S.signals.empty() || !S.oscillators.empty() || !S.streams.empty())
    return true;
  for (const StdpCfg* c : S.stdps)
    if (!c->modulator.empty()) return true;
  return false;
}

// --- CPU tick (ndl_rt_network.cpp) -------------------------------------------
// One simulation tick at dt_ms over the shared SimState: synaptic decay,
// propagation, injections (when a cursor is given), oscillators, streams,
// LIF step, modulated STDP, traces, spike logs + handlers. Advances
// S.sim_time_ms by dt_ms. `decay`/`synDecay` come from ndl_rt_prepare_ticks.
void ndl_rt_prepare_ticks(double dt_ms, std::vector<double>& decay, double& synDecay);
void ndl_rt_tick_cpu(double dt_ms, const std::vector<double>& decay, double synDecay,
                     bool applyInjections, int64_t& inj_idx, int64_t n_inj);

// Direct (immediate) structural operations — defined in ndl_rt_network.cpp,
// exposed for the not-running fast path of ndl_rt_set_plasticity/prune_weights.
void ndl_rt_apply_set_plasticity_direct(NdlGroup* src, NdlGroup* dst, bool enabled);
void ndl_rt_apply_prune_direct(NdlGroup* src, NdlGroup* dst, double threshold);
void ndl_rt_apply_normalize_direct(NdlGroup* src, NdlGroup* dst, double target_sum);

// Ф9.4b grow-on-demand (defined in ndl_rt_network.cpp): for the SPARSE
// connection src->dst add every (pre × post) synapse of the cross product
// that is not wired yet, at `weight`. Rows stay sorted; duplicates (in the
// batch or against existing wiring) are skipped. Returns the number of
// synapses actually added. bumpTopo=false is used by the device wrapper,
// which swaps the CSR buffers in place instead of forcing a re-attach.
int64_t ndl_rt_apply_grow_direct(NdlGroup* src, NdlGroup* dst,
                                 const int64_t* pre, int64_t n_pre,
                                 const int64_t* post, int64_t n_post,
                                 double weight, bool bumpTopo);

// --- scheduler (ndl_rt_sched.cpp) -------------------------------------------
// Deterministic parallel for: the range is partitioned deterministically, each
// chunk is disjoint; body(lo, hi, ctx) must only touch data belonging to its
// sub-range. body may run on worker threads.
void ndl_rt_parallel_for(int64_t begin, int64_t end, int64_t minGrain,
                         void (*body)(int64_t lo, int64_t hi, void* ctx),
                         void* ctx);

// --- GPU (ndl_rt_gpu.cpp) ---------------------------------------------------
// Attempts to execute `ticks` ticks of dt_ms on the GPU using the registered
// PTX. Returns false (without mutating simulation state) if the driver or PTX
// is unavailable, so the caller can fall back to the CPU path.
bool ndl_rt_gpu_run(int64_t ticks, double dt_ms);
void ndl_rt_warn_no_cuda();  // prints the fallback warning at most once

// --- GPU backend dispatch (ndl_rt_backend.cpp) ------------------------------
// Phase 2: the built-in simulation can run on a device. Selection order:
// CUDA (existing PTX path, pure v1.0 networks only) → OpenCL (hand-written
// kernels incl. all v2.0 features) → CPU. A device failure mid-run is
// reported as -1 AFTER a best-effort sync of device state back to the host
// mirrors, so the caller simply continues on the CPU path (migration on the
// fly).
//   returns  1 — executed on device
//            0 — refused before any mutation (state untouched → run CPU)
//           -1 — device failure mid-run (partial tick may have executed;
//                host mirrors carry the last synced state)
int ndl_rt_gpu_sim_execute(int64_t ticks, double dt_ms, bool continuous,
                           bool apply_injections, const std::vector<double>& decay,
                           double synDecay, std::string& err);
// Name of the backend that currently owns the simulation state
// ("cuda"/"opencl"), or nullptr when the CPU owns it (for :stats).
const char* ndl_rt_gpu_active_backend();
// Forget all device buffers (topology swap / shutdown). Safe to call anytime.
void ndl_rt_gpu_backend_invalidate();
// Structural hooks: while a device owns the weights, prune/normalize must run
// on the device (or through download-modify-upload). Called by
// drain_commands() and the direct structural ops; they route to the device
// when attached, otherwise to the host implementations.
void ndl_rt_struct_prune(NdlGroup* src, NdlGroup* dst, double threshold);
void ndl_rt_struct_normalize(NdlGroup* src, NdlGroup* dst, double target_sum);
// Ф9.4b: grow routes through the device the same way (download CSR → host
// merge → in-place buffer replacement; a re-attach would discard live
// membrane/trace state, so the buffers are swapped individually).
// Returns the added count on the host path; 0 on the device path (the count
// lands in Connection::grownTotal).
int64_t ndl_rt_struct_grow(NdlGroup* src, NdlGroup* dst, const int64_t* pre,
                           int64_t n_pre, const int64_t* post, int64_t n_post,
                           double weight);
// Download device state (v, trace, weights) into the host mirrors. No-op when
// the CPU owns the state. Used by checkpoint save and getters.
void ndl_rt_gpu_sync_down();
// Download just one group's membrane potentials (for the tensor getter).
void ndl_rt_gpu_sync_group_v(NdlGroup* g);

// --- internal state swap (ndl_rt_network.cpp) -------------------------------
// Swap the global SimState with a fresh empty one (benchmarks, :gputest on
// synthetic networks). The caller MUST restore with ndl_rt_state_restore.
// Both calls invalidate device buffers and assert the continuous loop is off.
SimState* ndl_rt_state_swap_fresh();
void ndl_rt_state_restore(SimState* old);

// Exported internals used by the GPU backend (defined in ndl_rt_network.cpp).
void ndl_rt_drain_commands();                      // structural cmds, between ticks
double ndl_rt_stdp_mod_value(const StdpCfg* cfg);  // M(t) of a STDP config
// CPU tick (exported so the backend migration path can finish a tick on CPU).
// (already declared above as ndl_rt_tick_cpu)
