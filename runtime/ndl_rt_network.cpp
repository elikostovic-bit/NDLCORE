// NDL v1.0 — runtime network layer: groups, connections, STDP, injections,
// CPU run loop, spike handlers/logs, weight introspection. GPU execution is
// delegated to ndl_rt_gpu.cpp; checkpoint IO to ndl_rt_io.cpp.
//
// Synapse model: exponentially decaying postsynaptic current,
//   I_syn ← I_syn·exp(−dt/tau_syn) + Σ W·s_prev + hold ,   tau_syn = 5 ms.
// Injections (`at T emit G[..](current = C)`) are SUSTAINED step currents:
// from T until the end of the run (or until another emit on the same neurons
// overwrites them).
//
// CPU tick order (INTERNALS §5):
//   (1) i_syn *= syn_decay             (2) propagate per connection
//   (3) holds ← injections in [t,t+dt) (4) i_syn += hold
//   (5) LIF step                       (6) STDP per plastic connection
//   (7) trace decay                    (8) spike logs + handlers
//   (9) prev_spikes = spikes
#include "ndl_rt_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// --- state -------------------------------------------------------------------

namespace {
// Heap-anchored global state; swappable for :bench gpu / :gputest synthetic
// networks (ndl_rt_state_swap_fresh / ndl_rt_state_restore).
SimState* g_sim_state = nullptr;
void set_sim_state(SimState* s) { g_sim_state = s; }
}  // namespace

SimState& ndl_rt_sim() {
  if (!g_sim_state) g_sim_state = new SimState();
  return *g_sim_state;
}

namespace {

SimState& S() { return ndl_rt_sim(); }

Connection* find_conn(NdlGroup* src, NdlGroup* dst) {
  for (Connection* c : S().conns)
    if (c->src == src && c->dst == dst) return c;
  return nullptr;
}

void drop_conns(NdlGroup* src, NdlGroup* dst) {
  std::vector<Connection*>& cs = S().conns;
  for (size_t k = cs.size(); k-- > 0;) {
    if (cs[k]->src == src && cs[k]->dst == dst) {
      delete cs[k];
      cs.erase(cs.begin() + (long)k);
    }
  }
}

double draw_weight(int32_t dist_kind, double p0, double p1) {
  switch (dist_kind) {
    case 0:  // constant
      return p0;
    case 1:  // gaussian(mu, sigma)
      return ndl_rt_random_gaussian(p0, p1);
    case 2:  // uniform(a, b)
      return ndl_rt_random_uniform(p0, p1);
    default:
      ndl_rt_panic("dense_connect: unknown distribution kind");
  }
  return 0.0;  // unreachable
}

// --- parallel bodies (disjoint [lo, hi) sub-ranges; deterministic) -----------

constexpr int64_t kDenseWorkThreshold = 1 << 20;  // n_dst*n_src for O(n_src*n_dst) work
constexpr int64_t kLinearGrain = 4096;            // grain for O(n) parallel loops
constexpr double kTauSynMs = 5.0;                 // synaptic current time constant

struct PropDenseCtx {
  const double* w;
  const uint8_t* pre;
  double* isyn;
  int64_t n_src;
};
void prop_dense_body(int64_t lo, int64_t hi, void* p) {
  PropDenseCtx* c = (PropDenseCtx*)p;
  for (int64_t j = lo; j < hi; ++j) {
    const double* row = c->w + j * c->n_src;
    double acc = 0.0;
    for (int64_t i = 0; i < c->n_src; ++i) acc += row[i] * (double)c->pre[i];
    c->isyn[j] += acc;
  }
}

struct PropSparseCtx {
  const uint64_t* rowptr;
  const uint32_t* cols;
  const double* vals;
  const uint8_t* pre;
  double* isyn;
};
void prop_sparse_body(int64_t lo, int64_t hi, void* p) {
  PropSparseCtx* c = (PropSparseCtx*)p;
  for (int64_t j = lo; j < hi; ++j) {
    double acc = 0.0;
    for (uint64_t k = c->rowptr[j]; k < c->rowptr[j + 1]; ++k)
      acc += c->vals[k] * (double)c->pre[c->cols[k]];
    c->isyn[j] += acc;
  }
}

struct PropO2OCtx {
  const double* w;
  const uint8_t* pre;
  double* isyn;
};
void prop_o2o_body(int64_t lo, int64_t hi, void* p) {
  PropO2OCtx* c = (PropO2OCtx*)p;
  for (int64_t i = lo; i < hi; ++i) c->isyn[i] += c->w[i] * (double)c->pre[i];
}

struct LifCtx {
  double* v;
  const double* isyn;   // synaptic (decaying) component
  const double* hold;   // sustained step-current component
  uint8_t* spikes;
  double dt, tau, threshold, rest, reset;
  double noise_sigma;          // 0 = off (default)
  uint64_t* noise_rng;         // per-neuron pcg32 states (same alg as GPU)
  double* vpre;                // Ф9.4c k-WTA: pre-reset potentials (null = off)
};
// pcg32 — the EXACT algorithm of ndl_ocl_kernels.cl (equal seeds → equal
// noise sequences on CPU and GPU).
static inline uint32_t ndl_host_pcg32(uint64_t* st) {
  const uint64_t s = *st;
  *st = s * 6364136223846793005ULL + 1442695040888963407ULL;
  const uint32_t xorshifted = (uint32_t)(((s >> 18u) ^ s) >> 27u);
  const uint32_t rot = (uint32_t)(s >> 59u);
  return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
}
static inline double ndl_host_uniform01(uint64_t* st) {
  return (double)ndl_host_pcg32(st) * (1.0 / 4294967296.0);
}
static inline double ndl_host_gauss(uint64_t* st) {
  double u1 = ndl_host_uniform01(st);
  const double u2 = ndl_host_uniform01(st);
  if (u1 < 1e-300) u1 = 1e-300;
  return std::sqrt(-2.0 * std::log(u1)) *
         std::cos(6.283185307179586476925286766559 * u2);
}
// Membrane noise sigma (env NDL_LIF_NOISE, default 0 = model unchanged).
static double lif_noise_sigma() {
  static const double s = []() {
    const char* e = std::getenv("NDL_LIF_NOISE");
    if (!e) return 0.0;
    const double v = std::atof(e);
    return (std::isfinite(v) && v > 0.0) ? v : 0.0;
  }();
  return s;
}
void lif_body(int64_t lo, int64_t hi, void* p) {
  LifCtx* c = (LifCtx*)p;
  const double sigma = c->noise_sigma;
  const double amp = sigma > 0.0 ? sigma * std::sqrt(c->dt) : 0.0;
  double* vpre = c->vpre;
  for (int64_t i = lo; i < hi; ++i) {
    double nv =
        c->v[i] + c->dt * (c->rest - c->v[i] + c->isyn[i] + c->hold[i]) / c->tau;
    if (amp > 0.0) nv += amp * ndl_host_gauss(&c->noise_rng[i]);
    if (vpre) vpre[i] = nv;  // Ф9.4c: potential BEFORE the reset decision
    const uint8_t s = nv >= c->threshold ? 1 : 0;
    c->v[i] = s ? c->reset : nv;
    c->spikes[i] = s;
  }
}

// Safety bound for plastic weights. Normal NDL programs live well inside
// +-100; the bound only engages on pathological accumulation (e.g. a modulated
// 3-factor rule driven for very long episodes) and keeps networks sane.
static inline double clampW(double w) {
  return w < -100.0 ? -100.0 : (w > 100.0 ? 100.0 : w);
}

struct StdpDenseCtx {
  double* w;
  const double* pre_trace;
  const uint8_t* post_spikes;
  const double* post_trace;
  const uint8_t* pre_prev;
  double lr_pot, lr_dep;
  double mod;  // v2.0: M(t) — modulator value (1.0 = unmodulated)
  int64_t n_src;
};
void stdp_dense_body(int64_t lo, int64_t hi, void* p) {
  StdpDenseCtx* c = (StdpDenseCtx*)p;
  for (int64_t j = lo; j < hi; ++j) {
    double* row = c->w + j * c->n_src;
    const double pot = c->lr_pot * (double)c->post_spikes[j];
    const double dep = c->lr_dep * (double)c->post_trace[j];
    // v2.0 3-factor rule: dW = (A+ * tr_pre * s_post - A- * tr_post * s_pre) * M(t)
    for (int64_t i = 0; i < c->n_src; ++i)
      row[i] = clampW(row[i] + c->mod * (pot * c->pre_trace[i] - dep * (double)c->pre_prev[i]));
  }
}

struct StdpSparseCtx {
  const uint64_t* rowptr;
  const uint32_t* cols;
  double* vals;
  const double* pre_trace;
  const uint8_t* post_spikes;
  const double* post_trace;
  const uint8_t* pre_prev;
  double lr_pot, lr_dep;
  double mod;  // v2.0: M(t)
};
void stdp_sparse_body(int64_t lo, int64_t hi, void* p) {
  StdpSparseCtx* c = (StdpSparseCtx*)p;
  for (int64_t j = lo; j < hi; ++j) {
    const double pot = c->lr_pot * (double)c->post_spikes[j];
    const double post_trace_j = c->post_trace[j];
    for (uint64_t k = c->rowptr[j]; k < c->rowptr[j + 1]; ++k) {
      const uint32_t i = c->cols[k];
      c->vals[k] = clampW(c->vals[k] +
          c->mod * (pot * c->pre_trace[i] - c->lr_dep * post_trace_j * (double)c->pre_prev[i]));
    }
  }
}

struct StdpO2OCtx {
  double* w;
  const double* pre_trace;
  const uint8_t* post_spikes;
  const double* post_trace;
  const uint8_t* pre_prev;
  double lr_pot, lr_dep;
  double mod;  // v2.0: M(t)
};
void stdp_o2o_body(int64_t lo, int64_t hi, void* p) {
  StdpO2OCtx* c = (StdpO2OCtx*)p;
  for (int64_t i = lo; i < hi; ++i)
    c->w[i] = clampW(c->w[i] + c->mod * (c->lr_pot * c->pre_trace[i] * (double)c->post_spikes[i] -
                         c->lr_dep * c->post_trace[i] * (double)c->pre_prev[i]));
}

struct TraceCtx {
  double* trace;
  const uint8_t* spikes;
  double decay;
};
void trace_body(int64_t lo, int64_t hi, void* p) {
  TraceCtx* c = (TraceCtx*)p;
  for (int64_t i = lo; i < hi; ++i) c->trace[i] = c->trace[i] * c->decay + (double)c->spikes[i];
}

void run_parallel(int64_t begin, int64_t end, int64_t grain,
                  void (*body)(int64_t, int64_t, void*), void* ctx, bool parallel) {
  if (!parallel) {
    body(begin, end, ctx);
    return;
  }
  ndl_rt_parallel_for(begin, end, grain, body, ctx);
}

// --- Ф9.4c: hard k-WTA on CoreE groups (CPU path, env NDL_KWTA=<fraction>) ---
// Raster diagnosis (Ф9.4b): at a large core the recall window catches ~14% of
// CoreE firing while the pair signal needs only ~3% — the wide state is the
// noise ∝E that reaches every code row through ALL synapses. The soft E/I loop
// fails to clamp it (inhibitory saturation). k-WTA is the contrast clamp: at
// most frac·E neurons may emit a spike per tick; the weakest by PRE-RESET
// potential are suppressed (spike withheld, membrane reset — electrically they
// fired, but nothing propagates, nothing potentiates, nothing logs). Legacy
// runs are untouched: the env default is 0 = the pass is skipped entirely.
// Determinism: ties break by the lower index; the pass is CPU-only and the
// host pins NDL_DEVICE=cpu when the arm is on (GPU kernels stay untouched).
static double kwta_frac() {
  static const double v = []() {
    const char* e = std::getenv("NDL_KWTA");
    if (!e) return 0.0;
    const double x = std::atof(e);
    return (std::isfinite(x) && x > 0.0 && x <= 1.0) ? x : 0.0;
  }();
  return v;
}
// scratch for pre-reset potentials (single sim thread; lif_body writes
// disjoint indices from worker threads, kwta reads after the barrier)
std::vector<double> g_kwta_vpre;
void apply_kwta(NdlGroup* g) {
  const double frac = kwta_frac();
  if (!(frac > 0.0) || g->size < 16) return;
  if (g->name.compare(0, 5, "CoreE") != 0) return;  // CoreE, CoreE2, ...
  int64_t count = 0;
  for (int64_t i = 0; i < g->size; ++i) count += g->spikes[(size_t)i];
  const int64_t K = (int64_t)(frac * (double)g->size);
  if (K <= 0 || count <= K) return;
  static std::vector<std::pair<double, int64_t>> cand;
  cand.clear();
  cand.reserve((size_t)count);
  for (int64_t i = 0; i < g->size; ++i)
    if (g->spikes[(size_t)i])
      cand.emplace_back(g_kwta_vpre[(size_t)i], i);
  const size_t Ks = (size_t)K;
  if (Ks >= cand.size()) return;
  std::partial_sort(cand.begin(), cand.begin() + Ks, cand.end(),
                    [](const std::pair<double, int64_t>& a,
                       const std::pair<double, int64_t>& b) {
                      return a.first != b.first ? a.first > b.first
                                                : a.second < b.second;
                    });
  for (size_t r = Ks; r < cand.size(); ++r) {
    const int64_t i = cand[r].second;
    g->spikes[(size_t)i] = 0;
    g->v[(size_t)i] = g->reset;  // suppressed: fired electrically, emitted nothing
  }
}

void apply_injection(const Injection& inj) {
  NdlGroup* g = inj.g;
  int64_t lo = inj.lo < 0 ? 0 : inj.lo;
  int64_t hi = inj.hi > g->size ? g->size : inj.hi;
  for (int64_t k = lo; k < hi; ++k) g->hold[(size_t)k] = inj.cur;  // step: assign
}

// --- v2.0 helpers ------------------------------------------------------------

// M(t) for a STDP config: current value of its modulator signal, 1.0 if none.
double stdp_modulator_value(const StdpCfg* cfg) {
  if (!cfg || cfg->modulator.empty()) return 1.0;
  SimState& st = S();
  for (const NdlSignal* sig : st.signals)
    if (sig->name == cfg->modulator)
      return sig->value.load(std::memory_order_relaxed);
  return 1.0; // unknown signal: learn as if unmodulated
}

// Oscillator phase: I_osc = A * sin(2*pi*f*t + phi) into every neuron of the
// target group (i_syn accumulates the same way injected currents do).
void apply_oscillators(double t_ms) {
  SimState& st = S();
  for (const NdlOscillator* o : st.oscillators) {
    if (!o->target || o->amplitude == 0.0) continue;
    const double w = 2.0 * 3.14159265358979323846 * o->freq_hz;
    const double i_osc = o->amplitude * std::sin(w * (t_ms / 1000.0) + o->phase);
    NdlGroup* g = o->target;
    for (int64_t k = 0; k < g->size; ++k) g->i_syn[(size_t)k] += i_osc;
  }
}

// Stream phase: drain pushed samples into the latest vector, then Poisson-encode:
// neuron i spikes with p = clamp(latest[i], 0, 1) * max_freq * dt and contributes
// a unit synaptic kick when it fires.
void apply_streams(double dt_ms) {
  SimState& st = S();
  for (NdlStream* s : st.streams) {
    if (!s->target) continue;
    // 1) drain the SPSC ring (host pushes) into latest[]
    {
      std::lock_guard<std::mutex> lk(s->mx);
      int64_t r = s->rpos.load(std::memory_order_acquire);
      const int64_t w = s->wpos.load(std::memory_order_acquire);
      while (r != w) {
        const int64_t slot = (r % NdlStream::kRingCap) * s->size;
        std::memcpy(s->latest.data(), &s->ring[(size_t)slot], sizeof(double) * (size_t)s->size);
        ++r;
      }
      s->rpos.store(r, std::memory_order_release);
    }
    // 2) Poisson encode into the bound group
    if (s->max_freq <= 0.0 || s->encoding != 0) continue;
    NdlGroup* g = s->target;
    const int64_t n = std::min(g->size, s->size);
    for (int64_t k = 0; k < n; ++k) {
      double x = s->latest[(size_t)k];
      if (x <= 0.0) continue;
      if (x > 1.0) x = 1.0;
      const double p = x * s->max_freq * dt_ms / 1000.0;  // Hz -> per-tick probability
      if (p > 0.0 && ndl_rt_rng_uniform01() < p) g->i_syn[(size_t)k] += s->kick;
    }
  }
}

// Structural commands (set_plasticity / prune) issued while the continuous
// loop runs; applied here on the simulation thread, between ticks.
void apply_set_plasticity(NdlGroup* src, NdlGroup* dst, bool enabled);
void apply_prune(NdlGroup* src, NdlGroup* dst, double threshold);
void apply_normalize(NdlGroup* src, NdlGroup* dst, double target_sum);

void drain_commands() {
  SimState& st = S();
  std::vector<StructCmd> batch;
  {
    std::lock_guard<std::mutex> lk(st.cmd_mx);
    batch.swap(st.cmds);
  }
  for (const StructCmd& c : batch) {
    if (c.k == StructCmd::K::SetPlasticity)
      apply_set_plasticity(c.src, c.dst, c.enabled);
    else if (c.k == StructCmd::K::Prune)
      ndl_rt_struct_prune(c.src, c.dst, c.threshold);
    else if (c.k == StructCmd::K::Grow) {
      // Ф9.4b grow-on-demand: the batch was deep-copied into the command, so
      // the index arrays are owned here and safe to touch on the sim thread.
      if (c.pre_idx && c.post_idx && !c.pre_idx->empty() && !c.post_idx->empty())
        ndl_rt_struct_grow(c.src, c.dst, c.pre_idx->data(),
                           (int64_t)c.pre_idx->size(), c.post_idx->data(),
                           (int64_t)c.post_idx->size(), c.weight);
    } else
      ndl_rt_struct_normalize(c.src, c.dst, c.target_sum);
  }
}

// v2.0 structural plasticity — direct application (caller decides queue vs now).
// Ф9.4f: every structural leaf takes UNIQUE — see SimState::struct_mx.
void apply_set_plasticity(NdlGroup* src, NdlGroup* dst, bool enabled) {
  std::unique_lock<std::shared_mutex> lk(S().struct_mx);
  Connection* c = find_conn(src, dst);
  if (!c)
    ndl_rt_panic(ndl_rt_sfmt("set_plasticity: no connection '%s' -> '%s'",
                             src->name.c_str(), dst->name.c_str())
                     .c_str());
  c->plastic = enabled;
}

void apply_prune(NdlGroup* src, NdlGroup* dst, double threshold) {
  std::unique_lock<std::shared_mutex> lk(S().struct_mx);
  Connection* c = find_conn(src, dst);
  if (!c)
    ndl_rt_panic(ndl_rt_sfmt("prune_weights: no connection '%s' -> '%s'",
                             src->name.c_str(), dst->name.c_str())
                     .c_str());
  if (!(threshold > 0.0)) return;
  switch (c->kind) {
    case CONN_DENSE:
      for (double& w : c->w)
        if (std::fabs(w) < threshold) w = 0.0;
      break;
    case CONN_ONE2ONE:
      for (double& w : c->w)
        if (std::fabs(w) < threshold) w = 0.0;
      break;
    case CONN_SPARSE: {
      // Rebuild the CSR rows in place, dropping pruned synapses.
      size_t out = 0;
      for (int64_t j = 0; j < dst->size; ++j) {
        const uint64_t begin = c->rowptr[(size_t)j];
        const uint64_t end = c->rowptr[(size_t)j + 1];
        c->rowptr[(size_t)j] = out;
        for (uint64_t k = begin; k < end; ++k) {
          if (std::fabs(c->vals[(size_t)k]) >= threshold) {
            c->cols[(size_t)out] = c->cols[(size_t)k];
            c->vals[(size_t)out] = c->vals[(size_t)k];
            ++out;
          }
        }
      }
      c->rowptr[(size_t)dst->size] = out;
      const long long before = (long long)c->cols.size();
      c->cols.resize(out);
      c->vals.resize(out);
      ndl_rt_csr_bump("prune", before, (long long)out);
      break;
    }
  }
}

// Homeostatic synaptic scaling (shrink-only): for every postsynaptic neuron,
// if the sum of |w| of its incoming synapses exceeds target_sum, scale the row
// down to target_sum. Learned specific synapses survive proportionally; only
// the global drive per neuron is bounded — the standard cure against one
// pattern capturing the whole readout.
void apply_normalize(NdlGroup* src, NdlGroup* dst, double target_sum) {
  std::unique_lock<std::shared_mutex> lk(S().struct_mx);  // Ф9.4f
  Connection* c = find_conn(src, dst);
  if (!c)
    ndl_rt_panic(ndl_rt_sfmt("normalize_incoming: no connection '%s' -> '%s'",
                             src->name.c_str(), dst->name.c_str())
                     .c_str());
  if (!(target_sum > 0.0)) return;
  switch (c->kind) {
    case CONN_DENSE:
    case CONN_ONE2ONE: {
      const int64_t n_src = (c->kind == CONN_DENSE) ? src->size : 1;
      for (int64_t j = 0; j < dst->size; ++j) {
        double* row = c->w.data() + j * n_src;
        double sum = 0.0;
        for (int64_t i = 0; i < n_src; ++i) sum += std::fabs(row[i]);
        if (sum > target_sum) {
          const double f = target_sum / sum;
          for (int64_t i = 0; i < n_src; ++i) row[i] *= f;
        }
      }
      break;
    }
    case CONN_SPARSE: {
      for (int64_t j = 0; j < dst->size; ++j) {
        const uint64_t begin = c->rowptr[(size_t)j];
        const uint64_t end = c->rowptr[(size_t)j + 1];
        double sum = 0.0;
        for (uint64_t k = begin; k < end; ++k)
          sum += std::fabs(c->vals[(size_t)k]);
        if (sum > target_sum) {
          const double f = target_sum / sum;
          for (uint64_t k = begin; k < end; ++k) c->vals[(size_t)k] *= f;
        }
      }
      break;
    }
  }
}

}  // namespace

// Exports consumed by the GPU backend (defined here so they can reach the
// anonymous-namespace helpers; the public ABI is untouched).
void ndl_rt_drain_commands() { drain_commands(); }
double ndl_rt_stdp_mod_value(const StdpCfg* cfg) { return stdp_modulator_value(cfg); }

// Ф9.4b grow-on-demand: add every (pre × post) synapse of the cross product
// that is not wired yet, at `weight`. The random projection gives a taught
// read row only winners×density binding synapses — at large cores the pair
// signal dilutes ∝1/E (Ф9.4 diagnosis). Growing the missed synapses makes
// the binding cost per pair CONSTANT (winners × code rows, ~700 synapses)
// — memory becomes proportional to the number of pairs, not to E.
// Rows are rebuilt by a single merge pass (existing rows are ascending;
// duplicates inside the batch and against existing wiring are skipped), so
// the O(nnz) rebuild stays cheap even for 100k-core brains.
// Linkage note: defined OUTSIDE the anonymous namespace — the device wrapper
// (ndl_rt_ocl.cpp) and the dispatcher (ndl_rt_backend.cpp) call it directly.
int64_t ndl_rt_apply_grow_direct(NdlGroup* src, NdlGroup* dst,
                                 const int64_t* pre, int64_t n_pre,
                                 const int64_t* post, int64_t n_post,
                                 double weight, bool bumpTopo) {
  if (!src || !dst) ndl_rt_panic("grow_synapses: null group");
  // Ф9.4f: UNIQUE for the whole merge+swap — the CSR buffers are replaced
  // here, and pool workers / host readers may still be dereferencing them.
  std::unique_lock<std::shared_mutex> lk(S().struct_mx);
  Connection* c = find_conn(src, dst);
  if (!c)
    ndl_rt_panic(ndl_rt_sfmt("grow_synapses: no connection '%s' -> '%s'",
                             src->name.c_str(), dst->name.c_str())
                     .c_str());
  if (c->kind != CONN_SPARSE)
    ndl_rt_panic("grow_synapses: only sparse connections support growing");
  if (!std::isfinite(weight))
    ndl_rt_panic("grow_synapses: weight must be finite");
  if (n_pre < 0 || n_post < 0)
    ndl_rt_panic("grow_synapses: negative index count");
  if (n_pre == 0 || n_post == 0) return 0;

  // sanitize: bounds-check, sort, dedup (both index lists ascending)
  std::vector<int32_t> P;
  P.reserve((size_t)n_pre);
  for (int64_t k = 0; k < n_pre; ++k)
    if (pre[k] >= 0 && pre[k] < src->size) P.push_back((int32_t)pre[k]);
  std::vector<int32_t> Q;
  Q.reserve((size_t)n_post);
  for (int64_t k = 0; k < n_post; ++k)
    if (post[k] >= 0 && post[k] < dst->size) Q.push_back((int32_t)post[k]);
  std::sort(P.begin(), P.end());
  P.erase(std::unique(P.begin(), P.end()), P.end());
  std::sort(Q.begin(), Q.end());
  Q.erase(std::unique(Q.begin(), Q.end()), Q.end());
  if (P.empty() || Q.empty()) return 0;

  const int64_t nd = dst->size;
  std::vector<uint64_t> nrp((size_t)nd + 1, 0);
  std::vector<uint32_t> ncl;
  std::vector<double> nvl;
  ncl.reserve(c->cols.size() + P.size() * Q.size());
  nvl.reserve(c->vals.size() + P.size() * Q.size());

  int64_t added = 0;
  size_t qi = 0;  // cursor into Q (ascending)
  for (int64_t j = 0; j < nd; ++j) {
    const uint64_t begin = c->rowptr[(size_t)j];
    const uint64_t end = c->rowptr[(size_t)j + 1];
    if (qi < Q.size() && Q[qi] == (int32_t)j) {
      ++qi;
      size_t k = (size_t)begin;
      size_t p = 0;
      while (k < (size_t)end || p < P.size()) {
        if (p >= P.size() || (k < (size_t)end && c->cols[k] < (uint32_t)P[p])) {
          ncl.push_back(c->cols[k]);
          nvl.push_back(c->vals[k]);
          ++k;
        } else if (k >= (size_t)end || c->cols[k] > (uint32_t)P[p]) {
          // Ф9.4e: групповой grow на связь с самим собой (src==dst, напр.
          // winners→winners cell assembly) не должен ставить автосинапс i→i —
          // самовозбуждение нейрона бессмысленно в LIF и только шумит.
          if (c->src == c->dst && (int64_t)P[p] == j) {
            ++p;
            continue;
          }
          ncl.push_back((uint32_t)P[p]);  // grow the missing synapse
          nvl.push_back(weight);
          ++added;
          ++p;
        } else {  // already wired — keep the existing synapse
          ncl.push_back(c->cols[k]);
          nvl.push_back(c->vals[k]);
          ++k;
          ++p;
        }
      }
    } else {
      for (uint64_t k = begin; k < end; ++k) {
        ncl.push_back(c->cols[(size_t)k]);
        nvl.push_back(c->vals[(size_t)k]);
      }
    }
    nrp[(size_t)j + 1] = (uint64_t)ncl.size();
  }
  if (added > 0) {
    const long long nnzBefore = (long long)(ncl.size() - (size_t)added);
    c->rowptr.swap(nrp);
    c->cols.swap(ncl);
    c->vals.swap(nvl);
    c->grownTotal.fetch_add(added, std::memory_order_relaxed);
    ndl_rt_csr_bump("grow", nnzBefore, (long long)c->cols.size());
    if (bumpTopo) S().topology_version++;
  }
  return added;
}

// Direct structural operations (immediate application). Used by the public ABI
// wrappers when the continuous loop is not running.
void ndl_rt_apply_set_plasticity_direct(NdlGroup* src, NdlGroup* dst, bool enabled) {
  apply_set_plasticity(src, dst, enabled);
}

void ndl_rt_apply_prune_direct(NdlGroup* src, NdlGroup* dst, double threshold) {
  apply_prune(src, dst, threshold);
}
// (normalize leaf below — locked like the others)

void ndl_rt_apply_normalize_direct(NdlGroup* src, NdlGroup* dst, double target_sum) {
  apply_normalize(src, dst, target_sum);
}

// --- groups -------------------------------------------------------------------

NdlGroup* ndl_rt_group_create(int64_t size, int32_t ntype, double tau,
                              double threshold, double rest, double reset,
                              const char* name) {
  if (size <= 0) ndl_rt_panic("group_create: size must be positive");
  if (!(tau > 0.0)) ndl_rt_panic("group_create: tau must be positive");  // rejects tau=0 and NaN
  NdlGroup* g = nullptr;
  try {
    g = new NdlGroup();
    g->name = name ? name : "";
    g->size = size;
    g->ntype = ntype;
    g->tau = tau;
    g->threshold = threshold;
    g->rest = rest;
    g->reset = reset;
    g->v.assign((size_t)size, rest);
    g->i_syn.assign((size_t)size, 0.0);
    g->hold.assign((size_t)size, 0.0);
    g->trace.assign((size_t)size, 0.0);
    g->spikes.assign((size_t)size, 0);
    g->prev_spikes.assign((size_t)size, 0);
    g->noise_rng.reserve((size_t)size);
    for (int64_t k = 0; k < size; ++k) g->noise_rng.push_back(ndl_rt_rng_next_u64());
    S().groups.push_back(g);
    S().topology_version++;
  } catch (const std::bad_alloc&) {
    delete g;
    ndl_rt_panic("out of memory");
  }
  return g;
}

int64_t ndl_rt_group_size(NdlGroup* g) {
  if (!g) ndl_rt_panic("group_size: null group");
  return g->size;
}

const char* ndl_rt_group_name(NdlGroup* g) {
  if (!g) ndl_rt_panic("group_name: null group");
  return g->name.c_str();
}

// --- connections ----------------------------------------------------------------
// A new connection between the same pair replaces the previous one (same rule
// as checkpoint loading); this keeps get_weight and STDP unambiguous.

void ndl_rt_dense_connect(NdlGroup* src, NdlGroup* dst, int32_t dist_kind,
                          double p0, double p1, int32_t plastic) {
  if (!src || !dst) ndl_rt_panic("dense_connect: null group");
  Connection* c = nullptr;
  try {
    c = new Connection();
    c->kind = CONN_DENSE;
    c->src = src;
    c->dst = dst;
    c->plastic = plastic != 0;
    const int64_t ns = src->size, nd = dst->size;
    c->w.assign((size_t)(ns * nd), 0.0);
    // Deterministic sequential init, dst-major order (j outer, i inner).
    for (int64_t j = 0; j < nd; ++j)
      for (int64_t i = 0; i < ns; ++i)
        c->w[(size_t)(j * ns + i)] = draw_weight(dist_kind, p0, p1);
    drop_conns(src, dst);
    S().conns.push_back(c);
    S().topology_version++;
  } catch (const std::bad_alloc&) {
    delete c;
    ndl_rt_panic("out of memory");
  }
}

void ndl_rt_sparse_connect(NdlGroup* src, NdlGroup* dst, double density,
                           double weight, int32_t plastic) {
  if (!src || !dst) ndl_rt_panic("sparse_connect: null group");
  Connection* c = nullptr;
  try {
    c = new Connection();
    c->kind = CONN_SPARSE;
    c->src = src;
    c->dst = dst;
    c->plastic = plastic != 0;
    const int64_t ns = src->size, nd = dst->size;
    c->rowptr.assign((size_t)nd + 1, 0);
    // One uniform draw per (i, j) pair, deterministic (j outer, i inner).
    for (int64_t j = 0; j < nd; ++j) {
      for (int64_t i = 0; i < ns; ++i) {
        const double u = ndl_rt_random_uniform(0.0, 1.0);
        if (u < density) {
          c->cols.push_back((uint32_t)i);
          c->vals.push_back(weight);
        }
      }
      c->rowptr[(size_t)j + 1] = (uint64_t)c->cols.size();
    }
    drop_conns(src, dst);
    S().conns.push_back(c);
    S().topology_version++;
  } catch (const std::bad_alloc&) {
    delete c;
    ndl_rt_panic("out of memory");
  }
}

void ndl_rt_one_to_one_connect(NdlGroup* src, NdlGroup* dst, double weight,
                               int32_t plastic) {
  if (!src || !dst) ndl_rt_panic("one_to_one_connect: null group");
  if (src->size != dst->size) {
    ndl_rt_panic(ndl_rt_sfmt("one_to_one_connect: size mismatch (src %lld vs dst %lld)",
                             (long long)src->size, (long long)dst->size)
                     .c_str());
  }
  Connection* c = nullptr;
  try {
    c = new Connection();
    c->kind = CONN_ONE2ONE;
    c->src = src;
    c->dst = dst;
    c->plastic = plastic != 0;
    c->w.assign((size_t)src->size, weight);
    drop_conns(src, dst);
    S().conns.push_back(c);
    S().topology_version++;
  } catch (const std::bad_alloc&) {
    delete c;
    ndl_rt_panic("out of memory");
  }
}

// --- plasticity -----------------------------------------------------------------

void ndl_rt_configure_stdp(NdlGroup* src, NdlGroup* dst, double lr_pot,
                           double lr_dep, double window_ms) {
  if (!src || !dst) ndl_rt_panic("configure_stdp: null group");
  if (!(window_ms > 0.0)) ndl_rt_panic("configure_stdp: window_ms must be positive");
  try {
    StdpCfg* cfg = nullptr;
    for (StdpCfg* c : S().stdps)
      if (c->src == src && c->dst == dst) { cfg = c; break; }
    if (cfg) {
      cfg->lr_pot = lr_pot;
      cfg->lr_dep = lr_dep;
      cfg->window_ms = window_ms;
    } else {
      cfg = new StdpCfg();
      cfg->src = src;
      cfg->dst = dst;
      cfg->lr_pot = lr_pot;
      cfg->lr_dep = lr_dep;
      cfg->window_ms = window_ms;
      S().stdps.push_back(cfg);
    }
    src->window_ms = window_ms;
    dst->window_ms = window_ms;
  } catch (const std::bad_alloc&) {
    ndl_rt_panic("out of memory");
  }
}

// --- spike handlers ----------------------------------------------------------------

void ndl_rt_register_spike_handler(NdlGroup* g, NdlSpikeHandler fn, void* user) {
  if (!g) ndl_rt_panic("register_spike_handler: null group");
  if (!fn) ndl_rt_panic("register_spike_handler: null handler");
  try {
    NdlGroup::Handler h;
    h.fn = fn;
    h.user = user;
    g->handlers.push_back(h);
  } catch (const std::bad_alloc&) {
    ndl_rt_panic("out of memory");
  }
}

// --- injections ----------------------------------------------------------------------

void ndl_rt_schedule_inject(NdlGroup* g, int64_t lo, int64_t hi, double t_ms,
                            double current) {
  if (!g) ndl_rt_panic("schedule_inject: null group");
  try {
    Injection inj;
    inj.g = g;
    inj.lo = lo;
    inj.hi = hi;
    inj.t = t_ms;
    inj.cur = current;
    S().injections.push_back(inj);
  } catch (const std::bad_alloc&) {
    ndl_rt_panic("out of memory");
  }
}

// --- run loop -------------------------------------------------------------------------

void ndl_rt_prepare_ticks(double dt_ms, std::vector<double>& decay, double& synDecay) {
  SimState& st = S();
  const size_t nG = st.groups.size();
  decay.assign(nG, 0.0);
  for (size_t gi = 0; gi < nG; ++gi)
    decay[gi] = std::exp(-dt_ms / ndl_rt_effective_window(st, st.groups[gi]));
  synDecay = std::exp(-dt_ms / kTauSynMs);
}

// Ф9.4f-debug: учёт эпохи CSR. Каждый структурный мутант bump-ает счётчик;
// tick_cpu снимает отпечаток до тела и сверяет в конце — «свап посреди тика»
// ловится сразу, а лог событий показывает, кто свапал последним перед крашем.
bool ndl_rt_csr_debug() {
  static const bool on = [] {
    const char* e = std::getenv("NDL_CSRDEBUG");
    return e && e[0] == '1';
  }();
  return on;
}
void ndl_rt_csr_bump(const char* op, long long a, long long b) {
  SimState& st = S();
  const uint64_t ep = st.csr_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (ndl_rt_csr_debug())
    std::fprintf(stderr,
                 "[csrdbg] %s %lld->%lld tid=%zu epoch=%llu\n", op, a, b,
                 std::hash<std::thread::id>{}(std::this_thread::get_id()),
                 (unsigned long long)ep);
}
uint64_t ndl_rt_csr_epoch_now() { return S().csr_epoch.load(std::memory_order_acquire); }

// Дамп всех групп и CSR-связей: адреса буферов позволяют по регистрам краха
// (rdi=cols, r8=psp, rdx=ctx) установить, какому соединению принадлежал
// прочитанный буфер и совпадает ли он с текущим поколением.
void ndl_rt_csr_diag() {
  if (!ndl_rt_csr_debug()) return;
  SimState& st = S();
  std::fprintf(stderr, "[csrdbg] --- groups ---\n");
  for (NdlGroup* g : st.groups)
    std::fprintf(stderr, "[csrdbg] g '%s' size=%lld spikes=%p prev=%p\n",
                 g->name.c_str(), (long long)g->size,
                 (void*)g->spikes.data(), (void*)g->prev_spikes.data());
  std::fprintf(stderr, "[csrdbg] --- conns ---\n");
  for (Connection* c : st.conns) {
    if (c->kind != CONN_SPARSE) continue;
    const uint64_t last = c->rowptr.empty() ? 0 : c->rowptr.back();
    std::fprintf(stderr,
                 "[csrdbg] c '%s'->'%s' rowptr=%p cols=%p vals=%p nnz=%zu "
                 "rp_last=%llu cols_cap=%zu vals_cap=%zu grown=%lld\n",
                 c->src->name.c_str(), c->dst->name.c_str(),
                 (void*)c->rowptr.data(), (void*)c->cols.data(),
                 (void*)c->vals.data(), c->cols.size(),
                 (unsigned long long)last, c->cols.capacity(),
                 c->vals.capacity(), (long long)c->grownTotal.load());
  }
}

// One tick of the CPU simulation. Tick order (INTERNALS §5, v2.0 additions in
// bold):
//   (1) i_syn *= syn_decay        (2) propagate per connection
//   (3) holds ← injections        **(3b) oscillators**    **(3c) streams**
//   (4) i_syn += hold + I_osc     (5) LIF step
//   (6) STDP × M(t) per plastic connection   **(6b) structural commands**
//   (7) trace decay               (8) spike logs + handlers
//   (9) prev_spikes = spikes      (10) sim_time_ms += dt
void ndl_rt_tick_cpu(double dt_ms, const std::vector<double>& decay, double synDecay,
                     bool applyInjections, int64_t& inj_idx, int64_t n_inj) {
  SimState& st = S();
  const size_t nG = st.groups.size();

  // (6b) structural commands queued since the last tick boundary (sleep/wake,
  // consolidation). Applied before the tick so the new state is in effect.
  // (drain takes UNIQUE on struct_mx via the leaves — must happen BEFORE the
  // shared_lock below, same thread, no recursion.)
  drain_commands();
  st.csr_tick_epoch = st.csr_epoch.load(std::memory_order_acquire);

  // Ф9.4f: SHARED for the whole tick body — pool workers (prop/LIF/STDP/trace)
  // dereference CSR buffers; the shared lock pins them against a concurrent
  // swap in apply_grow_direct (which waits for the tick to end).
  std::shared_lock<std::shared_mutex> struct_lk(st.struct_mx);

  // (1) exponential synaptic decay
  if (synDecay != 1.0)
    for (NdlGroup* g : st.groups)
      for (size_t k = 0; k < g->i_syn.size(); ++k) g->i_syn[k] *= synDecay;

  // (2) propagate per connection (acc over pre-synaptic *previous* spikes);
  // impulses accumulate in i_syn (the decaying synaptic component only)
  for (size_t ci = 0; ci < st.conns.size(); ++ci) {
    Connection* c = st.conns[ci];
    NdlGroup* pre = c->src;
    NdlGroup* post = c->dst;
    const uint8_t* psp = pre->prev_spikes.data();
    if (c->kind == CONN_DENSE) {
      const int64_t ns = pre->size, nd = post->size;
      PropDenseCtx ctx{c->w.data(), psp, post->i_syn.data(), ns};
      const bool par = ns != 0 && nd != 0 &&
                       (unsigned long long)ns * (unsigned long long)nd >=
                           (unsigned long long)kDenseWorkThreshold;
      run_parallel(0, nd, kLinearGrain, prop_dense_body, &ctx, par);
    } else if (c->kind == CONN_SPARSE) {
      PropSparseCtx ctx{c->rowptr.data(), c->cols.data(), c->vals.data(), psp,
                        post->i_syn.data()};
      const bool par = (int64_t)c->cols.size() >= kDenseWorkThreshold;
      run_parallel(0, post->size, kLinearGrain, prop_sparse_body, &ctx, par);
    } else {
      PropO2OCtx ctx{c->w.data(), psp, post->i_syn.data()};
      run_parallel(0, post->size, kLinearGrain, prop_o2o_body, &ctx,
                   post->size >= kLinearGrain);
    }
  }

  // (3) injections with t in [t0, t1): set sustained step currents
  if (applyInjections) {
    const double t0 = st.sim_time_ms;
    const double t1 = t0 + dt_ms;
    while (inj_idx < n_inj && st.injections[(size_t)inj_idx].t < t1) {
      const Injection& inj = st.injections[(size_t)inj_idx];
      if (inj.t >= t0 && inj.g) apply_injection(inj);
      ++inj_idx;
    }
  }

  // (3b) oscillators: I_osc = A * sin(2*pi*f*t + phi)
  apply_oscillators(st.sim_time_ms);

  // (3c) external streams: Poisson-encode the latest vector into bound groups
  apply_streams(dt_ms);

  // (4) LIF step (effective current = decaying synaptic + sustained hold)
  {
    const double sigma = lif_noise_sigma();
    const double kwta = kwta_frac();
    if (kwta > 0.0) {
      int64_t maxN = 0;
      for (size_t gi = 0; gi < nG; ++gi)
        maxN = std::max(maxN, st.groups[gi]->size);
      if ((int64_t)g_kwta_vpre.size() < maxN) g_kwta_vpre.resize((size_t)maxN);
    }
    for (size_t gi = 0; gi < nG; ++gi) {
      NdlGroup* g = st.groups[gi];
      LifCtx ctx{g->v.data(), g->i_syn.data(), g->hold.data(), g->spikes.data(),
                 dt_ms, g->tau, g->threshold, g->rest, g->reset, sigma,
                 g->noise_rng.data(),
                 kwta > 0.0 ? g_kwta_vpre.data() : nullptr};
      run_parallel(0, g->size, kLinearGrain, lif_body, &ctx,
                   g->size >= kLinearGrain);
      if (kwta > 0.0) apply_kwta(g);  // Ф9.4c: contrast clamp before STDP/log
    }
  }

  // (6) STDP × M(t) on plastic connections that have a config
  for (size_t ci = 0; ci < st.conns.size(); ++ci) {
    Connection* c = st.conns[ci];
    const StdpCfg* cfg = ndl_rt_find_stdp(st, c->src, c->dst);
    if (!c->plastic || !cfg) continue;
    NdlGroup* pre = c->src;
    NdlGroup* post = c->dst;
    const double* pre_trace = pre->trace.data();
    const double* post_trace = post->trace.data();
    const uint8_t* pre_prev = pre->prev_spikes.data();
    const uint8_t* post_sp = post->spikes.data();
    const double mod = stdp_modulator_value(cfg);  // v2.0: M(t)
    if (c->kind == CONN_DENSE) {
      const int64_t ns = pre->size, nd = post->size;
      StdpDenseCtx ctx{c->w.data(), pre_trace, post_sp, post_trace, pre_prev,
                       cfg->lr_pot, cfg->lr_dep, mod, ns};
      const bool par = ns != 0 && nd != 0 &&
                       (unsigned long long)ns * (unsigned long long)nd >=
                           (unsigned long long)kDenseWorkThreshold;
      run_parallel(0, nd, kLinearGrain, stdp_dense_body, &ctx, par);
    } else if (c->kind == CONN_SPARSE) {
      StdpSparseCtx ctx{c->rowptr.data(), c->cols.data(), c->vals.data(),
                        pre_trace, post_sp, post_trace, pre_prev,
                        cfg->lr_pot, cfg->lr_dep, mod};
      const bool par = (int64_t)c->cols.size() >= kDenseWorkThreshold;
      run_parallel(0, post->size, kLinearGrain, stdp_sparse_body, &ctx, par);
    } else {
      StdpO2OCtx ctx{c->w.data(), pre_trace, post_sp, post_trace, pre_prev,
                     cfg->lr_pot, cfg->lr_dep, mod};
      run_parallel(0, post->size, kLinearGrain, stdp_o2o_body, &ctx,
                   post->size >= kLinearGrain);
    }
  }

  // (7) traces
  for (size_t gi = 0; gi < nG; ++gi) {
    NdlGroup* g = st.groups[gi];
    TraceCtx ctx{g->trace.data(), g->spikes.data(), decay[gi]};
    run_parallel(0, g->size, kLinearGrain, trace_body, &ctx,
                 g->size >= kLinearGrain);
  }

  // (8) spike log + handlers (handlers get a stable copy of the list)
  for (NdlGroup* g : st.groups) {
    const int64_t n = g->size;
    const uint8_t* sp = g->spikes.data();
    for (int64_t i = 0; i < n; ++i)
      if (sp[i]) ndl_rt_log_spike(g, st.sim_time_ms, i);
    if (g->handlers.empty()) continue;
    const std::vector<NdlGroup::Handler> handlers = g->handlers;  // stable copy
    for (const NdlGroup::Handler& h : handlers)
      for (int64_t i = 0; i < n; ++i)
        if (sp[i]) h.fn(i, st.sim_time_ms, h.user);
  }

  // (9) prev_spikes = spikes
  for (NdlGroup* g : st.groups) g->prev_spikes = g->spikes;

  // (10) global time base
  st.sim_time_ms.store(st.sim_time_ms.load(std::memory_order_relaxed) + dt_ms,
                       std::memory_order_relaxed);

  // Ф9.4f-debug: эпоха не имела права меняться посреди тика (тело под SHARED,
  // структурные листья под UNIQUE). Разница = пойманная гонка.
  const uint64_t e1 = st.csr_epoch.load(std::memory_order_acquire);
  if (e1 != st.csr_tick_epoch)
    std::fprintf(stderr,
                 "[csrdbg] !!! CSR MUTATED MID-TICK epoch %llu -> %llu "
                 "sim=%.1f tid=%zu\n",
                 (unsigned long long)st.csr_tick_epoch,
                 (unsigned long long)e1, st.sim_time_ms.load(),
                 std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

double ndl_rt_sim_time(void) { return S().sim_time_ms; }

void ndl_rt_run(double duration_ms, double dt_ms, int32_t device) {
  SimState& st = S();
  if (!std::isfinite(dt_ms) || dt_ms <= 0.0) ndl_rt_panic("run: dt must be positive");
  if (!std::isfinite(duration_ms)) ndl_rt_panic("run: duration must be finite");
  if (st.cont_running.load(std::memory_order_relaxed))
    ndl_rt_panic("run: cannot call run() while the continuous loop is active");

  int64_t ticks = (int64_t)std::llround(duration_ms / dt_ms);
  if (ticks < 1) ticks = 1;

  std::stable_sort(st.injections.begin(), st.injections.end(),
                   [](const Injection& a, const Injection& b) { return a.t < b.t; });
  // v2.0: the tick loop runs on the global time base (sim_time_ms), so the
  // run-relative injection times are shifted once by the current time.
  for (Injection& inj : st.injections) inj.t += st.sim_time_ms;
  st.run_index++;

  // Each run starts electrically clean: no leftover currents or step holds.
  // (Done BEFORE the GPU attempt so a device run starts from the same state.)
  for (NdlGroup* g : st.groups) {
    std::fill(g->i_syn.begin(), g->i_syn.end(), 0.0);
    std::fill(g->hold.begin(), g->hold.end(), 0.0);
  }

  if (device == 1) {
    // Per-tick trace decay per group (dt is fixed for the run).
    std::vector<double> decay;
    double synDecay = 1.0;
    ndl_rt_prepare_ticks(dt_ms, decay, synDecay);
    std::string err;
    const int rc = ndl_rt_gpu_sim_execute(ticks, dt_ms, /*continuous=*/false,
                                          /*apply_injections=*/true, decay,
                                          synDecay, err);
    if (rc == 1) {
      st.injections.clear();  // consumed or dropped
      return;
    }
    if (rc == -1) {
      // Device lost mid-run: state was best-effort synced; continue on CPU.
      std::fprintf(stderr, "warning: GPU backend failed mid-run (%s); "
                           "migrating to CPU\n", err.c_str());
      st.injections.clear();
      return;  // ticks already executed up to the failure point
    }
    // rc == 0: refused before mutation → CPU path below.
    ndl_rt_warn_no_cuda();
  }

  // ---- CPU path ----

  // Per-tick trace decay per group, computed once per run (dt is fixed).
  std::vector<double> decay;
  double synDecay = 1.0;
  ndl_rt_prepare_ticks(dt_ms, decay, synDecay);

  // Injections are sorted once; a snapshot of the count keeps the tick scan
  // well-defined even if a spike handler schedules more injections (those are
  // cleared with everything else at the end of the run).
  int64_t inj_idx = 0;
  const int64_t n_inj = (int64_t)st.injections.size();

  for (int64_t tick = 0; tick < ticks; ++tick) {
    ndl_rt_tick_cpu(dt_ms, decay, synDecay, true, inj_idx, n_inj);
  }

  st.injections.clear();  // consumed or dropped
}

// --- internal state swap (benchmarks, :gputest) --------------------------------

SimState* ndl_rt_state_swap_fresh() {
  SimState& st = S();
  if (st.cont_running.load(std::memory_order_relaxed))
    ndl_rt_panic("state_swap: the continuous loop must be stopped");
  SimState* fresh = new SimState();
  SimState* old = &st;
  // Re-anchor the global pointer (ndl_rt_sim() owns it from here).
  set_sim_state(fresh);
  ndl_rt_gpu_backend_invalidate();
  return old;
}

void ndl_rt_state_restore(SimState* old) {
  if (!old) ndl_rt_panic("state_restore: null state");
  SimState& cur = S();
  if (&cur != old) {
    // Free the synthetic state's contents, then the state itself.
    for (NdlGroup* g : cur.groups) delete g;
    for (Connection* c : cur.conns) delete c;
    for (StdpCfg* c : cur.stdps) delete c;
    delete &cur;
  }
  set_sim_state(old);
  ndl_rt_gpu_backend_invalidate();
}

double ndl_rt_get_weight(NdlGroup* src, NdlGroup* dst, int64_t i, int64_t j) {
  if (!src || !dst) ndl_rt_panic("get_weight: null group");
  Connection* c = find_conn(src, dst);
  if (!c) {
    ndl_rt_panic(ndl_rt_sfmt("get_weight: no connection '%s' -> '%s'",
                             src->name.c_str(), dst->name.c_str())
                     .c_str());
  }
  if (i < 0 || i >= src->size || j < 0 || j >= dst->size)
    ndl_rt_panic("get_weight: index out of range");
  switch (c->kind) {
    case CONN_DENSE:
      return c->w[(size_t)(j * src->size + i)];
    case CONN_SPARSE:
      for (uint64_t k = c->rowptr[(size_t)j]; k < c->rowptr[(size_t)j + 1]; ++k)
        if ((int64_t)c->cols[(size_t)k] == i) return c->vals[(size_t)k];
      ndl_rt_panic(ndl_rt_sfmt("get_weight: no synapse at (%lld, %lld) of '%s' -> '%s'",
                               (long long)i, (long long)j, src->name.c_str(),
                               dst->name.c_str())
                       .c_str());
    case CONN_ONE2ONE:
      if (i != j) ndl_rt_panic("get_weight: one_to_one lookup requires i == j");
      return c->w[(size_t)i];
  }
  ndl_rt_panic("get_weight: unknown connection kind");
  return 0.0;  // unreachable
}

// --- v2.1: bounded spike log + connectivity introspection (Ф9.4) -------------

void ndl_rt_log_spike(NdlGroup* g, double t_ms, int64_t neuron) {
  auto& log = g->spike_log;
  static const int64_t cap = []() {
    const char* e = std::getenv("NDL_SPIKELOG_MAX");
    int64_t v = e && *e ? std::atoll(e) : 4000000;
    if (v < 100000) v = 100000;  // the raster tool needs a sane window anyway
    return v;
  }();
  log.push_back({t_ms, neuron});
  if ((int64_t)log.size() > cap)
    log.erase(log.begin(), log.begin() + log.size() / 2);  // drop oldest half
}

int64_t ndl_rt_row_hits(NdlGroup* src, NdlGroup* dst, const int64_t* idx,
                        int64_t n, int64_t rows_cap, int64_t* offs_out,
                        int64_t* cols_out, double* vals_out, int64_t cap_out) {
  if (!src || !dst) ndl_rt_panic("row_hits: null group");
  if (rows_cap < 0 || cap_out < 0 || n < 0)
    ndl_rt_panic("row_hits: negative size/cap");
  if (!offs_out) ndl_rt_panic("row_hits: offs_out required");
  Connection* c = find_conn(src, dst);
  if (!c || c->kind != CONN_SPARSE) return INT64_MIN;  // unambiguous vs -need
  ndl_rt_gpu_sync_down();  // device owns the weights — bring mirrors home
  // Ф9.4f: SHARED — the decoder reads live CSR buffers on the host thread
  // while the continuous loop may swap them between ticks.
  std::shared_lock<std::shared_mutex> lk(S().struct_mx);
  if (rows_cap > dst->size) rows_cap = dst->size;
  // presence bitmap over the (small) presynaptic subset
  std::vector<uint8_t> bits((size_t)((src->size + 7) / 8), 0);
  int64_t nset = 0;
  for (int64_t k = 0; k < n; ++k) {
    const int64_t i = idx[k];
    if (i < 0 || i >= src->size) continue;
    if (!((bits[(size_t)(i >> 3)] >> (i & 7)) & 1)) {
      bits[(size_t)(i >> 3)] |= (uint8_t)(1u << (i & 7));
      ++nset;
    }
  }
  (void)nset;
  // pass 1: count hits per row
  int64_t total = 0;
  for (int64_t j = 0; j < rows_cap; ++j) {
    int64_t cnt = 0;
    for (uint64_t k = c->rowptr[(size_t)j]; k < c->rowptr[(size_t)j + 1]; ++k)
      if ((bits[(size_t)(c->cols[(size_t)k] >> 3)] >> (c->cols[(size_t)k] & 7)) &
          1)
        ++cnt;
    offs_out[(size_t)j] = cnt;
    total += cnt;
  }
  int64_t acc = 0;
  for (int64_t j = 0; j < rows_cap; ++j) {
    const int64_t v = offs_out[(size_t)j];
    offs_out[(size_t)j] = acc;
    acc += v;
  }
  offs_out[(size_t)rows_cap] = acc;
  if (acc > cap_out) return -acc;  // caller retries with a larger buffer
  // pass 2: fill (cols + weights per row, ascending column order as stored)
  for (int64_t j = 0; j < rows_cap; ++j) {
    int64_t cur = offs_out[(size_t)j];
    for (uint64_t k = c->rowptr[(size_t)j]; k < c->rowptr[(size_t)j + 1]; ++k) {
      const uint32_t col = c->cols[(size_t)k];
      if ((bits[(size_t)(col >> 3)] >> (col & 7)) & 1) {
        cols_out[(size_t)cur] = (int64_t)col;
        vals_out[(size_t)cur] = c->vals[(size_t)k];
        ++cur;
      }
    }
  }
  return acc;
}
