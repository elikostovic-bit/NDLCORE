// NDL v2.0 — runtime layer for the asynchronous continuous-time environment:
//   * signals (neuromodulators) — M(t) of the modulated 3-factor STDP
//   * oscillators (alpha/gamma/delta rhythms)
//   * external streams (SEB ring + Poisson encoding into bound groups)
//   * structural plasticity (set_plasticity / prune_weights, queue-safe)
//   * continuous mode (std::thread event loop, std::atomic<bool> is_running)
//   * the STDP modulator binding (ndl_rt_stdp_set_modulator)
//
// Thread-safety model:
//   - the continuous loop is the only writer of neuron/connection state while
//     it runs; the NDL main thread is parked in ndl_rt_stop_continuous;
//   - host threads may push streams and set signals at any time (SPSC ring /
//     atomic double);
//   - structural commands issued while the loop runs are queued (cmd_mx) and
//     applied by the simulation thread between ticks (see drain_commands in
//     ndl_rt_network.cpp); issued while it is stopped they apply immediately.
#include "ndl_rt_internal.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace {

SimState& S() { return ndl_rt_sim(); }

NdlSignal* find_signal(const std::string& name) {
  for (NdlSignal* s : S().signals)
    if (s->name == name) return s;
  return nullptr;
}

NdlStream* find_stream(int64_t id) {
  if (id < 0 || (size_t)id >= S().streams.size()) return nullptr;
  return S().streams[(size_t)id];
}

void queue_or_apply(const StructCmd& c) {
  SimState& st = S();
  if (st.cont_running.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lk(st.cmd_mx);
    st.cmds.push_back(c);
  } else if (c.k == StructCmd::K::SetPlasticity) {
    ndl_rt_apply_set_plasticity_direct(c.src, c.dst, c.enabled);
  } else if (c.k == StructCmd::K::Prune) {
    ndl_rt_apply_prune_direct(c.src, c.dst, c.threshold);
  } else if (c.k == StructCmd::K::Normalize) {
    // (was falling into the Prune branch and silently no-op'ing: threshold 0
    // rejects — found while wiring the Grow branch, Ф9.4b)
    ndl_rt_apply_normalize_direct(c.src, c.dst, c.target_sum);
  }
}

void cont_loop(double dt_ms) {
  SimState& st = S();
  std::vector<double> decay;
  double synDecay = 1.0;
  ndl_rt_prepare_ticks(dt_ms, decay, synDecay);
  int64_t inj_idx = 0;
  const int64_t n_inj = 0;  // continuous mode does not consume injections

  // Electrically clean start of the continuous episode.
  for (NdlGroup* g : st.groups) {
    std::fill(g->i_syn.begin(), g->i_syn.end(), 0.0);
    std::fill(g->hold.begin(), g->hold.end(), 0.0);
  }

  // Phase 2: try to run the episode on a device (CUDA → OpenCL). A refusal
  // (rc 0, e.g. no driver) falls back to the CPU silently-ish (the dispatcher
  // warns once); a device FAILURE mid-episode (rc -1) migrates to the CPU on
  // the fly — the host mirrors carry the last synced state.
  bool gpu_ok = false;
  if (st.gpu_requested) {
    std::string err;
    const int rc = ndl_rt_gpu_sim_execute(1, dt_ms, /*continuous=*/true,
                                          /*apply_injections=*/false, decay,
                                          synDecay, err);
    if (rc == 1) {
      gpu_ok = true;
    } else if (rc == -1) {
      std::fprintf(stderr, "warning: GPU backend failed (%s); continuing on CPU\n",
                   err.c_str());
    }
  }

  // Real-time pacing: this is a CONTINUOUS-TIME environment — one dt of
  // simulated time must take dt of wall time, so rhythms (10 Hz alpha is a
  // 100 ms period), streams and dopamine-gated STDP traces evolve at the
  // pace a host (or a human teacher) experiences. Ticks are batched to keep
  // timer overhead low; if the machine falls behind, the lag is dropped
  // instead of accumulated (the loop never spirals).
  using clock = std::chrono::steady_clock;
  const auto dt = std::chrono::duration_cast<clock::duration>(
      std::chrono::duration<double>(dt_ms / 1000.0));
  constexpr int kBatch = 4;  // ticks per pacing sleep
  int batch = 0;
  auto next = clock::now();

  while (st.cont_running.load(std::memory_order_relaxed)) {
    if (gpu_ok) {
      std::string err;
      const int rc = ndl_rt_gpu_sim_execute(1, dt_ms, true, false, decay,
                                            synDecay, err);
      if (rc != 1) {  // device lost mid-episode → migrate to CPU
        std::fprintf(stderr, "warning: GPU backend lost (%s); migrating to CPU\n",
                     err.c_str());
        gpu_ok = false;
        ndl_rt_tick_cpu(dt_ms, decay, synDecay, false, inj_idx, n_inj);
      }
    } else {
      ndl_rt_tick_cpu(dt_ms, decay, synDecay, false, inj_idx, n_inj);
    }
    if (++batch >= kBatch) {
      batch = 0;
      next += kBatch * dt;
      const auto now = clock::now();
      if (now - next > std::chrono::milliseconds(50))
        next = now;  // fell too far behind: drop the lag, keep ticking
      std::this_thread::sleep_until(next);
    }
  }
}

}  // namespace

// --- signals -------------------------------------------------------------------

void ndl_rt_signal_create(const char* name, double initial) {
  if (!name) ndl_rt_panic("signal_create: null name");
  if (find_signal(name)) ndl_rt_panic("signal_create: duplicate signal");
  NdlSignal* s = nullptr;
  try {
    s = new NdlSignal();
    s->name = name;
    s->value.store(initial, std::memory_order_relaxed);
    S().signals.push_back(s);
  } catch (const std::bad_alloc&) {
    delete s;
    ndl_rt_panic("out of memory");
  }
}

void ndl_rt_signal_set(const char* name, double value) {
  if (!name) ndl_rt_panic("signal_set: null name");
  NdlSignal* s = find_signal(name);
  if (!s)
    ndl_rt_panic(ndl_rt_sfmt("signal_set: unknown signal '%s'", name).c_str());
  s->value.store(value, std::memory_order_relaxed);
}

double ndl_rt_signal_get(const char* name) {
  if (!name) ndl_rt_panic("signal_get: null name");
  NdlSignal* s = find_signal(name);
  if (!s)
    ndl_rt_panic(ndl_rt_sfmt("signal_get: unknown signal '%s'", name).c_str());
  return s->value.load(std::memory_order_relaxed);
}

// --- modulated 3-factor STDP binding --------------------------------------------

void ndl_rt_stdp_set_modulator(NdlGroup* src, NdlGroup* dst, const char* name) {
  if (!src || !dst) ndl_rt_panic("stdp_set_modulator: null group");
  StdpCfg* cfg = nullptr;
  for (StdpCfg* c : S().stdps)
    if (c->src == src && c->dst == dst) { cfg = c; break; }
  if (!cfg)
    ndl_rt_panic(ndl_rt_sfmt("stdp_set_modulator: no configure_stdp for '%s' -> '%s'",
                             src->name.c_str(), dst->name.c_str())
                     .c_str());
  if (name && *name) {
    // Bind by name; existence was validated by sema. If the signal is created
    // later (host-driven ordering), stdp_modulator_value resolves it lazily
    // and treats an unknown name as M = 1.
    cfg->modulator = name;
  } else {
    cfg->modulator.clear();
  }
}

// --- oscillators -----------------------------------------------------------------

void ndl_rt_oscillator_add(const char* name, NdlGroup* target, double freq_hz,
                           double amplitude, double phase) {
  if (!name) ndl_rt_panic("oscillator_add: null name");
  if (!target) ndl_rt_panic("oscillator_add: null target group");
  if (!(freq_hz > 0.0)) ndl_rt_panic("oscillator_add: frequency must be positive");
  if (!std::isfinite(amplitude) || !std::isfinite(phase) || !std::isfinite(freq_hz))
    ndl_rt_panic("oscillator_add: non-finite parameter");
  NdlOscillator* o = nullptr;
  try {
    o = new NdlOscillator();
    o->name = name;
    o->target = target;
    o->freq_hz = freq_hz;
    o->amplitude = amplitude;
    o->phase = phase;
    S().oscillators.push_back(o);
  } catch (const std::bad_alloc&) {
    delete o;
    ndl_rt_panic("out of memory");
  }
}

// --- external streams --------------------------------------------------------------

int64_t ndl_rt_stream_create(const char* name, int64_t size) {
  if (!name) ndl_rt_panic("stream_create: null name");
  if (size <= 0) ndl_rt_panic("stream_create: size must be positive");
  for (const NdlStream* s : S().streams)
    if (s->name == name) ndl_rt_panic("stream_create: duplicate stream");
  NdlStream* s = nullptr;
  try {
    s = new NdlStream();
    s->name = name;
    s->size = size;
    s->latest.assign((size_t)size, 0.0);
    s->ring.assign((size_t)(NdlStream::kRingCap * size), 0.0);
    S().streams.push_back(s);
  } catch (const std::bad_alloc&) {
    delete s;
    ndl_rt_panic("out of memory");
  }
  return (int64_t)S().streams.size() - 1;
}

int64_t ndl_rt_stream_find(const char* name) {
  if (!name) return -1;
  SimState& st = S();
  for (size_t i = 0; i < st.streams.size(); ++i)
    if (st.streams[i]->name == name) return (int64_t)i;
  return -1;
}

void ndl_rt_stream_push(int64_t stream, const double* data, int64_t n) {
  NdlStream* s = find_stream(stream);
  if (!s) ndl_rt_panic("stream_push: bad stream id");
  if (!data) ndl_rt_panic("stream_push: null data");
  if (n != s->size)
    ndl_rt_panic(ndl_rt_sfmt("stream_push: expected %lld values, got %lld",
                             (long long)s->size, (long long)n)
                     .c_str());
  // SPSC ring: reserve the next slot; drop the sample when the ring is full
  // (the consumer drains every tick, so this only happens on extreme overrun).
  const int64_t w = s->wpos.load(std::memory_order_relaxed);
  const int64_t r = s->rpos.load(std::memory_order_acquire);
  if (w - r >= NdlStream::kRingCap) return;  // ring full: drop
  const int64_t slot = (w % NdlStream::kRingCap) * s->size;
  std::memcpy(&s->ring[(size_t)slot], data, sizeof(double) * (size_t)s->size);
  s->wpos.store(w + 1, std::memory_order_release);
}

void ndl_rt_stream_set(int64_t stream, int64_t index, double value) {
  NdlStream* s = find_stream(stream);
  if (!s) ndl_rt_panic("stream_set: bad stream id");
  if (index < 0 || index >= s->size)
    ndl_rt_panic("stream_set: index out of range");
  std::lock_guard<std::mutex> lk(s->mx);
  s->latest[(size_t)index] = value;
}

void ndl_rt_bind_input_stream(int64_t stream, NdlGroup* target, int32_t encoding,
                              double max_freq, double kick) {
  NdlStream* s = find_stream(stream);
  if (!s) ndl_rt_panic("bind_input_stream: bad stream id");
  if (!target) ndl_rt_panic("bind_input_stream: null group");
  if (encoding != 0) ndl_rt_panic("bind_input_stream: unknown encoding (0 = Poisson)");
  if (!(max_freq > 0.0)) ndl_rt_panic("bind_input_stream: max_freq must be positive");
  if (!std::isfinite(kick)) ndl_rt_panic("bind_input_stream: kick must be finite");
  s->target = target;
  s->encoding = encoding;
  s->max_freq = max_freq;
  s->kick = kick;
}

// --- structural plasticity -----------------------------------------------------------

void ndl_rt_set_plasticity(NdlGroup* src, NdlGroup* dst, int32_t enabled) {
  if (!src || !dst) ndl_rt_panic("set_plasticity: null group");
  StructCmd c;
  c.k = StructCmd::K::SetPlasticity;
  c.src = src;
  c.dst = dst;
  c.enabled = enabled != 0;
  queue_or_apply(c);
}

void ndl_rt_prune_weights(NdlGroup* src, NdlGroup* dst, double threshold) {
  if (!src || !dst) ndl_rt_panic("prune_weights: null group");
  if (!std::isfinite(threshold) || threshold < 0.0)
    ndl_rt_panic("prune_weights: threshold must be a non-negative finite value");
  StructCmd c;
  c.k = StructCmd::K::Prune;
  c.src = src;
  c.dst = dst;
  c.threshold = threshold;
  queue_or_apply(c);
}

void ndl_rt_normalize_incoming(NdlGroup* src, NdlGroup* dst, double target_sum) {
  if (!src || !dst) ndl_rt_panic("normalize_incoming: null group");
  if (!std::isfinite(target_sum) || target_sum <= 0.0)
    ndl_rt_panic("normalize_incoming: target_sum must be a positive finite value");
  StructCmd c;
  c.k = StructCmd::K::Normalize;
  c.src = src;
  c.dst = dst;
  c.target_sum = target_sum;
  queue_or_apply(c);
}

// --- Ф9.4b: structural plasticity of binding (grow-on-demand) ----------------

int64_t ndl_rt_grow_synapses(NdlGroup* src, NdlGroup* dst,
                             const int64_t* pre, int64_t n_pre,
                             const int64_t* post, int64_t n_post,
                             double weight) {
  if (!src || !dst) ndl_rt_panic("grow_synapses: null group");
  if (!pre || !post)
    ndl_rt_panic("grow_synapses: null index array");
  if (!std::isfinite(weight))
    ndl_rt_panic("grow_synapses: weight must be finite");
  // Fail fast on the caller thread (the queue path would defer the panic to
  // the sim thread between ticks — same outcome, worse to debug).
  {
    SimState& st = S();
    Connection* c = nullptr;
    for (Connection* k : st.conns)
      if (k->src == src && k->dst == dst) { c = k; break; }
    if (!c)
      ndl_rt_panic(ndl_rt_sfmt("grow_synapses: no connection '%s' -> '%s'",
                               src->name.c_str(), dst->name.c_str())
                       .c_str());
    if (c->kind != CONN_SPARSE)
      ndl_rt_panic("grow_synapses: only sparse connections support growing");
  }
  SimState& st = S();
  if (st.cont_running.load(std::memory_order_relaxed)) {
    StructCmd c;
    c.k = StructCmd::K::Grow;
    c.src = src;
    c.dst = dst;
    c.weight = weight;
    c.pre_idx = std::make_shared<std::vector<int64_t>>(pre, pre + n_pre);
    c.post_idx = std::make_shared<std::vector<int64_t>>(post, post + n_post);
    {
      std::lock_guard<std::mutex> lk(st.cmd_mx);
      st.cmds.push_back(c);
    }
    return -1;  // queued; applied between ticks
  }
  return ndl_rt_struct_grow(src, dst, pre, n_pre, post, n_post, weight);
}

int64_t ndl_rt_grown_total(NdlGroup* src, NdlGroup* dst) {
  if (!src || !dst) return 0;
  SimState& st = S();
  for (Connection* c : st.conns)
    if (c->src == src && c->dst == dst)
      return c->grownTotal.load(std::memory_order_relaxed);
  return 0;
}

// --- continuous mode -------------------------------------------------------------------

void ndl_rt_run_continuous(double dt_ms, int32_t device) {
  SimState& st = S();
  if (!std::isfinite(dt_ms) || dt_ms <= 0.0)
    ndl_rt_panic("run_continuous: dt must be positive");
  if (st.cont_running.load(std::memory_order_relaxed))
    ndl_rt_panic("run_continuous: the continuous loop is already running");
  if (st.cont_thread.joinable()) st.cont_thread.join();  // previous episode

  // Phase 2: device == 1 asks for GPU execution (CUDA → OpenCL → CPU inside
  // the tick dispatcher); device == 0 pins the CPU path.
  st.gpu_requested = (device == 1);

  st.injections.clear();
  st.run_index++;
  st.cont_running.store(true, std::memory_order_relaxed);
  try {
    std::thread th([dt_ms]() { cont_loop(dt_ms); });
    st.cont_thread = std::move(th);
  } catch (...) {
    st.cont_running.store(false, std::memory_order_relaxed);
    ndl_rt_panic("run_continuous: failed to start the event loop thread");
  }
  // Safety net: even if the host forgets to stop the loop, exit cleanly.
  std::atexit([]() { ndl_rt_stop_continuous(); });
}

void ndl_rt_stop_continuous(void) {
  SimState& st = S();
  if (!st.cont_running.load(std::memory_order_relaxed)) return;
  st.cont_running.store(false, std::memory_order_relaxed);
  const bool csrdbg = std::getenv("NDL_CSRDEBUG");
  if (csrdbg) std::fprintf(stderr, "[csrdbg] stop_continuous: joining loop thread\n");
  if (st.cont_thread.joinable()) {
    if (st.cont_thread.get_id() == std::this_thread::get_id()) {
      // stop_continuous() called from a spike handler on the loop thread
      // itself: flag is set, the loop exits at the next tick boundary, and
      // the thread detaches (a self-join would deadlock).
      st.cont_thread.detach();
    } else {
      st.cont_thread.join();
    }
  }
  if (csrdbg) std::fprintf(stderr, "[csrdbg] stop_continuous: joined ok\n");
  // Device state → host mirrors, then release device buffers: with the loop
  // stopped the host owns the brain again (checkpoints, direct structural
  // ops, introspection all read the mirrors).
  ndl_rt_gpu_sync_down();
  ndl_rt_gpu_backend_invalidate();
}

int32_t ndl_rt_is_continuous(void) {
  return S().cont_running.load(std::memory_order_relaxed) ? 1 : 0;
}

void ndl_rt_wait_continuous(void) {
  SimState& st = S();
  if (st.cont_thread.get_id() == std::this_thread::get_id()) return; // never on the loop thread
  while (st.cont_running.load(std::memory_order_relaxed))
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (st.cont_thread.joinable()) st.cont_thread.join();
}
