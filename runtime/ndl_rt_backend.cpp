// NDL v2.1 — GPU backend dispatch (Phase 2).
//
// Selection order (ROADMAP Фаза 2): CUDA → OpenCL → CPU, with a one-time
// warning when the CPU path takes over.
//
//   * CUDA  — the existing PTX path (ndl_rt_gpu.cpp): discrete runs of pure
//     v1.0 networks with a compiler-emitted PTX module. Stateless per run:
//     everything is uploaded, executed, downloaded back (v, trace, weights).
//   * OpenCL — hand-written kernels (ndl_ocl_kernels.cl) for the built-in
//     simulation INCLUDING all v2.0 features (streams, oscillators, signals,
//     modulated STDP, structural commands). This is the backend that scales
//     the AGI brain (sparse plastic conns, continuous mode).
//
// Ownership model: while an OpenCL episode runs, the DEVICE owns v / trace /
// weights / noise RNG; the host keeps per-tick spike mirrors. Checkpoint
// saves and stop_continuous sync the device state down. A device failure
// mid-run is a migration: best-effort sync, then the CPU continues.
#include "ndl_rt_internal.h"

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// --- implemented in ndl_rt_ocl.cpp -------------------------------------------
// rc: 1 = executed, 0 = refused before mutation, -1 = failed mid-run.
int ndl_rt_ocl_execute(int64_t ticks, double dt_ms, bool continuous,
                       bool apply_injections, const std::vector<double>& decay,
                       double synDecay, std::string& err);
bool ndl_rt_ocl_attached();
const char* ndl_rt_ocl_device_name(std::string* err);
void ndl_rt_ocl_prune(NdlGroup* src, NdlGroup* dst, double threshold);
void ndl_rt_ocl_normalize(NdlGroup* src, NdlGroup* dst, double target_sum);
void ndl_rt_ocl_grow(NdlGroup* src, NdlGroup* dst, const int64_t* pre,
                     int64_t n_pre, const int64_t* post, int64_t n_post,
                     double weight);
void ndl_rt_ocl_sync_down();
void ndl_rt_ocl_sync_group_v(NdlGroup* g);
void ndl_rt_ocl_invalidate();

namespace {

std::mutex g_mu;
std::string g_last_backend;  // last device that executed ticks ("cuda"/"opencl")
bool g_warned_no_device = false;

void warn_no_device_once(const char* why) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_warned_no_device) return;
  g_warned_no_device = true;
  std::fprintf(stderr,
               "warning: no GPU backend for the built-in simulation (%s); "
               "using the CPU path\n",
               why && why[0] ? why : "no driver");
}

}  // namespace

int ndl_rt_gpu_sim_execute(int64_t ticks, double dt_ms, bool continuous,
                           bool apply_injections, const std::vector<double>& decay,
                           double synDecay, std::string& err) {
  SimState& st = ndl_rt_sim();
  err.clear();

  // (a) CUDA PTX path — discrete, pure v1.0, registered PTX module.
  // (ndl_rt_gpu_run refuses cleanly: one2one / plastic-sparse / no module /
  // no driver → fall through to OpenCL.)
  if (!continuous && !st.gpu_ptx.empty() && !ndl_rt_has_v2_sim_features(st)) {
    if (ndl_rt_gpu_run(ticks, dt_ms)) {
      std::lock_guard<std::mutex> lk(g_mu);
      g_last_backend = "cuda";
      return 1;
    }
    err = "cuda: PTX path refused";
  }

  // (b) OpenCL — the v2.0 built-in simulation (sparse plastic included).
  const int rc = ndl_rt_ocl_execute(ticks, dt_ms, continuous, apply_injections,
                                    decay, synDecay, err);
  if (rc == 1) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_last_backend = "opencl";
    return 1;
  }
  if (rc == -1) return -1;  // migration: state synced, caller continues on CPU

  // (c) refused → CPU (warn once per process).
  warn_no_device_once(err.c_str());
  return 0;
}

const char* ndl_rt_gpu_active_backend() {
  std::lock_guard<std::mutex> lk(g_mu);
  if (ndl_rt_ocl_attached()) return "opencl";
  return g_last_backend.empty() ? nullptr : g_last_backend.c_str();
}

// --- structural routing -------------------------------------------------------

void ndl_rt_struct_prune(NdlGroup* src, NdlGroup* dst, double threshold) {
  if (src && dst && ndl_rt_ocl_attached()) {
    ndl_rt_ocl_prune(src, dst, threshold);  // download → compact → upload
    return;
  }
  ndl_rt_apply_prune_direct(src, dst, threshold);
}

void ndl_rt_struct_normalize(NdlGroup* src, NdlGroup* dst, double target_sum) {
  if (src && dst && ndl_rt_ocl_attached()) {
    ndl_rt_ocl_normalize(src, dst, target_sum);  // device rowsum+scale kernels
    return;
  }
  ndl_rt_apply_normalize_direct(src, dst, target_sum);
}

// Ф9.4b: grow while attached goes download → host merge → in-place CSR
// buffer replacement (ndl_rt_ocl_grow); a full re-attach would rebuild every
// buffer from the host mirrors and discard the live membrane/trace state.
int64_t ndl_rt_struct_grow(NdlGroup* src, NdlGroup* dst, const int64_t* pre,
                           int64_t n_pre, const int64_t* post, int64_t n_post,
                           double weight) {
  if (src && dst && ndl_rt_ocl_attached()) {
    ndl_rt_ocl_grow(src, dst, pre, n_pre, post, n_post, weight);
    return 0;  // the added count lands in Connection::grownTotal
  }
  return ndl_rt_apply_grow_direct(src, dst, pre, n_pre, post, n_post, weight,
                                  true);
}

void ndl_rt_gpu_sync_down() {
  if (ndl_rt_ocl_attached()) ndl_rt_ocl_sync_down();
}

void ndl_rt_gpu_sync_group_v(NdlGroup* g) {
  if (g && ndl_rt_ocl_attached()) ndl_rt_ocl_sync_group_v(g);
}

void ndl_rt_gpu_backend_invalidate() { ndl_rt_ocl_invalidate(); }

// =============================================================================
// Phase 2: :gputest / :bench gpu backends (synthetic networks via state swap)
// =============================================================================

namespace {

// Deterministic mini-AGI: sense 64 → E 1024 ↔ I 256, E → read 64 (plastic,
// dopamine-modulated), read → E; alpha oscillator; burst injections.
void gputestBuildNet() {
  SimState& st = ndl_rt_sim();
  ndl_rt_group_create(64, 0, 10.0, -55.0, -70.0, -75.0, "gt_input");
  ndl_rt_group_create(1024, 0, 20.0, -55.0, -70.0, -75.0, "gt_coreE");
  ndl_rt_group_create(256, 1, 10.0, -50.0, -70.0, -75.0, "gt_coreI");
  ndl_rt_group_create(64, 0, 20.0, -55.0, -70.0, -75.0, "gt_read");

  ndl_rt_sparse_connect(st.groups[0], st.groups[1], 0.05, 12.0, 0);
  ndl_rt_sparse_connect(st.groups[1], st.groups[1], 0.06, 1.5, 0);
  ndl_rt_sparse_connect(st.groups[1], st.groups[2], 0.15, 2.5, 0);
  ndl_rt_sparse_connect(st.groups[2], st.groups[1], 0.20, -4.0, 0);
  ndl_rt_sparse_connect(st.groups[1], st.groups[3], 0.15, 3.0, 1);
  ndl_rt_sparse_connect(st.groups[3], st.groups[1], 0.08, 1.0, 0);
  ndl_rt_configure_stdp(st.groups[1], st.groups[3], 0.12, 0.005, 20.0);

  ndl_rt_signal_create("dopamine", 0.2);
  ndl_rt_stdp_set_modulator(st.groups[1], st.groups[3], "dopamine");
  ndl_rt_oscillator_add("alpha", st.groups[1], 10.0, 1.5, 0.0);
}

// 10 chunks × 1000 ms = 10k ticks; dopamine pulses + two burst injections
// inside every chunk (identical inputs for both legs).
void gputestRunEpisode(int32_t device) {
  SimState& st = ndl_rt_sim();
  for (int chunk = 0; chunk < 10; ++chunk) {
    ndl_rt_signal_set("dopamine", chunk % 2 == 0 ? 1.0 : 0.1);
    ndl_rt_schedule_inject(st.groups[0], 0, 16, 1.0, 40.0);
    ndl_rt_schedule_inject(st.groups[0], 32, 48, 500.0, 30.0);
    ndl_rt_run(1000.0, 1.0, device);
  }
}

struct GtStats {
  int64_t spikes[4] = {0, 0, 0, 0};
  double plastic_mean = 0.0;
  double v_sum = 0.0;
};

GtStats gputestCollect() {
  GtStats s;
  SimState& st = ndl_rt_sim();
  for (int gi = 0; gi < 4 && gi < (int)st.groups.size(); ++gi)
    s.spikes[gi] = (int64_t)st.groups[(size_t)gi]->spike_log.size();
  for (const Connection* c : st.conns) {
    if (!c->plastic) continue;
    double sum = 0.0;
    int64_t n = 0;
    if (c->kind == CONN_SPARSE) {
      for (double w : c->vals) sum += w;
      n = (int64_t)c->vals.size();
    } else {
      for (double w : c->w) sum += w;
      n = (int64_t)c->w.size();
    }
    s.plastic_mean = n ? sum / (double)n : 0.0;
    break;
  }
  const NdlGroup* read = st.groups[3];
  for (int64_t i = 0; i < read->size; ++i) s.v_sum += read->v[(size_t)i];
  return s;
}

double relDiffGt(double a, double b) {
  const double d = std::fabs(a - b);
  const double m = std::max(1e-9, std::max(std::fabs(a), std::fabs(b)));
  return d / m;
}

}  // namespace

int ndl_rt_gputest(char* buf, int cap) {
  int pos = 0;
  auto put = [&](const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    if (buf && pos < cap) pos += std::vsnprintf(buf + pos, (size_t)std::max(0, cap - pos), fmt, args);
    va_end(args);
  };

  // probe the OpenCL device (CUDA is tried inside the dispatcher per-leg; the
  // built-in simulation with v2.0 features is OpenCL territory)
  std::string devErr;
  const char* dev = ndl_rt_ocl_device_name(&devErr);
  if (!dev) {
    put("gputest: OpenCL backend unavailable — %s\n", devErr.c_str());
    put("валидация переносится на RX 6600 XT (машина хозяина)\n");
    return -1;
  }
  const bool exact = std::strstr(dev, "Mock") != nullptr;

  put("gputest: device = %s%s\n", dev, exact ? " (bit-exact mode)" : "");
  SimState* brain = nullptr;

  const auto leg = [&](int32_t device) {
    brain = ndl_rt_state_swap_fresh();
    ndl_rt_set_seed(97531ull);
    gputestBuildNet();
    gputestRunEpisode(device);
    const GtStats s = gputestCollect();
    ndl_rt_state_restore(brain);
    return s;
  };

  const GtStats cpu = leg(0);
  const GtStats gpu = leg(1);

  bool ok = true;
  for (int gi = 0; gi < 4; ++gi) {
    const int64_t a = cpu.spikes[gi], b = gpu.spikes[gi];
    const bool match = exact ? (a == b) : (std::llabs(a - b) * 20 <= std::max<int64_t>(1, a));
    put("  group %d: cpu=%lld gpu=%lld %s\n", gi, (long long)a, (long long)b,
        match ? "OK" : "MISMATCH");
    if (!match) ok = false;
  }
  const bool wOk = exact ? (relDiffGt(cpu.plastic_mean, gpu.plastic_mean) < 1e-12)
                         : (relDiffGt(cpu.plastic_mean, gpu.plastic_mean) < 0.02);
  put("  plastic mean w: cpu=%.9f gpu=%.9f %s\n", cpu.plastic_mean,
      gpu.plastic_mean, wOk ? "OK" : "MISMATCH");
  if (!wOk) ok = false;
  const bool vOk = exact ? (relDiffGt(cpu.v_sum, gpu.v_sum) < 1e-9)
                         : (relDiffGt(cpu.v_sum, gpu.v_sum) < 0.05);
  put("  read v_sum: cpu=%.6f gpu=%.6f %s\n", cpu.v_sum, gpu.v_sum,
      vOk ? "OK" : "MISMATCH");
  if (!vOk) ok = false;
  put("gputest: %s (10k ticks CPU vs GPU)\n", ok ? "PASS" : "FAIL");
  return ok ? 1 : 0;
}

void ndl_rt_bench_gpu(char* buf, int cap, int64_t max_neurons) {
  int pos = 0;
  auto put = [&](const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    if (buf && pos < cap) pos += std::vsnprintf(buf + pos, (size_t)std::max(0, cap - pos), fmt, args);
    va_end(args);
  };

  const char* dev = ndl_rt_ocl_device_name(nullptr);
  put("bench gpu: built-in sim, sparse E/I (in-degree ~200), streams + STDP\n");
  put("  device: %s\n", dev ? dev : "(none — CPU only)");
  put("  %10s %14s %14s %8s %10s\n", "neurons", "cpu ticks/s", "gpu ticks/s",
      "speedup", "x realtime");

  struct Row { int64_t nE, nI; const char* label; };
  const Row rows[] = {{2000, 500, "2.5k"},  {8000, 2000, "10k"},
                      {40000, 10000, "50k"}, {80000, 20000, "100k"}};

  SimState* brain = nullptr;
  for (const Row& r : rows) {
    if (r.nE + r.nI > max_neurons) break;
    const int64_t total = 64 + r.nE + r.nI + 64;

    const auto buildAndTime = [&](int32_t device) -> double {
      brain = ndl_rt_state_swap_fresh();
      SimState& st = ndl_rt_sim();
      ndl_rt_group_create(64, 0, 10.0, -55.0, -70.0, -75.0, "b_input");
      ndl_rt_group_create(r.nE, 0, 20.0, -55.0, -70.0, -75.0, "b_coreE");
      ndl_rt_group_create(r.nI, 1, 10.0, -50.0, -70.0, -75.0, "b_coreI");
      ndl_rt_group_create(64, 0, 20.0, -55.0, -70.0, -75.0, "b_read");
      const double dEE = std::min(0.06, 200.0 / (double)r.nE);
      const double dEI = std::min(0.15, 400.0 / (double)r.nE);
      const double dIE = std::min(0.20, 200.0 / (double)r.nE);
      ndl_rt_sparse_connect(st.groups[0], st.groups[1], std::min(0.05, 640.0 / (double)r.nE), 12.0, 0);
      ndl_rt_sparse_connect(st.groups[1], st.groups[1], dEE, 1.5, 0);
      ndl_rt_sparse_connect(st.groups[1], st.groups[2], dEI, 2.5, 0);
      ndl_rt_sparse_connect(st.groups[2], st.groups[1], dIE, -4.0, 0);
      ndl_rt_sparse_connect(st.groups[1], st.groups[3], 0.15, 3.0, 1);
      ndl_rt_sparse_connect(st.groups[3], st.groups[1], std::min(0.08, 200.0 / (double)r.nE), 1.0, 0);
      ndl_rt_configure_stdp(st.groups[1], st.groups[3], 0.12, 0.005, 20.0);
      ndl_rt_signal_create("dopamine", 0.5);
      ndl_rt_stdp_set_modulator(st.groups[1], st.groups[3], "dopamine");
      const int64_t sid = ndl_rt_stream_create("b_sense", 64);
      ndl_rt_bind_input_stream(sid, st.groups[0], 0, 200.0, 14.0);
      std::vector<double> vec((size_t)64, 0.5);
      ndl_rt_stream_push(sid, vec.data(), 64);

      ndl_rt_run(200.0, 1.0, device);  // warm-up
      const auto t0 = std::chrono::steady_clock::now();
      ndl_rt_run(1000.0, 1.0, device);
      const auto t1 = std::chrono::steady_clock::now();
      ndl_rt_state_restore(brain);
      const double wall = std::chrono::duration<double>(t1 - t0).count();
      return wall > 0 ? 1000.0 / wall : 0.0;
    };

    const double cpuTps = buildAndTime(0);
    double gpuTps = 0.0;
    if (dev) gpuTps = buildAndTime(1);
    const double spd = (gpuTps > 0 && cpuTps > 0) ? gpuTps / cpuTps : 0.0;
    const double xrt = gpuTps > 0 ? gpuTps / 1000.0 : cpuTps / 1000.0;
    put("  %10lld %14.0f %14.0f %8s %10.2f\n", (long long)total, cpuTps, gpuTps,
        gpuTps > 0 ? std::to_string(spd).c_str() : "-", xrt);
  }
  put("  (x realtime: ticks/s ÷ 1000; tempo 0.3 needs ≥ 0.30)\n");
}
