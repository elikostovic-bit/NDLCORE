// NDL v1.0 — GPU execution backend for libndl_rt.
//
// Strategy: the compiler emits a standalone PTX module (§8); the runtime loads
// it through the CUDA DRIVER API (cuModuleLoadData → in-driver JIT). No CUDA
// toolkit headers are required at build time — all driver entry points are
// resolved with dlopen/dlsym at runtime. If the driver, the device or the PTX
// is unavailable, ndl_rt_gpu_run returns false and the CPU path takes over
// (with a single one-time warning).
//
// GPU tick order mirrors the CPU loop (INTERNALS §5) exactly:
//   memset i_syn → propagate (dense/csr) → inject_apply → lif_step
//   → stdp_update (plastic dense conns with a config) → trace_update
//   → spike download → host spike log + handlers → prev = spikes
#include "ndl_rt_internal.h"

#if defined(_WIN32)
  #if !defined(WIN32_LEAN_AND_MEAN)
    #define WIN32_LEAN_AND_MEAN
  #endif
  #if !defined(NOMINMAX)
    #define NOMINMAX
  #endif
  #include <windows.h>  // LoadLibraryA / GetProcAddress (no dlfcn.h on Windows)
#else
  #include <dlfcn.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal CUDA driver API declarations (no cuda.h — resolved dynamically)
// ---------------------------------------------------------------------------
namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;  // 0 = default stream
using CUdeviceptr = unsigned long long;

constexpr CUresult CUDA_SUCCESS = 0;

using pfn_cuInit = CUresult (*)(unsigned);
using pfn_cuDeviceGet = CUresult (*)(CUdevice*, int);
using pfn_cuDevicePrimaryCtxRetain = CUresult (*)(CUcontext*, CUdevice);
using pfn_cuCtxSetCurrent = CUresult (*)(CUcontext);
using pfn_cuModuleLoadData = CUresult (*)(CUmodule*, const void*);
using pfn_cuModuleGetFunction = CUresult (*)(CUfunction*, CUmodule, const char*);
using pfn_cuMemAlloc = CUresult (*)(CUdeviceptr*, size_t);
using pfn_cuMemFree = CUresult (*)(CUdeviceptr);
using pfn_cuMemsetD8 = CUresult (*)(CUdeviceptr, unsigned char, size_t);
using pfn_cuMemcpyHtoD = CUresult (*)(CUdeviceptr, const void*, size_t);
using pfn_cuMemcpyDtoH = CUresult (*)(void*, CUdeviceptr, size_t);
using pfn_cuLaunchKernel = CUresult (*)(CUfunction, unsigned, unsigned, unsigned,
                                        unsigned, unsigned, unsigned, unsigned,
                                        CUstream, void**, void**);
using pfn_cuGetErrorString = CUresult (*)(CUresult, const char**);

struct Driver {
  void* handle = nullptr;
  pfn_cuInit cuInit = nullptr;
  pfn_cuDeviceGet cuDeviceGet = nullptr;
  pfn_cuDevicePrimaryCtxRetain cuDevicePrimaryCtxRetain = nullptr;
  pfn_cuCtxSetCurrent cuCtxSetCurrent = nullptr;
  pfn_cuModuleLoadData cuModuleLoadData = nullptr;
  pfn_cuModuleGetFunction cuModuleGetFunction = nullptr;
  pfn_cuMemAlloc cuMemAlloc = nullptr;
  pfn_cuMemFree cuMemFree = nullptr;
  pfn_cuMemsetD8 cuMemsetD8 = nullptr;
  pfn_cuMemcpyHtoD cuMemcpyHtoD = nullptr;
  pfn_cuMemcpyDtoH cuMemcpyDtoH = nullptr;
  pfn_cuLaunchKernel cuLaunchKernel = nullptr;
  pfn_cuGetErrorString cuGetErrorString = nullptr;

  bool ok() const { return cuInit && cuDeviceGet && cuDevicePrimaryCtxRetain &&
                           cuCtxSetCurrent && cuModuleLoadData && cuModuleGetFunction &&
                           cuMemAlloc && cuMemFree && cuMemsetD8 && cuMemcpyHtoD &&
                           cuMemcpyDtoH && cuLaunchKernel; }
};

template <typename T>
T symOrNull(void* h, const char* name) {
#if defined(_WIN32)
  // GetProcAddress returns FARPROC (already a function pointer type); casting
  // it directly to a different signature trips GCC's -Wcast-function-type
  // (on by default with -Wextra since GCC 8). Routing through void* is the
  // standard, warning-free pattern for dynamic symbol resolution — the call
  // itself is safe because each pointer is immediately used with its real
  // CUDA driver signature.
  FARPROC p = ::GetProcAddress(static_cast<HMODULE>(h), name);
  return reinterpret_cast<T>(reinterpret_cast<void*>(p));
#else
  return reinterpret_cast<T>(::dlsym(h, name));
#endif
}

bool loadDriver(Driver& d) {
#if defined(_WIN32)
  static const char* candidates[] = {"nvcuda.dll", nullptr};
  for (const char* name : candidates) {
    if (!name) break;
    d.handle = reinterpret_cast<void*>(::LoadLibraryA(name));
    if (d.handle) break;
  }
#else
  static const char* candidates[] = {"libcuda.so.1", "libcuda.so", nullptr};
  for (const char* name : candidates) {
    if (!name) break;
    d.handle = ::dlopen(name, RTLD_NOW | RTLD_LOCAL);
    if (d.handle) break;
  }
#endif
  if (!d.handle) return false;
  d.cuInit = symOrNull<pfn_cuInit>(d.handle, "cuInit");
  d.cuDeviceGet = symOrNull<pfn_cuDeviceGet>(d.handle, "cuDeviceGet");
  d.cuDevicePrimaryCtxRetain =
      symOrNull<pfn_cuDevicePrimaryCtxRetain>(d.handle, "cuDevicePrimaryCtxRetain");
  d.cuCtxSetCurrent = symOrNull<pfn_cuCtxSetCurrent>(d.handle, "cuCtxSetCurrent");
  d.cuModuleLoadData = symOrNull<pfn_cuModuleLoadData>(d.handle, "cuModuleLoadData");
  d.cuModuleGetFunction =
      symOrNull<pfn_cuModuleGetFunction>(d.handle, "cuModuleGetFunction");
  d.cuMemAlloc = symOrNull<pfn_cuMemAlloc>(d.handle, "cuMemAlloc");
  d.cuMemFree = symOrNull<pfn_cuMemFree>(d.handle, "cuMemFree");
  d.cuMemsetD8 = symOrNull<pfn_cuMemsetD8>(d.handle, "cuMemsetD8");
  d.cuMemcpyHtoD = symOrNull<pfn_cuMemcpyHtoD>(d.handle, "cuMemcpyHtoD");
  d.cuMemcpyDtoH = symOrNull<pfn_cuMemcpyDtoH>(d.handle, "cuMemcpyDtoH");
  d.cuLaunchKernel = symOrNull<pfn_cuLaunchKernel>(d.handle, "cuLaunchKernel");
  d.cuGetErrorString = symOrNull<pfn_cuGetErrorString>(d.handle, "cuGetErrorString");
  return d.ok();
}

}  // namespace

// ---------------------------------------------------------------------------
// Public ABI: PTX registration + fallback warning
// ---------------------------------------------------------------------------

void ndl_rt_gpu_load_ptx(const char* ptx_source) {
  if (!ptx_source) return;
  try {
    ndl_rt_sim().gpu_ptx = ptx_source;
  } catch (...) {
    // ignore: GPU path will simply refuse
  }
}

void ndl_rt_warn_no_cuda() {
  static std::mutex mu;
  SimState& st = ndl_rt_sim();
  std::lock_guard<std::mutex> lk(mu);
  if (st.warned_no_cuda) return;
  st.warned_no_cuda = true;
  std::fprintf(stderr,
               "warning: CUDA driver or NDL PTX module not available; "
               "falling back to CPU simulation\n");
}

// ---------------------------------------------------------------------------
// GPU run
// ---------------------------------------------------------------------------

namespace {

struct DevBuf {
  CUdeviceptr ptr = 0;
  size_t bytes = 0;
};

class GpuRun {
 public:
  bool execute(int64_t ticks, double dt_ms);

 private:
  Driver drv_;
  CUcontext ctx_ = nullptr;
  CUmodule mod_ = nullptr;
  CUfunction f_lif_ = nullptr, f_dense_ = nullptr, f_csr_ = nullptr;
  CUfunction f_inj_ = nullptr, f_stdp_ = nullptr, f_trace_ = nullptr;
  CUfunction f_hold_ = nullptr, f_syn_ = nullptr;

  const char* errStr(CUresult r) {
    const char* s = nullptr;
    if (drv_.cuGetErrorString) drv_.cuGetErrorString(r, &s);
    return s ? s : "unknown CUDA error";
  }
  bool fail(const std::string& what, CUresult r) {
    err_ = what + ": " + errStr(r) + " (code " + std::to_string(r) + ")";
    return false;
  }
  bool alloc(DevBuf& b, size_t bytes) {
    b.bytes = bytes;
    if (bytes == 0) { b.ptr = 0; return true; }
    return drv_.cuMemAlloc(&b.ptr, bytes) == CUDA_SUCCESS;
  }
  bool launch(CUfunction f, unsigned grid, unsigned block, void** params) {
    CUresult r = drv_.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, nullptr, params, nullptr);
    return r == CUDA_SUCCESS;
  }
  static unsigned gridFor(int64_t n, unsigned block) {
    if (n <= 0) return 0;
    return (unsigned)((n + (int64_t)block - 1) / (int64_t)block);
  }

  std::string err_;
  std::vector<DevBuf> keepalive_;  // all device allocations freed in dtor
 public:
  const std::string& error() const { return err_; }
  ~GpuRun();
};

GpuRun::~GpuRun() {
  if (drv_.cuMemFree)
    for (DevBuf& b : keepalive_)
      if (b.ptr) drv_.cuMemFree(b.ptr);
  // Intentionally do NOT destroy the primary context (it is process-wide and
  // may be reused by subsequent runs).
}

bool GpuRun::execute(int64_t ticks, double dt_ms) {
  SimState& st = ndl_rt_sim();

  // Preconditions — refuse (fall back to CPU) when unsupported.
  if (st.gpu_ptx.empty()) { err_ = "no PTX module registered"; return false; }
  if (ticks < 1) return true;  // nothing to do; treat as success

  for (const Connection* c : st.conns) {
    if (c->kind == CONN_ONE2ONE) { err_ = "one_to_one connections unsupported on GPU v1.0"; return false; }
    if (c->kind == CONN_SPARSE && c->plastic) { err_ = "plastic sparse connections unsupported on GPU v1.0"; return false; }
    if (c->kind == CONN_DENSE && c->plastic && !ndl_rt_find_stdp(st, c->src, c->dst)) {
      // plastic without stdp config: CPU path skips STDP in this case too — OK.
    }
  }

  if (!loadDriver(drv_)) { err_ = "CUDA driver library not found"; return false; }
  CUresult r = drv_.cuInit(0);
  if (r != CUDA_SUCCESS) return fail("cuInit", r);
  CUdevice dev = 0;
  r = drv_.cuDeviceGet(&dev, 0);
  if (r != CUDA_SUCCESS) return fail("cuDeviceGet", r);
  r = drv_.cuDevicePrimaryCtxRetain(&ctx_, dev);
  if (r != CUDA_SUCCESS) return fail("cuDevicePrimaryCtxRetain", r);
  r = drv_.cuCtxSetCurrent(ctx_);
  if (r != CUDA_SUCCESS) return fail("cuCtxSetCurrent", r);

  // JIT the PTX module.
  r = drv_.cuModuleLoadData(&mod_, st.gpu_ptx.c_str());
  if (r != CUDA_SUCCESS) return fail("cuModuleLoadData (PTX JIT)", r);
  struct { CUfunction* fn; const char* name; } kernels[] = {
      {&f_lif_, "ndl_gpu_lif_step"},         {&f_dense_, "ndl_gpu_propagate_dense"},
      {&f_csr_, "ndl_gpu_propagate_csr"},    {&f_inj_, "ndl_gpu_inject_apply"},
      {&f_stdp_, "ndl_gpu_stdp_update"},     {&f_trace_, "ndl_gpu_trace_update"},
      {&f_hold_, "ndl_gpu_hold_apply"},      {&f_syn_, "ndl_gpu_syn_decay"},
  };
  for (auto& k : kernels) {
    r = drv_.cuModuleGetFunction(k.fn, mod_, k.name);
    if (r != CUDA_SUCCESS) return fail(std::string("cuModuleGetFunction(") + k.name + ")", r);
  }

  constexpr unsigned kBlock = 128;

  // --- device buffers per group -------------------------------------------
  struct GroupBuf {
    DevBuf v, syn, hold, ieff, spikes_cur, spikes_prev, trace;
  };
  std::vector<GroupBuf> gb(st.groups.size());
  for (size_t gi = 0; gi < st.groups.size(); ++gi) {
    NdlGroup* g = st.groups[gi];
    const size_t n = (size_t)g->size;
    if (!alloc(gb[gi].v, n * 8) || !alloc(gb[gi].syn, n * 8) ||
        !alloc(gb[gi].hold, n * 8) || !alloc(gb[gi].ieff, n * 8) ||
        !alloc(gb[gi].spikes_cur, n) || !alloc(gb[gi].spikes_prev, n) ||
        !alloc(gb[gi].trace, n * 8)) {
      err_ = "cuMemAlloc failed for group buffers";
      return false;
    }
    if (drv_.cuMemcpyHtoD(gb[gi].v.ptr, g->v.data(), n * 8) != CUDA_SUCCESS ||
        drv_.cuMemcpyHtoD(gb[gi].spikes_prev.ptr, g->prev_spikes.data(), n) != CUDA_SUCCESS ||
        drv_.cuMemsetD8(gb[gi].syn.ptr, 0, n * 8) != CUDA_SUCCESS ||
        drv_.cuMemsetD8(gb[gi].hold.ptr, 0, n * 8) != CUDA_SUCCESS ||
        drv_.cuMemsetD8(gb[gi].ieff.ptr, 0, n * 8) != CUDA_SUCCESS ||
        drv_.cuMemsetD8(gb[gi].spikes_cur.ptr, 0, n) != CUDA_SUCCESS ||
        drv_.cuMemsetD8(gb[gi].trace.ptr, 0, n * 8) != CUDA_SUCCESS) {
      err_ = "initial HtoD copy failed";
      return false;
    }
    keepalive_.push_back(gb[gi].v); keepalive_.push_back(gb[gi].syn);
    keepalive_.push_back(gb[gi].hold); keepalive_.push_back(gb[gi].ieff);
    keepalive_.push_back(gb[gi].spikes_cur); keepalive_.push_back(gb[gi].spikes_prev);
    keepalive_.push_back(gb[gi].trace);
  }

  // --- device buffers per connection --------------------------------------
  struct ConnBuf {
    DevBuf w;        // dense (dst-major) or one-to-one
    DevBuf rowptr, cols, vals;  // sparse
  };
  std::vector<ConnBuf> cb(st.conns.size());
  for (size_t ci = 0; ci < st.conns.size(); ++ci) {
    const Connection* c = st.conns[ci];
    if (c->kind == CONN_DENSE) {
      const size_t bytes = (size_t)c->w.size() * 8;
      if (!alloc(cb[ci].w, bytes)) { err_ = "cuMemAlloc failed (dense weights)"; return false; }
      if (drv_.cuMemcpyHtoD(cb[ci].w.ptr, c->w.data(), bytes) != CUDA_SUCCESS) {
        err_ = "weight upload failed"; return false;
      }
      keepalive_.push_back(cb[ci].w);
    } else if (c->kind == CONN_SPARSE) {
      const size_t nrp = c->rowptr.size() * 8, nc = c->cols.size() * 4, nv = c->vals.size() * 8;
      if (!alloc(cb[ci].rowptr, nrp) || !alloc(cb[ci].cols, nc) || !alloc(cb[ci].vals, nv)) {
        err_ = "cuMemAlloc failed (CSR)"; return false;
      }
      if (drv_.cuMemcpyHtoD(cb[ci].rowptr.ptr, c->rowptr.data(), nrp) != CUDA_SUCCESS ||
          drv_.cuMemcpyHtoD(cb[ci].cols.ptr, c->cols.data(), nc) != CUDA_SUCCESS ||
          drv_.cuMemcpyHtoD(cb[ci].vals.ptr, c->vals.data(), nv) != CUDA_SUCCESS) {
        err_ = "CSR upload failed"; return false;
      }
      keepalive_.push_back(cb[ci].rowptr);
      keepalive_.push_back(cb[ci].cols);
      keepalive_.push_back(cb[ci].vals);
    }
  }

  // Per-tick trace decay (dt fixed for the run), same rule as the CPU path.
  const size_t nG = st.groups.size();
  std::vector<double> decay(nG, 0.0);
  for (size_t gi = 0; gi < nG; ++gi)
    decay[gi] = std::exp(-dt_ms / ndl_rt_effective_window(st, st.groups[gi]));

  std::vector<const StdpCfg*> conn_cfg(st.conns.size(), nullptr);
  for (size_t ci = 0; ci < st.conns.size(); ++ci)
    conn_cfg[ci] = ndl_rt_find_stdp(st, st.conns[ci]->src, st.conns[ci]->dst);

  // Host-side staging for injections: inj_pos advances over the sorted queue;
  // per tick we re-scan the [inj_scan_start, inj_pos) window per group.
  size_t inj_pos = 0;
  const size_t n_inj = st.injections.size();
  std::vector<uint8_t> host_spikes;

  for (int64_t tick = 0; tick < ticks; ++tick) {
    const double t0 = (double)tick * dt_ms;
    const double t1 = t0 + dt_ms;

    // (1) exponential synaptic decay: syn *= exp(-dt/tau_syn)
    {
      const double synDecay = std::exp(-dt_ms / 5.0);
      for (size_t gi = 0; gi < nG; ++gi) {
        CUdeviceptr sptr = gb[gi].syn.ptr;
        double dec = synDecay;
        unsigned n = (unsigned)st.groups[gi]->size;
        void* params[] = {&sptr, &dec, &n};
        if (!launch(f_syn_, gridFor(n, kBlock), kBlock, params)) { err_ = "syn_decay launch failed"; return false; }
      }
    }

    // (2) propagate per connection
    for (size_t ci = 0; ci < st.conns.size(); ++ci) {
      const Connection* c = st.conns[ci];
      const NdlGroup* pre = c->src;
      const NdlGroup* post = c->dst;
      // resolve group buffers by pointer identity
      size_t pre_i = 0, post_i = 0;
      for (size_t gi = 0; gi < nG; ++gi) {
        if (st.groups[gi] == c->src) pre_i = gi;
        if (st.groups[gi] == c->dst) post_i = gi;
      }
      GroupBuf& sb = gb[pre_i];
      GroupBuf& dbuf = gb[post_i];
      if (c->kind == CONN_DENSE) {
        CUdeviceptr wptr = cb[ci].w.ptr;
        CUdeviceptr sptr = sb.spikes_prev.ptr;
        CUdeviceptr iptr = dbuf.syn.ptr;
        unsigned nsrc = (unsigned)pre->size, ndst = (unsigned)post->size;
        void* params[] = {&wptr, &sptr, &iptr, &nsrc, &ndst};
        if (!launch(f_dense_, gridFor(ndst, kBlock), kBlock, params)) { err_ = "propagate_dense launch failed"; return false; }
      } else if (c->kind == CONN_SPARSE) {
        CUdeviceptr rptr = cb[ci].rowptr.ptr, cptr = cb[ci].cols.ptr, vptr = cb[ci].vals.ptr;
        CUdeviceptr sptr = sb.spikes_prev.ptr, iptr = dbuf.syn.ptr;
        unsigned ndst = (unsigned)post->size;
        void* params[] = {&rptr, &cptr, &vptr, &sptr, &iptr, &ndst};
        if (!launch(f_csr_, gridFor(ndst, kBlock), kBlock, params)) { err_ = "propagate_csr launch failed"; return false; }
      }
    }

    // (3) injections with t in [t0, t1): aggregated per neuron per group so
    // that each index appears at most once per tick → race-free scatter.
    const size_t inj_scan_start = inj_pos;
    while (inj_pos < n_inj && st.injections[inj_pos].t < t1) ++inj_pos;
    for (size_t gi = 0; gi < nG; ++gi) {
      NdlGroup* g = st.groups[gi];
      std::vector<unsigned long long> idxs;
      std::vector<double> curs;
      for (size_t q = inj_scan_start; q < inj_pos; ++q) {
        const Injection& inj = st.injections[q];
        if (inj.g != g) continue;
        if (!(inj.t >= t0 && inj.t < t1)) continue;
        int64_t lo = std::max<int64_t>(inj.lo, 0);
        int64_t hi = std::min<int64_t>(inj.hi, g->size);
        for (int64_t k = lo; k < hi; ++k) {
          bool merged = false;
          for (size_t z = 0; z < idxs.size(); ++z)
            if (idxs[z] == (unsigned long long)k) { curs[z] += inj.cur; merged = true; break; }
          if (!merged) { idxs.push_back((unsigned long long)k); curs.push_back(inj.cur); }
        }
      }
      if (idxs.empty()) continue;
      DevBuf ib, cb2;
      if (!alloc(ib, idxs.size() * 8) || !alloc(cb2, curs.size() * 8)) { err_ = "injection upload failed"; return false; }
      if (drv_.cuMemcpyHtoD(ib.ptr, idxs.data(), idxs.size() * 8) != CUDA_SUCCESS ||
          drv_.cuMemcpyHtoD(cb2.ptr, curs.data(), curs.size() * 8) != CUDA_SUCCESS) {
        err_ = "injection upload failed"; return false;
      }
      keepalive_.push_back(ib); keepalive_.push_back(cb2);
      CUdeviceptr iptr = ib.ptr, cptr = cb2.ptr, hptr = gb[gi].hold.ptr;
      unsigned count = (unsigned)idxs.size();
      void* params[] = {&iptr, &cptr, &count, &hptr};
      if (!launch(f_inj_, gridFor(count, kBlock), kBlock, params)) { err_ = "inject_apply launch failed"; return false; }
    }

    // (3b) effective current: i_eff = syn + hold
    for (size_t gi = 0; gi < nG; ++gi) {
      CUdeviceptr sptr = gb[gi].syn.ptr, hptr = gb[gi].hold.ptr, eptr = gb[gi].ieff.ptr;
      unsigned n = (unsigned)st.groups[gi]->size;
      void* params[] = {&sptr, &hptr, &eptr, &n};
      if (!launch(f_hold_, gridFor(n, kBlock), kBlock, params)) { err_ = "hold_apply launch failed"; return false; }
    }

    // (4) LIF step per group (reads i_eff)
    for (size_t gi = 0; gi < nG; ++gi) {
      NdlGroup* g = st.groups[gi];
      CUdeviceptr vptr = gb[gi].v.ptr, iptr = gb[gi].ieff.ptr, sptr = gb[gi].spikes_cur.ptr;
      double tau = g->tau, thr = g->threshold, rest = g->rest, reset = g->reset, dt = dt_ms;
      unsigned n = (unsigned)g->size;
      void* params[] = {&vptr, &iptr, &sptr, &tau, &thr, &rest, &reset, &dt, &n};
      if (!launch(f_lif_, gridFor(n, kBlock), kBlock, params)) { err_ = "lif_step launch failed"; return false; }
    }

    // (5) STDP on plastic dense connections with a config
    for (size_t ci = 0; ci < st.conns.size(); ++ci) {
      const Connection* c = st.conns[ci];
      const StdpCfg* cfg = conn_cfg[ci];
      if (!c->plastic || !cfg || c->kind != CONN_DENSE) continue;
      size_t pre_i = 0, post_i = 0;
      for (size_t gi = 0; gi < nG; ++gi) {
        if (st.groups[gi] == c->src) pre_i = gi;
        if (st.groups[gi] == c->dst) post_i = gi;
      }
      CUdeviceptr wptr = cb[ci].w.ptr;
      CUdeviceptr trs = gb[pre_i].trace.ptr, trd = gb[post_i].trace.ptr;
      CUdeviceptr sprev = gb[pre_i].spikes_prev.ptr, snew = gb[post_i].spikes_cur.ptr;
      double lrp = cfg->lr_pot, lrd = cfg->lr_dep;
      unsigned nsrc = (unsigned)c->src->size, ndst = (unsigned)c->dst->size;
      void* params[] = {&wptr, &trs, &trd, &sprev, &snew, &lrp, &lrd, &nsrc, &ndst};
      if (!launch(f_stdp_, gridFor(ndst, kBlock), kBlock, params)) { err_ = "stdp_update launch failed"; return false; }
    }

    // (6) traces
    for (size_t gi = 0; gi < nG; ++gi) {
      CUdeviceptr trptr = gb[gi].trace.ptr, sptr = gb[gi].spikes_cur.ptr;
      double dec = decay[gi];
      unsigned n = (unsigned)st.groups[gi]->size;
      void* params[] = {&trptr, &sptr, &dec, &n};
      if (!launch(f_trace_, gridFor(n, kBlock), kBlock, params)) { err_ = "trace_update launch failed"; return false; }
    }

    // (7) spike download → log + handlers (host side, same as CPU)
    for (size_t gi = 0; gi < nG; ++gi) {
      NdlGroup* g = st.groups[gi];
      const size_t n = (size_t)g->size;
      host_spikes.resize(n);
      if (drv_.cuMemcpyDtoH(host_spikes.data(), gb[gi].spikes_cur.ptr, n) != CUDA_SUCCESS) {
        err_ = "spike download failed"; return false;
      }
      for (size_t i = 0; i < n; ++i)
        if (host_spikes[i]) ndl_rt_log_spike(g, t0, (int64_t)i);
      if (!g->handlers.empty()) {
        const std::vector<NdlGroup::Handler> handlers = g->handlers;
        for (const NdlGroup::Handler& h : handlers)
          for (size_t i = 0; i < n; ++i)
            if (host_spikes[i]) h.fn((int64_t)i, t0, h.user);
      }
      g->prev_spikes = host_spikes;  // (8) host mirror stays in sync
    }

    // (8) device-side swap: prev ← current spikes
    for (size_t gi = 0; gi < nG; ++gi)
      std::swap(gb[gi].spikes_cur, gb[gi].spikes_prev);
  }

  // Final download: membrane potentials + traces.
  for (size_t gi = 0; gi < nG; ++gi) {
    NdlGroup* g = st.groups[gi];
    if (drv_.cuMemcpyDtoH(g->v.data(), gb[gi].v.ptr, (size_t)g->size * 8) != CUDA_SUCCESS ||
        drv_.cuMemcpyDtoH(g->trace.data(), gb[gi].trace.ptr, (size_t)g->size * 8) != CUDA_SUCCESS) {
      err_ = "final download failed"; return false;
    }
  }

  // Phase 2 fix: plastic weights live on the device during the run — download
  // them so the host mirrors stay authoritative between runs. (Previously only
  // v/trace were synced; device-learned weight changes were lost on the next
  // run's re-upload.)
  for (size_t ci = 0; ci < st.conns.size(); ++ci) {
    Connection* c = st.conns[ci];
    if (!c->plastic || cb[ci].w.ptr == 0) continue;  // sparse plastic is refused above
    if (drv_.cuMemcpyDtoH(c->w.data(), cb[ci].w.ptr, c->w.size() * 8) != CUDA_SUCCESS) {
      err_ = "weight download failed"; return false;
    }
  }

  return true;
}

}  // namespace

bool ndl_rt_gpu_run(int64_t ticks, double dt_ms) {
  GpuRun run;
  const bool ok = run.execute(ticks, dt_ms);
  static bool detailed_warned = false;
  if (!ok && !run.error().empty() && !detailed_warned) {
    // Detailed reason helps users diagnose their GPU setup; printed once.
    detailed_warned = true;
    std::fprintf(stderr, "warning: GPU run unavailable: %s\n", run.error().c_str());
  }
  return ok;
}
