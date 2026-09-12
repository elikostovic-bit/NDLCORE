// NDL v2.1 — OpenCL backend for the built-in simulation (Phase 2).
//
// Target: AMD RX 6600 XT (RDNA2) — the driver ships OpenCL 2.x, so on the
// owner's Windows machine nothing needs to be installed. No OpenCL SDK is
// required at BUILD time either: the ICD loader (OpenCL.dll / libOpenCL.so.1)
// is resolved dynamically, exactly like the CUDA driver path, and the kernels
// are compiled by the in-driver OpenCL C compiler from runtime/ndl_ocl_kernels.cl
// (searched next to the executable; override with NDL_OCL_KERNELS).
//
// The device owns v / trace / weights / noise-RNG while an episode runs; the
// host keeps per-tick spike mirrors (logs + handlers + prev_spikes). Because
// the kernels are double-precision, cl_khr_fp64 is REQUIRED (the RX 6600 XT
// provides it); without fp64 the backend refuses and the CPU path takes over.
//
// Tick order mirrors ndl_rt_tick_cpu exactly:
//   drain_commands → syn_decay → propagate (dense/csr/o2o) → inject_range →
//   osc_add → stream_poisson → lif → stdp ×M(t) (dense/csr/o2o) → trace →
//   spike download + host logs/handlers → prev = cur (pointer swap)
#include "ndl_rt_internal.h"

#if defined(_WIN32)
  #if !defined(WIN32_LEAN_AND_MEAN)
    #define WIN32_LEAN_AND_MEAN
  #endif
  #if !defined(NOMINMAX)
    #define NOMINMAX
  #endif
  #include <windows.h>
#else
  #include <dlfcn.h>
  #include <unistd.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal OpenCL API surface (no CL headers — resolved dynamically)
// ---------------------------------------------------------------------------
namespace {

using cl_int = int32_t;
using cl_uint = uint32_t;
using cl_bool = cl_uint;
using cl_platform_id = void*;
using cl_device_id = void*;
using cl_context = void*;
using cl_command_queue = void*;
using cl_mem = void*;
using cl_program = void*;
using cl_kernel = void*;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_int CL_TRUE = 1;
constexpr cl_uint CL_MEM_READ_WRITE = 1;
constexpr cl_uint CL_MEM_READ_ONLY = 4;
constexpr cl_uint CL_MEM_COPY_HOST_PTR = 16;
constexpr cl_uint CL_DEVICE_NAME = 0x102B;
constexpr cl_uint CL_DEVICE_EXTENSIONS = 0x102F;
constexpr cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;

using pfn_clGetPlatformIDs = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
using pfn_clGetDeviceIDs = cl_int (*)(cl_platform_id, cl_uint, cl_uint,
                                      cl_device_id*, cl_uint*);
using pfn_clCreateContext = cl_context (*)(const void*, cl_uint, cl_device_id*,
                                           void (*)(const char*, const void*,
                                                    size_t, void*),
                                           void*, cl_int*);
using pfn_clCreateCommandQueue = cl_command_queue (*)(cl_context, cl_device_id,
                                                      cl_uint, cl_int*);
using pfn_clCreateBuffer = cl_mem (*)(cl_context, cl_uint, size_t, void*,
                                      cl_int*);
using pfn_clCreateProgramWithSource = cl_program (*)(cl_context, cl_uint,
                                                     const char**, const size_t*,
                                                     cl_int*);
using pfn_clBuildProgram = cl_int (*)(cl_program, cl_uint, cl_device_id*,
                                      const char*,
                                      void (*)(cl_program, void*), void*);
using pfn_clGetProgramBuildInfo = cl_int (*)(cl_program, cl_device_id, cl_uint,
                                             size_t, void*, size_t*);
using pfn_clGetDeviceInfo = cl_int (*)(cl_device_id, cl_uint, size_t, void*,
                                       size_t*);
using pfn_clCreateKernel = cl_kernel (*)(cl_program, const char*, cl_int*);
using pfn_clSetKernelArg = cl_int (*)(cl_kernel, cl_uint, size_t, const void*);
using pfn_clEnqueueNDRangeKernel = cl_int (*)(cl_command_queue, cl_kernel,
                                              cl_uint, const size_t*,
                                              const size_t*, const size_t*,
                                              cl_uint, const cl_mem*,
                                              const void**);
using pfn_clEnqueueReadBuffer = cl_int (*)(cl_command_queue, cl_mem, cl_bool,
                                           size_t, size_t, void*, cl_uint,
                                           const cl_mem*, const void**);
using pfn_clEnqueueWriteBuffer = cl_int (*)(cl_command_queue, cl_mem, cl_bool,
                                            size_t, size_t, const void*,
                                            cl_uint, const cl_mem*,
                                            const void**);
using pfn_clFinish = cl_int (*)(cl_command_queue);
using pfn_clReleaseMemObject = cl_int (*)(cl_mem);
using pfn_clReleaseKernel = cl_int (*)(cl_kernel);
using pfn_clReleaseProgram = cl_int (*)(cl_program);
using pfn_clReleaseContext = cl_int (*)(cl_context);
using pfn_clReleaseCommandQueue = cl_int (*)(cl_command_queue);

struct OclApi {
  void* handle = nullptr;
  pfn_clGetPlatformIDs GetPlatformIDs = nullptr;
  pfn_clGetDeviceIDs GetDeviceIDs = nullptr;
  pfn_clCreateContext CreateContext = nullptr;
  pfn_clCreateCommandQueue CreateCommandQueue = nullptr;
  pfn_clCreateBuffer CreateBuffer = nullptr;
  pfn_clCreateProgramWithSource CreateProgramWithSource = nullptr;
  pfn_clBuildProgram BuildProgram = nullptr;
  pfn_clGetProgramBuildInfo GetProgramBuildInfo = nullptr;
  pfn_clGetDeviceInfo GetDeviceInfo = nullptr;
  pfn_clCreateKernel CreateKernel = nullptr;
  pfn_clSetKernelArg SetKernelArg = nullptr;
  pfn_clEnqueueNDRangeKernel EnqueueNDRangeKernel = nullptr;
  pfn_clEnqueueReadBuffer EnqueueReadBuffer = nullptr;
  pfn_clEnqueueWriteBuffer EnqueueWriteBuffer = nullptr;
  pfn_clFinish Finish = nullptr;
  pfn_clReleaseMemObject ReleaseMemObject = nullptr;
  pfn_clReleaseKernel ReleaseKernel = nullptr;
  pfn_clReleaseProgram ReleaseProgram = nullptr;
  pfn_clReleaseContext ReleaseContext = nullptr;
  pfn_clReleaseCommandQueue ReleaseCommandQueue = nullptr;

  bool ok() const {
    return GetPlatformIDs && GetDeviceIDs && CreateContext && CreateCommandQueue &&
           CreateBuffer && CreateProgramWithSource && BuildProgram &&
           GetProgramBuildInfo && GetDeviceInfo && CreateKernel && SetKernelArg &&
           EnqueueNDRangeKernel && EnqueueReadBuffer && EnqueueWriteBuffer &&
           Finish && ReleaseMemObject && ReleaseKernel && ReleaseProgram &&
           ReleaseContext && ReleaseCommandQueue;
  }
};

template <typename T>
T oclSym(void* h, const char* name) {
#if defined(_WIN32)
  FARPROC p = ::GetProcAddress(static_cast<HMODULE>(h), name);
  return reinterpret_cast<T>(reinterpret_cast<void*>(p));
#else
  return reinterpret_cast<T>(::dlsym(h, name));
#endif
}

bool oclLoad(OclApi& api) {
#if defined(_WIN32)
  auto tryLoad = [&](const char* name) -> void* {
    return reinterpret_cast<void*>(::LoadLibraryA(name));
  };
  const char* candidates[] = {"OpenCL.dll", nullptr};
#else
  auto tryLoad = [&](const char* name) -> void* {
    return ::dlopen(name, RTLD_NOW | RTLD_LOCAL);
  };
  const char* candidates[] = {"libOpenCL.so.1", "libOpenCL.so", nullptr};
#endif
  if (const char* env = std::getenv("NDL_OCL_LIBRARY")) api.handle = tryLoad(env);
  if (!api.handle) {
    for (const char* name : candidates) {
      if (!name) break;
      api.handle = tryLoad(name);
      if (api.handle) break;
    }
  }
  if (!api.handle) return false;
  api.GetPlatformIDs = oclSym<pfn_clGetPlatformIDs>(api.handle, "clGetPlatformIDs");
  api.GetDeviceIDs = oclSym<pfn_clGetDeviceIDs>(api.handle, "clGetDeviceIDs");
  api.CreateContext = oclSym<pfn_clCreateContext>(api.handle, "clCreateContext");
  api.CreateCommandQueue =
      oclSym<pfn_clCreateCommandQueue>(api.handle, "clCreateCommandQueue");
  api.CreateBuffer = oclSym<pfn_clCreateBuffer>(api.handle, "clCreateBuffer");
  api.CreateProgramWithSource =
      oclSym<pfn_clCreateProgramWithSource>(api.handle, "clCreateProgramWithSource");
  api.BuildProgram = oclSym<pfn_clBuildProgram>(api.handle, "clBuildProgram");
  api.GetProgramBuildInfo =
      oclSym<pfn_clGetProgramBuildInfo>(api.handle, "clGetProgramBuildInfo");
  api.GetDeviceInfo = oclSym<pfn_clGetDeviceInfo>(api.handle, "clGetDeviceInfo");
  api.CreateKernel = oclSym<pfn_clCreateKernel>(api.handle, "clCreateKernel");
  api.SetKernelArg = oclSym<pfn_clSetKernelArg>(api.handle, "clSetKernelArg");
  api.EnqueueNDRangeKernel =
      oclSym<pfn_clEnqueueNDRangeKernel>(api.handle, "clEnqueueNDRangeKernel");
  api.EnqueueReadBuffer =
      oclSym<pfn_clEnqueueReadBuffer>(api.handle, "clEnqueueReadBuffer");
  api.EnqueueWriteBuffer =
      oclSym<pfn_clEnqueueWriteBuffer>(api.handle, "clEnqueueWriteBuffer");
  api.Finish = oclSym<pfn_clFinish>(api.handle, "clFinish");
  api.ReleaseMemObject =
      oclSym<pfn_clReleaseMemObject>(api.handle, "clReleaseMemObject");
  api.ReleaseKernel = oclSym<pfn_clReleaseKernel>(api.handle, "clReleaseKernel");
  api.ReleaseProgram = oclSym<pfn_clReleaseProgram>(api.handle, "clReleaseProgram");
  api.ReleaseContext = oclSym<pfn_clReleaseContext>(api.handle, "clReleaseContext");
  api.ReleaseCommandQueue =
      oclSym<pfn_clReleaseCommandQueue>(api.handle, "clReleaseCommandQueue");
  return api.ok();
}

// Kernel source: ndl_ocl_kernels.cl, searched next to the executable first.
std::string readWholeFile(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return std::string();
  std::string s;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
  std::fclose(f);
  return s;
}

std::string findKernelSource() {
  if (const char* env = std::getenv("NDL_OCL_KERNELS")) {
    std::string s = readWholeFile(env);
    if (!s.empty()) return s;
  }
  std::string exedir;
#if defined(_WIN32)
  char path[MAX_PATH] = {0};
  if (::GetModuleFileNameA(nullptr, path, MAX_PATH) > 0) exedir = path;
#else
  char path[4096];
  const ssize_t len = ::readlink("/proc/self/exe", path, sizeof(path) - 1);
  if (len > 0) {
    path[len] = '\0';
    exedir = path;
  }
#endif
  if (!exedir.empty()) {
    const size_t slash = exedir.find_last_of("/\\");
    if (slash != std::string::npos) exedir.resize(slash);
    else exedir.clear();
  }
  const char* rels[] = {"", "..", nullptr};
  for (int i = 0; rels[i]; ++i) {
    const std::string p = (exedir.empty() ? std::string(".") : exedir) +
                          (rels[i][0] ? std::string("/") + rels[i] : std::string()) +
                          "/ndl_ocl_kernels.cl";
    std::string s = readWholeFile(p.c_str());
    if (!s.empty()) return s;
  }
  return std::string();
}

// ---------------------------------------------------------------------------
// Device context (created once per process, reused across episodes)
// ---------------------------------------------------------------------------
struct Kernels {
  cl_kernel syn_decay = nullptr, prop_dense = nullptr, prop_csr = nullptr,
            prop_o2o = nullptr, inject_range = nullptr, osc_add = nullptr,
            stream_poisson = nullptr, lif = nullptr, stdp_dense = nullptr,
            stdp_csr = nullptr, stdp_o2o = nullptr, trace = nullptr,
            norm_rowsum_csr = nullptr, norm_scale_csr = nullptr,
            norm_rowsum_dense = nullptr, norm_scale_dense = nullptr,
            norm_rowsum_o2o = nullptr, norm_scale_o2o = nullptr;
};

struct OclDev {
  bool tried = false;
  bool ready = false;
  std::string refuse;
  OclApi api;
  cl_context ctx = nullptr;
  cl_command_queue q = nullptr;
  cl_device_id dev = nullptr;
  cl_program prog = nullptr;
  Kernels k;
  std::string name;

  void releaseProgram() {
    if (prog && api.ReleaseProgram) api.ReleaseProgram(prog);
    prog = nullptr;
    cl_kernel* ks[] = {&k.syn_decay, &k.prop_dense, &k.prop_csr, &k.prop_o2o,
                       &k.inject_range, &k.osc_add, &k.stream_poisson, &k.lif,
                       &k.stdp_dense, &k.stdp_csr, &k.stdp_o2o, &k.trace,
                       &k.norm_rowsum_csr, &k.norm_scale_csr,
                       &k.norm_rowsum_dense, &k.norm_scale_dense,
                       &k.norm_rowsum_o2o, &k.norm_scale_o2o};
    for (cl_kernel* kk : ks) {
      if (*kk && api.ReleaseKernel) api.ReleaseKernel(*kk);
      *kk = nullptr;
    }
  }
};

OclDev g_dev;
std::mutex g_dev_mu;

bool devInit(std::string& err) {
  std::lock_guard<std::mutex> lk(g_dev_mu);
  if (g_dev.tried) {
    err = g_dev.refuse;
    return g_dev.ready;
  }
  g_dev.tried = true;
  OclDev& D = g_dev;
  if (!oclLoad(D.api)) {
    D.refuse = "OpenCL loader not found (OpenCL.dll / libOpenCL.so.1)";
    err = D.refuse;
    return false;
  }
  cl_uint got = 0;
  cl_platform_id plats[8] = {nullptr};
  cl_int r = D.api.GetPlatformIDs(8, plats, &got);
  if (r != CL_SUCCESS || got == 0) {
    D.refuse = "no OpenCL platform";
    err = D.refuse;
    return false;
  }
  cl_uint platIdx = 0, devIdx = 0;
  if (const char* e = std::getenv("NDL_OCL_PLATFORM")) platIdx = (cl_uint)std::atoi(e);
  if (const char* e = std::getenv("NDL_OCL_DEVICE")) devIdx = (cl_uint)std::atoi(e);
  if (platIdx >= got) platIdx = 0;
  cl_device_id devs[8] = {nullptr};
  cl_uint ndev = 0;
  r = D.api.GetDeviceIDs(plats[platIdx], 0xFFFFFFFF /*ALL*/, 8, devs, &ndev);
  if (r != CL_SUCCESS || ndev == 0) {
    D.refuse = "no OpenCL device";
    err = D.refuse;
    return false;
  }
  if (devIdx >= ndev) devIdx = 0;
  D.dev = devs[devIdx];

  cl_int cerr = 0;
  D.ctx = D.api.CreateContext(nullptr, 1, &D.dev, nullptr, nullptr, &cerr);
  if (!D.ctx || cerr != CL_SUCCESS) {
    D.refuse = "clCreateContext failed";
    err = D.refuse;
    return false;
  }
  D.q = D.api.CreateCommandQueue(D.ctx, D.dev, 0, &cerr);
  if (!D.q || cerr != CL_SUCCESS) {
    D.refuse = "clCreateCommandQueue failed";
    err = D.refuse;
    return false;
  }
  // fp64 is mandatory: the whole model is double precision
  char exts[8192] = {0};
  size_t sz = 0;
  D.api.GetDeviceInfo(D.dev, CL_DEVICE_EXTENSIONS, sizeof(exts) - 1, exts, &sz);
  if (!std::strstr(exts, "cl_khr_fp64")) {
    D.refuse = "device lacks cl_khr_fp64 (double precision)";
    err = D.refuse;
    return false;
  }
  char name[512] = {0};
  D.api.GetDeviceInfo(D.dev, CL_DEVICE_NAME, sizeof(name) - 1, name, &sz);
  D.name = name;

  const std::string src = findKernelSource();
  if (src.empty()) {
    D.refuse = "ndl_ocl_kernels.cl not found (put it next to the executable "
               "or set NDL_OCL_KERNELS)";
    err = D.refuse;
    return false;
  }
  const char* srcPtr = src.c_str();
  const size_t srcLen = src.size();
  D.prog = D.api.CreateProgramWithSource(D.ctx, 1, &srcPtr, &srcLen, &cerr);
  if (!D.prog || cerr != CL_SUCCESS) {
    D.refuse = "clCreateProgramWithSource failed";
    err = D.refuse;
    return false;
  }
  r = D.api.BuildProgram(D.prog, 1, &D.dev, nullptr, nullptr, nullptr);
  if (r != CL_SUCCESS) {
    char log[4096] = {0};
    size_t lsz = 0;
    D.api.GetProgramBuildInfo(D.prog, D.dev, CL_PROGRAM_BUILD_LOG,
                              sizeof(log) - 1, log, &lsz);
    D.refuse = std::string("kernel build failed: ") + log;
    err = D.refuse;
    D.releaseProgram();
    return false;
  }
  struct { cl_kernel* fn; const char* name; } kn[] = {
      {&D.k.syn_decay, "ndl_syn_decay"},       {&D.k.prop_dense, "ndl_prop_dense"},
      {&D.k.prop_csr, "ndl_prop_csr"},         {&D.k.prop_o2o, "ndl_prop_o2o"},
      {&D.k.inject_range, "ndl_inject_range"}, {&D.k.osc_add, "ndl_osc_add"},
      {&D.k.stream_poisson, "ndl_stream_poisson"},
      {&D.k.lif, "ndl_lif"},                   {&D.k.stdp_dense, "ndl_stdp_dense"},
      {&D.k.stdp_csr, "ndl_stdp_csr"},         {&D.k.stdp_o2o, "ndl_stdp_o2o"},
      {&D.k.trace, "ndl_trace"},
      {&D.k.norm_rowsum_csr, "ndl_norm_rowsum_csr"},
      {&D.k.norm_scale_csr, "ndl_norm_scale_csr"},
      {&D.k.norm_rowsum_dense, "ndl_norm_rowsum_dense"},
      {&D.k.norm_scale_dense, "ndl_norm_scale_dense"},
      {&D.k.norm_rowsum_o2o, "ndl_norm_rowsum_o2o"},
      {&D.k.norm_scale_o2o, "ndl_norm_scale_o2o"},
  };
  for (auto& it : kn) {
    cl_int kr = 0;
    *it.fn = D.api.CreateKernel(D.prog, it.name, &kr);
    if (!*it.fn || kr != CL_SUCCESS) {
      D.refuse = std::string("clCreateKernel(") + it.name + ") failed";
      err = D.refuse;
      D.releaseProgram();
      return false;
    }
  }
  D.ready = true;
  err.clear();
  return true;
}

// ---------------------------------------------------------------------------
// Device-resident simulation state (per attach)
// ---------------------------------------------------------------------------
struct GroupBuf {
  cl_mem v = nullptr, isyn = nullptr, hold = nullptr, trace = nullptr,
         spk_cur = nullptr, spk_prev = nullptr, nrng = nullptr;
};
struct ConnBuf {
  cl_mem w = nullptr;  // dense (dst-major) or one-to-one
  cl_mem rowptr = nullptr, cols = nullptr, vals = nullptr, rowsum = nullptr;
};
struct StreamBuf {
  cl_mem rng = nullptr, latest = nullptr;
  int64_t n = 0;  // min(target size, stream size)
};

struct OclState {
  bool attached = false;
  int64_t topo = -1;
  std::vector<GroupBuf> gb;
  std::vector<ConnBuf> cb;
  std::vector<StreamBuf> sb;
  std::vector<cl_mem> all;  // every allocation, freed together on detach

  void freeAll() {
    OclApi& A = g_dev.api;
    if (A.ReleaseMemObject)
      for (cl_mem m : all)
        if (m) A.ReleaseMemObject(m);
    all.clear();
    gb.clear();
    cb.clear();
    sb.clear();
    attached = false;
    topo = -1;
  }
} g_st;

bool mkBuf(size_t bytes, cl_uint flags, void* host, cl_mem& out) {
  cl_int r = 0;
  out = g_dev.api.CreateBuffer(g_dev.ctx, flags, bytes, host, &r);
  if (!out || r != CL_SUCCESS) return false;
  g_st.all.push_back(out);
  return true;
}

bool upload(cl_mem m, const void* host, size_t bytes) {
  return g_dev.api.EnqueueWriteBuffer(g_dev.q, m, CL_TRUE, 0, bytes, host, 0,
                                     nullptr, nullptr) == CL_SUCCESS;
}
bool download(cl_mem m, void* host, size_t bytes) {
  return g_dev.api.EnqueueReadBuffer(g_dev.q, m, CL_TRUE, 0, bytes, host, 0,
                                     nullptr, nullptr) == CL_SUCCESS;
}

// Replace a device buffer whose host mirror has OUTGROWN the allocation
// (Ф9.4b grow_synapses: CSR cols/vals grow). Releases the old buffer,
// allocates a fresh one of the new size and uploads. Every other buffer
// keeps its state — no re-attach, no live-state loss.
bool replaceBuf(cl_mem& m, const void* host, size_t bytes) {
  cl_mem nm = nullptr;
  if (!mkBuf(bytes, CL_MEM_READ_WRITE, nullptr, nm)) return false;
  if (!upload(nm, host, bytes)) return false;
  if (m) {
    g_dev.api.ReleaseMemObject(m);
    for (auto& x : g_st.all)
      if (x == m) { x = nm; break; }
  }
  m = nm;
  return true;
}
bool launch(cl_kernel k, size_t global) {
  if (global == 0) return true;
  const size_t local = 64;
  const size_t g = ((global + local - 1) / local) * local;
  return g_dev.api.EnqueueNDRangeKernel(g_dev.q, k, 1, nullptr, &g, &local, 0,
                                        nullptr, nullptr) == CL_SUCCESS;
}

size_t groupIndex(const SimState& st, const NdlGroup* g) {
  for (size_t gi = 0; gi < st.groups.size(); ++gi)
    if (st.groups[gi] == g) return gi;
  return (size_t)-1;
}

// Build (or rebuild) every device buffer from the current SimState.
bool attachState(std::string& err) {
  SimState& st = ndl_rt_sim();
  g_st.freeAll();

  auto fail = [&](const char* what) {
    err = what;
    g_st.freeAll();
    return false;
  };

  // --- groups ---------------------------------------------------------------
  g_st.gb.resize(st.groups.size());
  for (size_t gi = 0; gi < st.groups.size(); ++gi) {
    NdlGroup* g = st.groups[gi];
    const size_t n = (size_t)g->size;
    GroupBuf& b = g_st.gb[gi];
    bool ok = true;
    ok &= mkBuf(n * 8, CL_MEM_READ_WRITE, nullptr, b.v);
    ok &= mkBuf(n * 8, CL_MEM_READ_WRITE, nullptr, b.isyn);
    ok &= mkBuf(n * 8, CL_MEM_READ_WRITE, nullptr, b.hold);
    ok &= mkBuf(n * 8, CL_MEM_READ_WRITE, nullptr, b.trace);
    ok &= mkBuf(n, CL_MEM_READ_WRITE, nullptr, b.spk_cur);
    ok &= mkBuf(n, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                (void*)g->prev_spikes.data(), b.spk_prev);
    ok &= mkBuf(n * 8, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                (void*)g->noise_rng.data(), b.nrng);
    if (!ok) return fail("clCreateBuffer failed (group)");
    std::vector<double> zd(n, 0.0);
    std::vector<uint8_t> zb(n, 0);
    if (!upload(b.v, g->v.data(), n * 8) || !upload(b.isyn, zd.data(), n * 8) ||
        !upload(b.hold, zd.data(), n * 8) ||
        !upload(b.trace, g->trace.data(), n * 8) ||
        !upload(b.spk_cur, zb.data(), n))
      return fail("initial upload failed (group)");
  }

  // --- connections ------------------------------------------------------------
  g_st.cb.resize(st.conns.size());
  for (size_t ci = 0; ci < st.conns.size(); ++ci) {
    const Connection* c = st.conns[ci];
    ConnBuf& b = g_st.cb[ci];
    bool ok = true;
    if (c->kind == CONN_DENSE || c->kind == CONN_ONE2ONE) {
      const size_t bytes = c->w.size() * 8;
      ok &= mkBuf(bytes, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                  (void*)c->w.data(), b.w);
    } else {  // CONN_SPARSE
      ok &= mkBuf(c->rowptr.size() * 8,
                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                  (void*)c->rowptr.data(), b.rowptr);
      ok &= mkBuf(c->cols.size() * 4, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                  (void*)c->cols.data(), b.cols);
      ok &= mkBuf(c->vals.size() * 8, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                  (void*)c->vals.data(), b.vals);
      ok &= mkBuf(((size_t)c->dst->size) * 8, CL_MEM_READ_WRITE, nullptr, b.rowsum);
    }
    if (!ok) return fail("clCreateBuffer failed (connection)");
  }

  // --- streams (Poisson encoders) ---------------------------------------------
  g_st.sb.resize(st.streams.size());
  for (size_t si = 0; si < st.streams.size(); ++si) {
    NdlStream* s = st.streams[si];
    StreamBuf& b = g_st.sb[si];
    b.n = s->target ? std::min<int64_t>(s->target->size, s->size) : 0;
    bool ok = true;
    if (b.n > 0) {
      // per-neuron pcg32 seeds from the host RNG (seed'ом с хоста)
      std::vector<uint64_t> seeds((size_t)b.n);
      for (auto& sd : seeds) sd = ndl_rt_rng_next_u64();
      ok &= mkBuf(seeds.size() * 8, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                  (void*)seeds.data(), b.rng);
      ok &= mkBuf((size_t)s->size * 8, CL_MEM_READ_WRITE, nullptr, b.latest);
    }
    if (!ok) return fail("clCreateBuffer failed (stream)");
  }

  g_st.topo = st.topology_version;
  g_st.attached = true;
  err.clear();
  return true;
}

// Per-tick device execution. Returns false + err on device failure (the caller
// syncs down and migrates to the CPU).
bool oclTick(double dt_ms, bool apply_injections, int64_t& inj_idx, int64_t n_inj,
             const std::vector<double>& decay, double synDecay, std::string& err) {
  SimState& st = ndl_rt_sim();
  OclApi& A = g_dev.api;
  Kernels& K = g_dev.k;
  const size_t nG = st.groups.size();

  auto setp = [&](cl_kernel k, cl_uint idx, const void* v, size_t sz) {
    return A.SetKernelArg(k, idx, sz, v) == CL_SUCCESS;
  };
  auto die = [&](const char* what) {
    err = what;
    return false;
  };

  // Structural commands queued since the last tick (plasticity → host flag;
  // prune/normalize → routed back into this backend by the dispatcher).
  ndl_rt_drain_commands();

  // (1) synaptic current decay
  {
    const double dec = synDecay;
    for (size_t gi = 0; gi < nG; ++gi) {
      const unsigned n = (unsigned)st.groups[gi]->size;
      cl_mem m = g_st.gb[gi].isyn;
      if (!setp(K.syn_decay, 0, &m, sizeof(m)) || !setp(K.syn_decay, 1, &dec, 8) ||
          !setp(K.syn_decay, 2, &n, 4) || !launch(K.syn_decay, n))
        return die("syn_decay launch failed");
    }
  }

  // (2) propagation per connection (over previous-tick spikes)
  for (size_t ci = 0; ci < st.conns.size(); ++ci) {
    const Connection* c = st.conns[ci];
    const size_t pre_i = groupIndex(st, c->src);
    const size_t post_i = groupIndex(st, c->dst);
    if (pre_i == (size_t)-1 || post_i == (size_t)-1)
      return die("connection references unknown group");
    const GroupBuf& sb = g_st.gb[pre_i];
    GroupBuf& db = g_st.gb[post_i];
    if (c->kind == CONN_DENSE) {
      const unsigned nsrc = (unsigned)c->src->size;
      const unsigned ndst = (unsigned)c->dst->size;
      cl_mem w = g_st.cb[ci].w, pre = sb.spk_prev, isyn = db.isyn;
      if (!setp(K.prop_dense, 0, &w, sizeof(w)) ||
          !setp(K.prop_dense, 1, &pre, sizeof(pre)) ||
          !setp(K.prop_dense, 2, &isyn, sizeof(isyn)) ||
          !setp(K.prop_dense, 3, &nsrc, 4) || !setp(K.prop_dense, 4, &ndst, 4) ||
          !launch(K.prop_dense, ndst))
        return die("prop_dense launch failed");
    } else if (c->kind == CONN_SPARSE) {
      const unsigned ndst = (unsigned)c->dst->size;
      cl_mem rp = g_st.cb[ci].rowptr, cl = g_st.cb[ci].cols,
             vl = g_st.cb[ci].vals, pre = sb.spk_prev, isyn = db.isyn;
      if (!setp(K.prop_csr, 0, &rp, sizeof(rp)) ||
          !setp(K.prop_csr, 1, &cl, sizeof(cl)) ||
          !setp(K.prop_csr, 2, &vl, sizeof(vl)) ||
          !setp(K.prop_csr, 3, &pre, sizeof(pre)) ||
          !setp(K.prop_csr, 4, &isyn, sizeof(isyn)) ||
          !setp(K.prop_csr, 5, &ndst, 4) || !launch(K.prop_csr, ndst))
        return die("prop_csr launch failed");
    } else {
      const unsigned n = (unsigned)c->dst->size;
      cl_mem w = g_st.cb[ci].w, pre = sb.spk_prev, isyn = db.isyn;
      if (!setp(K.prop_o2o, 0, &w, sizeof(w)) ||
          !setp(K.prop_o2o, 1, &pre, sizeof(pre)) ||
          !setp(K.prop_o2o, 2, &isyn, sizeof(isyn)) ||
          !setp(K.prop_o2o, 3, &n, 4) || !launch(K.prop_o2o, n))
        return die("prop_o2o launch failed");
    }
  }

  // (3) injections: sustained step currents, ASSIGN semantics (a later one
  // wins), contiguous ranges — one small launch per injection.
  if (apply_injections) {
    const double t0 = st.sim_time_ms;
    const double t1 = t0 + dt_ms;
    while (inj_idx < n_inj && st.injections[(size_t)inj_idx].t < t1) {
      const Injection& inj = st.injections[(size_t)inj_idx];
      if (inj.t >= t0 && inj.g) {
        const size_t gi = groupIndex(st, inj.g);
        if (gi != (size_t)-1) {
          int64_t lo = std::max<int64_t>(inj.lo, 0);
          int64_t hi = std::min<int64_t>(inj.hi, inj.g->size);
          if (hi > lo) {
            const unsigned base = (unsigned)lo;
            const unsigned cnt = (unsigned)(hi - lo);
            const double cur = inj.cur;
            cl_mem h = g_st.gb[gi].hold;
            if (!setp(K.inject_range, 0, &h, sizeof(h)) ||
                !setp(K.inject_range, 1, &base, 4) ||
                !setp(K.inject_range, 2, &cnt, 4) ||
                !setp(K.inject_range, 3, &cur, 8) ||
                !launch(K.inject_range, cnt))
              return die("inject_range launch failed");
          }
        }
      }
      ++inj_idx;
    }
  }

  // (3b) oscillators: I_osc computed on the host, one add-kernel per target
  for (const NdlOscillator* o : st.oscillators) {
    if (!o->target || o->amplitude == 0.0) continue;
    const size_t gi = groupIndex(st, o->target);
    if (gi == (size_t)-1) continue;
    const double w = 2.0 * 3.14159265358979323846 * o->freq_hz;
    const double iosc = o->amplitude * std::sin(w * (st.sim_time_ms / 1000.0) + o->phase);
    const unsigned n = (unsigned)o->target->size;
    cl_mem m = g_st.gb[gi].isyn;
    if (!setp(K.osc_add, 0, &m, sizeof(m)) || !setp(K.osc_add, 1, &iosc, 8) ||
        !setp(K.osc_add, 2, &n, 4) || !launch(K.osc_add, n))
      return die("osc_add launch failed");
  }

  // (3c) external streams: drain the SPSC ring on the host, upload latest,
  // Poisson-encode on the device (per-neuron pcg32 state lives on device)
  for (size_t si = 0; si < st.streams.size(); ++si) {
    NdlStream* s = st.streams[si];
    StreamBuf& b = g_st.sb[si];
    if (!s->target || b.n <= 0 || s->max_freq <= 0.0 || s->encoding != 0) continue;
    {
      std::lock_guard<std::mutex> lk(s->mx);
      int64_t r = s->rpos.load(std::memory_order_acquire);
      const int64_t w = s->wpos.load(std::memory_order_acquire);
      while (r != w) {
        const int64_t slot = (r % NdlStream::kRingCap) * s->size;
        std::memcpy(s->latest.data(), &s->ring[(size_t)slot],
                    sizeof(double) * (size_t)s->size);
        ++r;
      }
      s->rpos.store(r, std::memory_order_release);
    }
    if (!upload(b.latest, s->latest.data(), (size_t)s->size * 8))
      return die("stream latest upload failed");
    const double p_scale = s->max_freq * dt_ms / 1000.0;
    const double kick = s->kick;
    const size_t gi = groupIndex(st, s->target);
    cl_mem rng = b.rng, latest = b.latest, isyn = g_st.gb[gi].isyn;
    const unsigned sn = (unsigned)b.n;
    if (!setp(K.stream_poisson, 0, &rng, sizeof(rng)) ||
        !setp(K.stream_poisson, 1, &latest, sizeof(latest)) ||
        !setp(K.stream_poisson, 2, &p_scale, 8) ||
        !setp(K.stream_poisson, 3, &kick, 8) ||
        !setp(K.stream_poisson, 4, &isyn, sizeof(isyn)) ||
        !setp(K.stream_poisson, 5, &sn, 4) ||
        !launch(K.stream_poisson, (size_t)b.n))
      return die("stream_poisson launch failed");
  }

  // (4) LIF step per group
  {
    // membrane noise sigma (env), same value the CPU path uses
    static const double sigma = []() {
      const char* e = std::getenv("NDL_LIF_NOISE");
      if (!e) return 0.0;
      const double v = std::atof(e);
      return (std::isfinite(v) && v > 0.0) ? v : 0.0;
    }();
    for (size_t gi = 0; gi < nG; ++gi) {
      NdlGroup* g = st.groups[gi];
      const unsigned n = (unsigned)g->size;
      cl_mem v = g_st.gb[gi].v, isyn = g_st.gb[gi].isyn, hold = g_st.gb[gi].hold,
             spk = g_st.gb[gi].spk_cur, nrng = g_st.gb[gi].nrng;
      double dt = dt_ms, tau = g->tau, thr = g->threshold, rest = g->rest,
             reset = g->reset, sig = sigma;
      if (!setp(K.lif, 0, &v, sizeof(v)) || !setp(K.lif, 1, &isyn, sizeof(isyn)) ||
          !setp(K.lif, 2, &hold, sizeof(hold)) ||
          !setp(K.lif, 3, &spk, sizeof(spk)) || !setp(K.lif, 4, &dt, 8) ||
          !setp(K.lif, 5, &tau, 8) || !setp(K.lif, 6, &thr, 8) ||
          !setp(K.lif, 7, &rest, 8) || !setp(K.lif, 8, &reset, 8) ||
          !setp(K.lif, 9, &sig, 8) || !setp(K.lif, 10, &nrng, sizeof(nrng)) ||
          !setp(K.lif, 11, &n, 4) || !launch(K.lif, n))
        return die("lif launch failed");
    }
  }

  // (6) STDP × M(t) on plastic connections with a config (device weights)
  for (size_t ci = 0; ci < st.conns.size(); ++ci) {
    const Connection* c = st.conns[ci];
    const StdpCfg* cfg = ndl_rt_find_stdp(st, c->src, c->dst);
    if (!c->plastic || !cfg) continue;
    const size_t pre_i = groupIndex(st, c->src);
    const size_t post_i = groupIndex(st, c->dst);
    const GroupBuf& sb = g_st.gb[pre_i];
    GroupBuf& db = g_st.gb[post_i];
    const double mod = ndl_rt_stdp_mod_value(cfg);
    const double lrp = cfg->lr_pot, lrd = cfg->lr_dep;
    if (c->kind == CONN_DENSE) {
      const unsigned nsrc = (unsigned)c->src->size;
      const unsigned ndst = (unsigned)c->dst->size;
      cl_mem w = g_st.cb[ci].w, ptr_ = sb.trace, psp = db.spk_cur,
             ptr2 = db.trace, pp = sb.spk_prev;
      if (!setp(K.stdp_dense, 0, &w, sizeof(w)) ||
          !setp(K.stdp_dense, 1, &ptr_, sizeof(ptr_)) ||
          !setp(K.stdp_dense, 2, &psp, sizeof(psp)) ||
          !setp(K.stdp_dense, 3, &ptr2, sizeof(ptr2)) ||
          !setp(K.stdp_dense, 4, &pp, sizeof(pp)) ||
          !setp(K.stdp_dense, 5, &lrp, 8) || !setp(K.stdp_dense, 6, &lrd, 8) ||
          !setp(K.stdp_dense, 7, &mod, 8) || !setp(K.stdp_dense, 8, &nsrc, 4) ||
          !setp(K.stdp_dense, 9, &ndst, 4) || !launch(K.stdp_dense, ndst))
        return die("stdp_dense launch failed");
    } else if (c->kind == CONN_SPARSE) {
      const unsigned ndst = (unsigned)c->dst->size;
      cl_mem rp = g_st.cb[ci].rowptr, cl = g_st.cb[ci].cols,
             vl = g_st.cb[ci].vals, ptr_ = sb.trace, psp = db.spk_cur,
             ptr2 = db.trace, pp = sb.spk_prev;
      if (!setp(K.stdp_csr, 0, &rp, sizeof(rp)) ||
          !setp(K.stdp_csr, 1, &cl, sizeof(cl)) ||
          !setp(K.stdp_csr, 2, &vl, sizeof(vl)) ||
          !setp(K.stdp_csr, 3, &ptr_, sizeof(ptr_)) ||
          !setp(K.stdp_csr, 4, &psp, sizeof(psp)) ||
          !setp(K.stdp_csr, 5, &ptr2, sizeof(ptr2)) ||
          !setp(K.stdp_csr, 6, &pp, sizeof(pp)) ||
          !setp(K.stdp_csr, 7, &lrp, 8) || !setp(K.stdp_csr, 8, &lrd, 8) ||
          !setp(K.stdp_csr, 9, &mod, 8) || !setp(K.stdp_csr, 10, &ndst, 4) ||
          !launch(K.stdp_csr, ndst))
        return die("stdp_csr launch failed");
    } else {
      const unsigned n = (unsigned)c->dst->size;
      cl_mem w = g_st.cb[ci].w, ptr_ = sb.trace, psp = db.spk_cur,
             ptr2 = db.trace, pp = sb.spk_prev;
      if (!setp(K.stdp_o2o, 0, &w, sizeof(w)) ||
          !setp(K.stdp_o2o, 1, &ptr_, sizeof(ptr_)) ||
          !setp(K.stdp_o2o, 2, &psp, sizeof(psp)) ||
          !setp(K.stdp_o2o, 3, &ptr2, sizeof(ptr2)) ||
          !setp(K.stdp_o2o, 4, &pp, sizeof(pp)) ||
          !setp(K.stdp_o2o, 5, &lrp, 8) || !setp(K.stdp_o2o, 6, &lrd, 8) ||
          !setp(K.stdp_o2o, 7, &mod, 8) || !setp(K.stdp_o2o, 8, &n, 4) ||
          !launch(K.stdp_o2o, n))
        return die("stdp_o2o launch failed");
    }
  }

  // (7) trace decay per group
  for (size_t gi = 0; gi < nG; ++gi) {
    const unsigned n = (unsigned)st.groups[gi]->size;
    const double dec = decay[gi];
    cl_mem tr = g_st.gb[gi].trace, spk = g_st.gb[gi].spk_cur;
    if (!setp(K.trace, 0, &tr, sizeof(tr)) || !setp(K.trace, 1, &spk, sizeof(spk)) ||
        !setp(K.trace, 2, &dec, 8) || !setp(K.trace, 3, &n, 4) ||
        !launch(K.trace, n))
      return die("trace launch failed");
  }

  // (8) spike download → logs + handlers + host mirror; (9) prev = cur (swap)
  std::vector<uint8_t> spikes;
  for (size_t gi = 0; gi < nG; ++gi) {
    NdlGroup* g = st.groups[gi];
    const size_t n = (size_t)g->size;
    spikes.resize(n);
    if (!download(g_st.gb[gi].spk_cur, spikes.data(), n))
      return die("spike download failed");
    for (size_t i = 0; i < n; ++i)
      if (spikes[i]) ndl_rt_log_spike(g, st.sim_time_ms, (int64_t)i);
    if (!g->handlers.empty()) {
      const std::vector<NdlGroup::Handler> handlers = g->handlers;
      for (const NdlGroup::Handler& h : handlers)
        for (size_t i = 0; i < n; ++i)
          if (spikes[i]) h.fn((int64_t)i, st.sim_time_ms, h.user);
    }
    g->prev_spikes = spikes;       // host mirror stays in sync
    g->spikes = spikes;            // current-tick mirror (getters, fallback)
    std::swap(g_st.gb[gi].spk_cur, g_st.gb[gi].spk_prev);  // device-side swap
  }

  // (10) global time base (host-authoritative)
  st.sim_time_ms.store(st.sim_time_ms.load(std::memory_order_relaxed) + dt_ms,
                       std::memory_order_relaxed);
  return true;
}

// Download the whole device state into the host mirrors.
void oclSyncDown() {
  SimState& st = ndl_rt_sim();
  if (!g_st.attached) return;
  for (size_t gi = 0; gi < st.groups.size() && gi < g_st.gb.size(); ++gi) {
    NdlGroup* g = st.groups[gi];
    const size_t n = (size_t)g->size;
    download(g_st.gb[gi].v, g->v.data(), n * 8);
    download(g_st.gb[gi].trace, g->trace.data(), n * 8);
    download(g_st.gb[gi].nrng, g->noise_rng.data(), n * 8);
  }
  for (size_t ci = 0; ci < st.conns.size() && ci < g_st.cb.size(); ++ci) {
    Connection* c = st.conns[ci];
    if (c->kind == CONN_DENSE || c->kind == CONN_ONE2ONE)
      download(g_st.cb[ci].w, c->w.data(), c->w.size() * 8);
    else if (c->kind == CONN_SPARSE)
      download(g_st.cb[ci].vals, c->vals.data(), c->vals.size() * 8);
  }
}

// Download one group's membrane potentials (tensor getter).
void oclSyncGroupV(NdlGroup* g) {
  SimState& st = ndl_rt_sim();
  if (!g_st.attached) return;
  const size_t gi = groupIndex(st, g);
  if (gi == (size_t)-1 || gi >= g_st.gb.size()) return;
  download(g_st.gb[gi].v, g->v.data(), (size_t)g->size * 8);
}

// Prune while attached: download → host compaction (exact CPU semantics,
// including the CSR row rebuild) → re-upload rowptr/cols/vals. Prune is a
// rare host command; correctness beats cleverness.
void oclPrune(NdlGroup* src, NdlGroup* dst, double threshold) {
  SimState& st = ndl_rt_sim();
  if (!g_st.attached) return;
  for (size_t ci = 0; ci < st.conns.size(); ++ci) {
    Connection* c = st.conns[ci];
    if (c->src != src || c->dst != dst) continue;
    if (c->kind == CONN_DENSE || c->kind == CONN_ONE2ONE) {
      download(g_st.cb[ci].w, c->w.data(), c->w.size() * 8);
      ndl_rt_apply_prune_direct(src, dst, threshold);  // host compaction
      upload(g_st.cb[ci].w, c->w.data(), c->w.size() * 8);
    } else if (c->kind == CONN_SPARSE) {
      download(g_st.cb[ci].rowptr, c->rowptr.data(), c->rowptr.size() * 8);
      download(g_st.cb[ci].cols, c->cols.data(), c->cols.size() * 4);
      download(g_st.cb[ci].vals, c->vals.data(), c->vals.size() * 8);
      ndl_rt_apply_prune_direct(src, dst, threshold);  // host compaction
      upload(g_st.cb[ci].rowptr, c->rowptr.data(), c->rowptr.size() * 8);
      upload(g_st.cb[ci].cols, c->cols.data(), c->cols.size() * 4);
      upload(g_st.cb[ci].vals, c->vals.data(), c->vals.size() * 8);
    }
    break;
  }
}

// Grow while attached: download the CSR into the host mirrors, apply the
// host merge, then replace the three CSR buffers in place (rowptr keeps its
// size, cols/vals grow). bumpTopo stays false — a topology bump would force
// a full re-attach at the next tick and discard live v/trace/weights.
void oclGrow(NdlGroup* src, NdlGroup* dst, const int64_t* pre, int64_t n_pre,
             const int64_t* post, int64_t n_post, double weight) {
  SimState& st = ndl_rt_sim();
  if (!g_st.attached) {
    ndl_rt_apply_grow_direct(src, dst, pre, n_pre, post, n_post, weight, true);
    return;
  }
  for (size_t ci = 0; ci < st.conns.size(); ++ci) {
    Connection* c = st.conns[ci];
    if (c->src != src || c->dst != dst) continue;
    if (c->kind != CONN_SPARSE) break;
    download(g_st.cb[ci].rowptr, c->rowptr.data(), c->rowptr.size() * 8);
    download(g_st.cb[ci].cols, c->cols.data(), c->cols.size() * 4);
    download(g_st.cb[ci].vals, c->vals.data(), c->vals.size() * 8);
    const int64_t added =
        ndl_rt_apply_grow_direct(src, dst, pre, n_pre, post, n_post, weight,
                                 false);
    if (added > 0) {
      replaceBuf(g_st.cb[ci].rowptr, c->rowptr.data(), c->rowptr.size() * 8);
      replaceBuf(g_st.cb[ci].cols, c->cols.data(), c->cols.size() * 4);
      replaceBuf(g_st.cb[ci].vals, c->vals.data(), c->vals.size() * 8);
    }
    break;
  }
}

// Homeostatic synaptic scaling on the DEVICE (rowsum + scale kernels).
void oclNormalize(NdlGroup* src, NdlGroup* dst, double target_sum) {
  SimState& st = ndl_rt_sim();
  if (!g_st.attached) return;
  OclApi& A = g_dev.api;
  Kernels& K = g_dev.k;
  for (size_t k = 0; k < st.conns.size(); ++k) {
    Connection* c = st.conns[k];
    if (c->src != src || c->dst != dst) continue;
    auto setp = [&](cl_kernel kk, cl_uint idx, const void* v, size_t sz) {
      return A.SetKernelArg(kk, idx, sz, v) == CL_SUCCESS;
    };
    if (c->kind == CONN_DENSE) {
      const unsigned nsrc = (unsigned)c->src->size;
      const unsigned ndst = (unsigned)c->dst->size;
      cl_mem w = g_st.cb[k].w, rs = g_st.cb[k].rowsum;
      if (!setp(K.norm_rowsum_dense, 0, &w, sizeof(w)) ||
          !setp(K.norm_rowsum_dense, 1, &nsrc, 4) ||
          !setp(K.norm_rowsum_dense, 2, &rs, sizeof(rs)) ||
          !setp(K.norm_rowsum_dense, 3, &ndst, 4) ||
          !launch(K.norm_rowsum_dense, ndst))
        return;
      if (!setp(K.norm_scale_dense, 0, &w, sizeof(w)) ||
          !setp(K.norm_scale_dense, 1, &nsrc, 4) ||
          !setp(K.norm_scale_dense, 2, &rs, sizeof(rs)) ||
          !setp(K.norm_scale_dense, 3, &target_sum, 8) ||
          !setp(K.norm_scale_dense, 4, &ndst, 4) ||
          !launch(K.norm_scale_dense, ndst))
        return;
    } else if (c->kind == CONN_ONE2ONE) {
      const unsigned n = (unsigned)c->dst->size;
      cl_mem w = g_st.cb[k].w, rs = g_st.cb[k].rowsum;
      if (!setp(K.norm_rowsum_o2o, 0, &w, sizeof(w)) ||
          !setp(K.norm_rowsum_o2o, 1, &rs, sizeof(rs)) ||
          !setp(K.norm_rowsum_o2o, 2, &n, 4) || !launch(K.norm_rowsum_o2o, n))
        return;
      if (!setp(K.norm_scale_o2o, 0, &w, sizeof(w)) ||
          !setp(K.norm_scale_o2o, 1, &rs, sizeof(rs)) ||
          !setp(K.norm_scale_o2o, 2, &target_sum, 8) ||
          !setp(K.norm_scale_o2o, 3, &n, 4) || !launch(K.norm_scale_o2o, n))
        return;
    } else if (c->kind == CONN_SPARSE) {
      const unsigned ndst = (unsigned)c->dst->size;
      cl_mem rp = g_st.cb[k].rowptr, vl = g_st.cb[k].vals, rs = g_st.cb[k].rowsum;
      if (!setp(K.norm_rowsum_csr, 0, &rp, sizeof(rp)) ||
          !setp(K.norm_rowsum_csr, 1, &vl, sizeof(vl)) ||
          !setp(K.norm_rowsum_csr, 2, &rs, sizeof(rs)) ||
          !setp(K.norm_rowsum_csr, 3, &ndst, 4) ||
          !launch(K.norm_rowsum_csr, ndst))
        return;
      if (!setp(K.norm_scale_csr, 0, &rp, sizeof(rp)) ||
          !setp(K.norm_scale_csr, 1, &vl, sizeof(vl)) ||
          !setp(K.norm_scale_csr, 2, &rs, sizeof(rs)) ||
          !setp(K.norm_scale_csr, 3, &target_sum, 8) ||
          !setp(K.norm_scale_csr, 4, &ndst, 4) ||
          !launch(K.norm_scale_csr, ndst))
        return;
    }
    break;
  }
}

}  // namespace

// --- public (consumed by ndl_rt_backend.cpp) ----------------------------------

int ndl_rt_ocl_execute(int64_t ticks, double dt_ms, bool continuous,
                       bool apply_injections, const std::vector<double>& decay,
                       double synDecay, std::string& err) {
  if (!devInit(err)) return 0;
  SimState& st = ndl_rt_sim();
  // topology changed (or first use) → (re)build device buffers
  if (!g_st.attached || g_st.topo != st.topology_version) {
    if (!attachState(err)) return 0;
  }
  int64_t inj_idx = 0;
  const int64_t n_inj = apply_injections ? (int64_t)st.injections.size() : 0;
  for (int64_t t = 0; t < ticks; ++t) {
    if (!oclTick(dt_ms, apply_injections, inj_idx, n_inj, decay, synDecay, err)) {
      // device lost mid-run: best-effort sync, release, migrate
      oclSyncDown();
      g_st.freeAll();
      return -1;
    }
  }
  if (!continuous) {
    // Discrete run: hand the state back to the host (v, trace, weights,
    // noise RNG), like the CUDA path does.
    oclSyncDown();
    g_st.freeAll();
  }
  return 1;
}

bool ndl_rt_ocl_attached() { return g_st.attached; }

const char* ndl_rt_ocl_device_name(std::string* err) {
  std::string e;
  if (!devInit(e)) {
    if (err) *err = e;
    return nullptr;
  }
  if (err) err->clear();
  return g_dev.name.c_str();
}
void ndl_rt_ocl_prune(NdlGroup* s, NdlGroup* d, double t) { oclPrune(s, d, t); }
void ndl_rt_ocl_normalize(NdlGroup* s, NdlGroup* d, double t) { oclNormalize(s, d, t); }
void ndl_rt_ocl_grow(NdlGroup* s, NdlGroup* d, const int64_t* pre, int64_t n_pre,
                     const int64_t* post, int64_t n_post, double w) {
  oclGrow(s, d, pre, n_pre, post, n_post, w);
}
void ndl_rt_ocl_sync_down() { oclSyncDown(); }
void ndl_rt_ocl_sync_group_v(NdlGroup* g) { oclSyncGroupV(g); }
void ndl_rt_ocl_invalidate() { g_st.freeAll(); }
