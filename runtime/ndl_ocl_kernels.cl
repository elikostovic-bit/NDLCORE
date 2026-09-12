// NDL v2.1 — built-in simulation kernels for the OpenCL backend.
//
// This file is used TWICE (single source of truth):
//   1. as OpenCL C source: ndl_rt_ocl.cpp uploads it via clCreateProgramWithSource;
//   2. as plain C: the mock-ICD test harness (mock_cl.cpp) includes it with
//      NDL_HOST_KERNELS defined, so the host-side test executes EXACTLY the
//      same math the device will.
//
// The bodies are deliberately plain C99 (no vector types, no local memory,
// no barriers) so both translations are semantically identical.
//
// Tick order implemented by the backend (mirrors ndl_rt_tick_cpu):
//   syn_decay → propagate (dense/csr/o2o) → inject_range → osc_add →
//   stream_poisson → lif → stdp (dense/csr/o2o, × M(t)) → trace →
//   [spike download + handlers on host] → prev = cur (pointer swap)
#ifndef NDL_HOST_KERNELS
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef NDL_HOST_KERNELS
// Host (mock-ICD) build: neutralize OpenCL address-space/Kernel qualifiers
// and provide the device builtin types. get_global_id() is supplied by the
// mock driver (it sets the gid before each work-item call).
#ifndef __kernel
#define __kernel
#endif
#ifndef __global
#define __global
#endif
#ifndef __constant
#define __constant
#endif
#include <stdint.h>
#include <math.h>
typedef uint8_t uchar;
typedef uint64_t ulong;
typedef unsigned int uint;
#endif

// ---------------------------------------------------------------------------
// RNG: pcg32 (per-neuron state, seeded by the host) + Box-Muller gaussian.
// The host uses the SAME algorithm for LIF noise, so with equal seeds the
// CPU and GPU noise sequences are identical (not just statistically equal).
// ---------------------------------------------------------------------------
uint ndl_pcg32(__global ulong* st) {
  const ulong s = *st;
  *st = s * 6364136223846793005ULL + 1442695040888963407ULL;
  const uint xorshifted = (uint)(((s >> 18u) ^ s) >> 27u);
  const uint rot = (uint)(s >> 59u);
  return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
}

double ndl_uniform01(__global ulong* st) {
  return (double)ndl_pcg32(st) * (1.0 / 4294967296.0);
}

double ndl_gauss(__global ulong* st) {
  double u1 = ndl_uniform01(st);
  const double u2 = ndl_uniform01(st);
  if (u1 < 1e-300) u1 = 1e-300;
  return sqrt(-2.0 * log(u1)) * cos(6.283185307179586476925286766559 * u2);
}

// ---------------------------------------------------------------------------
// (1) synaptic current decay: i_syn *= exp(-dt/tau_syn)
// ---------------------------------------------------------------------------
__kernel void ndl_syn_decay(__global double* isyn, const double dec, const uint n) {
  const uint i = (uint)get_global_id(0);
  if (i < n) isyn[i] *= dec;
}

// ---------------------------------------------------------------------------
// (2) propagation (accumulate over previous-tick spikes of the source group)
// ---------------------------------------------------------------------------
__kernel void ndl_prop_dense(__global const double* w, __global const uchar* pre,
                             __global double* isyn, const uint nsrc, const uint n) {
  const uint j = (uint)get_global_id(0);
  if (j >= n) return;
  // one work-item per postsynaptic neuron; w is dst-major: w[j * nsrc + i]
  double acc = 0.0;
  for (uint i = 0; i < nsrc; ++i) acc += w[(size_t)j * nsrc + i] * (double)pre[i];
  isyn[j] += acc;
}

__kernel void ndl_prop_csr(__global const ulong* rowptr, __global const uint* cols,
                           __global const double* vals, __global const uchar* pre,
                           __global double* isyn, const uint n) {
  const uint j = (uint)get_global_id(0);
  if (j >= n) return;
  double acc = 0.0;
  for (ulong k = rowptr[j]; k < rowptr[j + 1]; ++k)
    acc += vals[k] * (double)pre[cols[k]];
  isyn[j] += acc;
}

__kernel void ndl_prop_o2o(__global const double* w, __global const uchar* pre,
                           __global double* isyn, const uint n) {
  const uint i = (uint)get_global_id(0);
  if (i >= n) return;
  isyn[i] += w[i] * (double)pre[i];
}

// ---------------------------------------------------------------------------
// (3) injections: SUSTAINED step currents, ASSIGN semantics (a later injection
// on the same neuron wins) — exactly like apply_injection() on the CPU.
// One launch per injection over the contiguous range [lo, lo + count).
// ---------------------------------------------------------------------------
__kernel void ndl_inject_range(__global double* hold, const uint base,
                               const uint n, const double cur) {
  const uint k = (uint)get_global_id(0);
  if (k < n) hold[base + k] = cur;
}

// ---------------------------------------------------------------------------
// (3b) oscillator: I_osc computed on the host (one sin per oscillator per tick)
// ---------------------------------------------------------------------------
__kernel void ndl_osc_add(__global double* isyn, const double iosc, const uint n) {
  const uint i = (uint)get_global_id(0);
  if (i >= n) return;
  isyn[i] += iosc;
}

// ---------------------------------------------------------------------------
// (3c) external stream: Poisson-encode the latest vector into the bound group.
// p = clamp(latest[k], 0, 1) * max_freq * dt / 1000; spike → i_syn += kick.
// ---------------------------------------------------------------------------
__kernel void ndl_stream_poisson(__global ulong* rng, __global const double* latest,
                                 const double p_scale, const double kick,
                                 __global double* isyn, const uint n) {
  const uint k = (uint)get_global_id(0);
  if (k >= n) return;
  double x = latest[k];
  if (x > 1.0) x = 1.0;
  if (x < 0.0) x = 0.0;
  const double p = x * p_scale;
  if (p > 0.0 && ndl_uniform01(&rng[k]) < p) isyn[k] += kick;
}

// ---------------------------------------------------------------------------
// (5) LIF step. Membrane noise (optional): v' += sigma * sqrt(dt) * N(0,1).
// ---------------------------------------------------------------------------
__kernel void ndl_lif(__global double* v, __global const double* isyn,
                      __global const double* hold, __global uchar* spikes,
                      const double dt, const double tau, const double thr,
                      const double rest, const double reset,
                      const double noise_sigma, __global ulong* nrng,
                      const uint n) {
  const uint i = (uint)get_global_id(0);
  if (i >= n) return;
  double nv = v[i] + dt * (rest - v[i] + isyn[i] + hold[i]) / tau;
  if (noise_sigma > 0.0) nv += noise_sigma * sqrt(dt) * ndl_gauss(&nrng[i]);
  const uchar s = nv >= thr ? (uchar)1 : (uchar)0;
  v[i] = s ? reset : nv;
  spikes[i] = s;
}

// ---------------------------------------------------------------------------
// (6) STDP × M(t) with the v2.0 3-factor rule and the +-100 safety clamp.
// ---------------------------------------------------------------------------
__kernel void ndl_stdp_dense(__global double* w, __global const double* pre_tr,
                             __global const uchar* post_sp,
                             __global const double* post_tr,
                             __global const uchar* pre_prev, const double lr_pot,
                             const double lr_dep, const double mod, const uint nsrc,
                             const uint n) {
  const uint j = (uint)get_global_id(0);
  if (j >= n) return;
  const double pot = lr_pot * (double)post_sp[j];
  const double dep = lr_dep * (double)post_tr[j];
  __global double* row = w + (size_t)j * nsrc;
  for (uint i = 0; i < nsrc; ++i) {
    double nw = row[i] + mod * (pot * pre_tr[i] - dep * (double)pre_prev[i]);
    if (nw > 100.0) nw = 100.0;
    if (nw < -100.0) nw = -100.0;
    row[i] = nw;
  }
}

__kernel void ndl_stdp_csr(__global const ulong* rowptr, __global const uint* cols,
                           __global double* vals, __global const double* pre_tr,
                           __global const uchar* post_sp,
                           __global const double* post_tr,
                           __global const uchar* pre_prev, const double lr_pot,
                           const double lr_dep, const double mod, const uint n) {
  const uint j = (uint)get_global_id(0);
  if (j >= n) return;
  const double pot = lr_pot * (double)post_sp[j];
  const double dep = lr_dep * (double)post_tr[j];
  for (ulong k = rowptr[j]; k < rowptr[j + 1]; ++k) {
    const uint i = cols[k];
    double nw = vals[k] + mod * (pot * pre_tr[i] - dep * (double)pre_prev[i]);
    if (nw > 100.0) nw = 100.0;
    if (nw < -100.0) nw = -100.0;
    vals[k] = nw;
  }
}

__kernel void ndl_stdp_o2o(__global double* w, __global const double* pre_tr,
                           __global const uchar* post_sp,
                           __global const double* post_tr,
                           __global const uchar* pre_prev, const double lr_pot,
                           const double lr_dep, const double mod, const uint n) {
  const uint i = (uint)get_global_id(0);
  if (i >= n) return;
  double nw = w[i] + mod * (lr_pot * pre_tr[i] * (double)post_sp[i] -
                            lr_dep * post_tr[i] * (double)pre_prev[i]);
  if (nw > 100.0) nw = 100.0;
  if (nw < -100.0) nw = -100.0;
  w[i] = nw;
}

// ---------------------------------------------------------------------------
// (7) eligibility traces
// ---------------------------------------------------------------------------
__kernel void ndl_trace(__global double* trace, __global const uchar* spikes,
                        const double dec, const uint n) {
  const uint i = (uint)get_global_id(0);
  if (i >= n) return;
  trace[i] = trace[i] * dec + (double)spikes[i];
}

// ---------------------------------------------------------------------------
// Homeostatic synaptic scaling (shrink-only), two passes over a connection:
// rowsum: sum of |w| per postsynaptic row; scale: if sum > target, scale the
// row down to target. Same formula as apply_normalize() on the CPU.
// ---------------------------------------------------------------------------
__kernel void ndl_norm_rowsum_csr(__global const ulong* rowptr,
                                  __global const double* vals,
                                  __global double* rowsum, const uint n) {
  const uint j = (uint)get_global_id(0);
  if (j >= n) return;
  double sum = 0.0;
  for (ulong k = rowptr[j]; k < rowptr[j + 1]; ++k) sum += fabs(vals[k]);
  rowsum[j] = sum;
}

__kernel void ndl_norm_scale_csr(__global const ulong* rowptr, __global double* vals,
                                 __global const double* rowsum, const double target,
                                 const uint n) {
  const uint j = (uint)get_global_id(0);
  if (j >= n) return;
  const double sum = rowsum[j];
  if (sum > target) {
    const double f = target / sum;
    for (ulong k = rowptr[j]; k < rowptr[j + 1]; ++k) vals[k] *= f;
  }
}

__kernel void ndl_norm_rowsum_dense(__global const double* w, const uint nsrc,
                                    __global double* rowsum, const uint n) {
  const uint j = (uint)get_global_id(0);
  if (j >= n) return;
  double sum = 0.0;
  for (uint i = 0; i < nsrc; ++i) sum += fabs(w[(size_t)j * nsrc + i]);
  rowsum[j] = sum;
}

__kernel void ndl_norm_scale_dense(__global double* w, const uint nsrc,
                                   __global const double* rowsum,
                                   const double target, const uint n) {
  const uint j = (uint)get_global_id(0);
  if (j >= n) return;
  const double sum = rowsum[j];
  if (sum > target) {
    const double f = target / sum;
    for (uint i = 0; i < nsrc; ++i) w[(size_t)j * nsrc + i] *= f;
  }
}

// One-to-one rows are single synapses; the rowsum kernel is trivial.
__kernel void ndl_norm_rowsum_o2o(__global const double* w, __global double* rowsum,
                                  const uint n) {
  const uint i = (uint)get_global_id(0);
  if (i >= n) return;
  rowsum[i] = fabs(w[i]);
}

__kernel void ndl_norm_scale_o2o(__global double* w, __global const double* rowsum,
                                 const double target, const uint n) {
  const uint i = (uint)get_global_id(0);
  if (i >= n) return;
  if (rowsum[i] > target) w[i] *= target / rowsum[i];
}
