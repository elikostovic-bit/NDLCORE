// NDL v1.0 — CUDA PTX code generator — GPU backend
// Implementation of the frozen contract in include/ndl/irgen_ptx.hpp.
//
// Emits one standalone text PTX module (PTX ISA 7.0, .target sm_70,
// .address_size 64, all math in f64) holding the fixed kernel set of
// INTERNALS.md §8:
//
//   ndl_gpu_lif_step        v' = v + dt*(rest - v + I)/tau; spike = v' >= threshold
//   ndl_gpu_propagate_dense i_syn[j] += Σ_i w[j*n_src+i] * f64(s_in[i])
//   ndl_gpu_propagate_csr   i_syn[j] += Σ_k vals[k] * f64(s_in[cols[k]])
//   ndl_gpu_inject_apply    hold[idx[t]] = cur[t]        (step-current assignment)
//   ndl_gpu_hold_apply      i_eff[j] = i_syn[j] + hold[j] (effective current)
//   ndl_gpu_syn_decay       i_syn[i] *= decay            (tau_syn = 5 ms)
//   ndl_gpu_stdp_update     w[j*n_src+i] += lr_pot*tr_src[i]*s_new[j]
//                                         - lr_dep*tr_dst[j]*s_prev[i]
//   ndl_gpu_trace_update    tr[i] = tr[i]*decay + f64(s_new[i])
//
// libndl_rt loads this text via cuModuleLoadData (driver JIT) when a run with
// device=GPU executes and launches each kernel over a 1-D grid covering
// [0, n); every kernel guards with tid >= n -> ret.
//
// Device buffer layouts (§8 contract with libndl_rt):
//   f64 arrays — 8 bytes/elem: v, i_syn, w, vals, cur, tr_src, tr_dst, tr
//   u8 arrays  — 1 byte/elem, values 0/1: spike, s_in, s_prev, s_new
//   u64 arrays — 8 bytes/elem: index arrays rowptr, cols, inject idx
// Pointer params arrive as generic addresses and are converted with
// cvta.to.global.u64 before any ld.global/st.global.
//
// Register convention (uniform across kernels):
//   %r1..%r3 thread coords, %r4 flat thread id;
//   pointer params land in %rd(2k) (generic) / %rd(2k+1) (global-space);
//   f64 params and temporaries in %fd1.. ; predicates %p1 (guard), %p2 (loops).
// The only f64 immediate in the module is +0.0 (accumulator init); every other
// f64 value arrives as a kernel parameter. Loop counters are plain s32/s64.

#include "ndl/irgen_ptx.hpp"

#include <string>
#include <string_view>
#include <utility>

namespace ndl {
namespace {

// ---------------------------------------------------------------------------
// Append-only PTX text builder: one line per call, '\n'-terminated.
// Instruction lines are literal strings — this is assembler, so no runtime
// formatting anywhere except the header comment.
// ---------------------------------------------------------------------------
class PtxOut {
public:
  void line(std::string_view s) {
    out_.append(s);
    out_.push_back('\n');
  }
  void blank() { out_.push_back('\n'); }
  std::string take() { return std::move(out_); }

private:
  std::string out_;
};

// --- shared fragments ------------------------------------------------------

// Register file with margin; uniform across all kernels.
// (.reg .b32 %r<N> declares %r0 .. %r<N-1>.)
void emitRegDecls(PtxOut& o) {
  o.line("  .reg .pred %p<8>;");
  o.line("  .reg .b32 %r<16>;");
  o.line("  .reg .b64 %rd<24>;");
  o.line("  .reg .f64 %fd<16>;");
}

// Global 1-D thread index: %r4 = %ctaid.x * %ntid.x + %tid.x (s32).
void emitThreadIndex(PtxOut& o) {
  o.line("  mov.u32 %r1, %ctaid.x;");
  o.line("  mov.u32 %r2, %ntid.x;");
  o.line("  mov.u32 %r3, %tid.x;");
  o.line("  mad.lo.s32 %r4, %r1, %r2, %r3;");
}

// u64 pointer param -> generic reg, then cvta.to.global.u64 -> global reg.
void loadPtrParam(PtxOut& o, const char* param, const char* genericReg,
                  const char* globalReg) {
  o.line("  ld.param.u64 " + std::string(genericReg) + ", [" + param + "];");
  o.line("  cvta.to.global.u64 " + std::string(globalReg) + ", " + genericReg + ";");
}

void loadF64Param(PtxOut& o, const char* param, const char* reg) {
  o.line("  ld.param.f64 " + std::string(reg) + ", [" + param + "];");
}

void loadU32Param(PtxOut& o, const char* param, const char* reg) {
  o.line("  ld.param.u32 " + std::string(reg) + ", [" + param + "];");
}

// Guard: if (*idxReg >= *nReg) branch to doneLabel (uses predicate %p1).
void emitGuardU32(PtxOut& o, const char* idxReg, const char* nReg,
                  const char* doneLabel) {
  o.line("  setp.ge.s32 %p1, " + std::string(idxReg) + ", " + nReg + ";");
  o.line("  @%p1 bra " + std::string(doneLabel) + ";");
}

// ---------------------------------------------------------------------------
// ndl_gpu_lif_step(u64 v, u64 i_syn, u64 spike, f64 tau, f64 threshold,
//                  f64 rest, f64 reset, f64 dt, u32 n)
//   i  = tid; if (i >= n) ret
//   I  = i_syn[i]; v' = v + dt*(rest - v + I)/tau; s = (v' >= threshold)
//   v[i] = s ? reset : v';  spike[i] = s ? 1 : 0
// regs: %rd2 v*  %rd4 i_syn*  %rd6 spike* (global-space)
// ---------------------------------------------------------------------------
void emitLifStep(PtxOut& o) {
  o.line("// ---------------------------------------------------------------------------");
  o.line("// ndl_gpu_lif_step: LIF membrane update + spike output");
  o.line("// ---------------------------------------------------------------------------");
  o.line(".visible .entry ndl_gpu_lif_step(");
  o.line("    .param .u64 p0,");  // v      — f64[n]
  o.line("    .param .u64 p1,");  // i_syn  — f64[n]
  o.line("    .param .u64 p2,");  // spike  — u8[n]
  o.line("    .param .f64 p3,");  // tau
  o.line("    .param .f64 p4,");  // threshold
  o.line("    .param .f64 p5,");  // rest
  o.line("    .param .f64 p6,");  // reset
  o.line("    .param .f64 p7,");  // dt
  o.line("    .param .u32 p8");   // n
  o.line(") {");
  emitRegDecls(o);
  emitThreadIndex(o);
  o.blank();
  loadPtrParam(o, "p0", "%rd1", "%rd2");  // v
  loadPtrParam(o, "p1", "%rd3", "%rd4");  // i_syn
  loadPtrParam(o, "p2", "%rd5", "%rd6");  // spike
  loadF64Param(o, "p3", "%fd1");          // tau
  loadF64Param(o, "p4", "%fd2");          // threshold
  loadF64Param(o, "p5", "%fd3");          // rest
  loadF64Param(o, "p6", "%fd4");          // reset
  loadF64Param(o, "p7", "%fd5");          // dt
  loadU32Param(o, "p8", "%r5");           // n
  o.blank();
  emitGuardU32(o, "%r4", "%r5", "L0");    // i >= n -> done
  o.blank();
  o.line("  mul.wide.s32 %rd7, %r4, 8;");
  o.line("  add.s64 %rd8, %rd2, %rd7;         // &v[i]");
  o.line("  add.s64 %rd9, %rd4, %rd7;         // &i_syn[i]");
  o.line("  ld.global.f64 %fd6, [%rd8];       // v");
  o.line("  ld.global.f64 %fd7, [%rd9];       // I = i_syn[i]");
  o.line("  sub.f64 %fd8, %fd3, %fd6;         // rest - v");
  o.line("  add.f64 %fd9, %fd8, %fd7;         // rest - v + I");
  o.line("  mul.f64 %fd10, %fd5, %fd9;        // dt*(rest - v + I)");
  o.line("  div.rn.f64 %fd11, %fd10, %fd1;    // ... / tau");
  o.line("  add.f64 %fd12, %fd6, %fd11;       // v' = v + dt*(rest-v+I)/tau");
  o.line("  setp.ge.f64 %p2, %fd12, %fd2;     // s = (v' >= threshold)");
  o.line("  selp.f64 %fd13, %fd4, %fd12, %p2; // v'' = s ? reset : v'");
  o.line("  st.global.f64 [%rd8], %fd13;      // v[i] = v''");
  o.line("  selp.s32 %r6, 1, 0, %p2;          // spike byte = s ? 1 : 0");
  o.line("  mul.wide.s32 %rd7, %r4, 1;");
  o.line("  add.s64 %rd10, %rd6, %rd7;        // &spike[i]");
  o.line("  st.global.u8 [%rd10], %r6;        // spike[i] = byte");
  o.line("L0:");
  o.line("  ret;");
  o.line("}");
}

// ---------------------------------------------------------------------------
// ndl_gpu_propagate_dense(u64 w, u64 s_in, u64 i_syn, u32 n_src, u32 n_dst)
//   j = tid; if (j >= n_dst) ret
//   acc = 0; for i in [0, n_src): acc += w[j*n_src + i] * f64(s_in[i])
//   i_syn[j] += acc
// regs: %rd2 w*  %rd4 s_in*  %rd6 i_syn* (global-space)
// ---------------------------------------------------------------------------
void emitPropagateDense(PtxOut& o) {
  o.line("// ---------------------------------------------------------------------------");
  o.line("// ndl_gpu_propagate_dense: i_syn[j] += Σ_i w[j*n_src+i]*f64(s_in[i])");
  o.line("// ---------------------------------------------------------------------------");
  o.line(".visible .entry ndl_gpu_propagate_dense(");
  o.line("    .param .u64 p0,");  // w     — f64[n_dst*n_src], row-major
  o.line("    .param .u64 p1,");  // s_in  — u8[n_src]
  o.line("    .param .u64 p2,");  // i_syn — f64[n_dst]
  o.line("    .param .u32 p3,");  // n_src
  o.line("    .param .u32 p4");   // n_dst
  o.line(") {");
  emitRegDecls(o);
  emitThreadIndex(o);
  o.blank();
  loadPtrParam(o, "p0", "%rd1", "%rd2");  // w
  loadPtrParam(o, "p1", "%rd3", "%rd4");  // s_in
  loadPtrParam(o, "p2", "%rd5", "%rd6");  // i_syn
  loadU32Param(o, "p3", "%r5");           // n_src
  loadU32Param(o, "p4", "%r6");           // n_dst
  o.blank();
  emitGuardU32(o, "%r4", "%r6", "L3");    // j >= n_dst -> done
  o.blank();
  o.line("  mov.f64 %fd1, 0d0000000000000000; // acc = 0.0");
  o.line("  mov.u32 %r7, 0;                   // i = 0");
  o.line("L1:");
  o.line("  setp.ge.s32 %p2, %r7, %r5;        // i >= n_src ?");
  o.line("  @%p2 bra L2;");
  o.line("  mad.lo.s32 %r8, %r4, %r5, %r7;    // off = j*n_src + i");
  o.line("  mul.wide.s32 %rd7, %r8, 8;");
  o.line("  add.s64 %rd8, %rd2, %rd7;         // &w[off]");
  o.line("  ld.global.f64 %fd2, [%rd8];");
  o.line("  mul.wide.s32 %rd7, %r7, 1;");
  o.line("  add.s64 %rd9, %rd4, %rd7;         // &s_in[i]");
  o.line("  ld.global.u8 %r9, [%rd9];");
  o.line("  cvt.rn.f64.s32 %fd3, %r9;         // f64(s_in[i])");
  o.line("  mul.f64 %fd4, %fd2, %fd3;");
  o.line("  add.f64 %fd1, %fd1, %fd4;         // acc += w * s");
  o.line("  add.s32 %r7, %r7, 1;");
  o.line("  bra L1;");
  o.line("L2:");
  o.line("  mul.wide.s32 %rd7, %r4, 8;");
  o.line("  add.s64 %rd10, %rd6, %rd7;        // &i_syn[j]");
  o.line("  ld.global.f64 %fd5, [%rd10];");
  o.line("  add.f64 %fd6, %fd5, %fd1;");
  o.line("  st.global.f64 [%rd10], %fd6;      // i_syn[j] += acc");
  o.line("L3:");
  o.line("  ret;");
  o.line("}");
}

// ---------------------------------------------------------------------------
// ndl_gpu_propagate_csr(u64 rowptr, u64 cols, u64 vals, u64 s_in, u64 i_syn,
//                       u32 n_dst)
//   j = tid; if (j >= n_dst) ret
//   acc = Σ_{k = rowptr[j] .. rowptr[j+1]-1} vals[k] * f64(s_in[cols[k]])
//   i_syn[j] += acc
// regs: %rd2 rowptr*  %rd4 cols*  %rd6 vals*  %rd8 s_in*  %rd10 i_syn*
// ---------------------------------------------------------------------------
void emitPropagateCsr(PtxOut& o) {
  o.line("// ---------------------------------------------------------------------------");
  o.line("// ndl_gpu_propagate_csr: i_syn[j] += Σ_k vals[k]*f64(s_in[cols[k]])");
  o.line("// ---------------------------------------------------------------------------");
  o.line(".visible .entry ndl_gpu_propagate_csr(");
  o.line("    .param .u64 p0,");  // rowptr — u64[n_dst+1]
  o.line("    .param .u64 p1,");  // cols   — u64[nnz]
  o.line("    .param .u64 p2,");  // vals   — f64[nnz]
  o.line("    .param .u64 p3,");  // s_in   — u8[n_src]
  o.line("    .param .u64 p4,");  // i_syn  — f64[n_dst]
  o.line("    .param .u32 p5");   // n_dst
  o.line(") {");
  emitRegDecls(o);
  emitThreadIndex(o);
  o.blank();
  loadPtrParam(o, "p0", "%rd1", "%rd2");   // rowptr
  loadPtrParam(o, "p1", "%rd3", "%rd4");   // cols
  loadPtrParam(o, "p2", "%rd5", "%rd6");   // vals
  loadPtrParam(o, "p3", "%rd7", "%rd8");   // s_in
  loadPtrParam(o, "p4", "%rd9", "%rd10");  // i_syn
  loadU32Param(o, "p5", "%r5");            // n_dst
  o.blank();
  emitGuardU32(o, "%r4", "%r5", "L4");     // j >= n_dst -> done
  o.blank();
  o.line("  mul.wide.s32 %rd11, %r4, 8;");
  o.line("  add.s64 %rd12, %rd2, %rd11;       // &rowptr[j]");
  o.line("  ld.global.u64 %rd13, [%rd12];     // k_lo = rowptr[j]");
  o.line("  add.s32 %r7, %r4, 1;");
  o.line("  mul.wide.s32 %rd11, %r7, 8;");
  o.line("  add.s64 %rd14, %rd2, %rd11;       // &rowptr[j+1]");
  o.line("  ld.global.u64 %rd15, [%rd14];     // k_hi = rowptr[j+1]");
  o.line("  mov.f64 %fd1, 0d0000000000000000; // acc = 0.0");
  o.line("  mov.u64 %rd16, %rd13;             // k = k_lo");
  o.line("L2:");
  o.line("  setp.ge.s64 %p2, %rd16, %rd15;    // k >= k_hi ?");
  o.line("  @%p2 bra L3;");
  o.line("  mul.lo.s64 %rd17, %rd16, 8;       // k*8");
  o.line("  add.s64 %rd18, %rd6, %rd17;       // &vals[k]");
  o.line("  ld.global.f64 %fd2, [%rd18];");
  o.line("  add.s64 %rd19, %rd4, %rd17;       // &cols[k]");
  o.line("  ld.global.u64 %rd20, [%rd19];     // col = cols[k]");
  o.line("  add.s64 %rd21, %rd8, %rd20;       // &s_in[col] (1 byte/elem)");
  o.line("  ld.global.u8 %r6, [%rd21];");
  o.line("  cvt.rn.f64.s32 %fd3, %r6;         // f64(s_in[col])");
  o.line("  mul.f64 %fd4, %fd2, %fd3;");
  o.line("  add.f64 %fd1, %fd1, %fd4;         // acc += vals[k] * s");
  o.line("  add.s64 %rd16, %rd16, 1;          // ++k");
  o.line("  bra L2;");
  o.line("L3:");
  o.line("  mul.wide.s32 %rd22, %r4, 8;");
  o.line("  add.s64 %rd23, %rd10, %rd22;      // &i_syn[j]");
  o.line("  ld.global.f64 %fd5, [%rd23];");
  o.line("  add.f64 %fd6, %fd5, %fd1;");
  o.line("  st.global.f64 [%rd23], %fd6;      // i_syn[j] += acc");
  o.line("L4:");
  o.line("  ret;");
  o.line("}");
}

// ---------------------------------------------------------------------------
// ndl_gpu_inject_apply(u64 idx, u64 cur, u32 count, u64 hold)
//   t = tid; if (t < count) hold[idx[t]] = cur[t]
// regs: %rd2 idx*  %rd4 cur*  %rd6 hold* (global-space)
// ---------------------------------------------------------------------------
void emitInjectApply(PtxOut& o) {
  o.line("// ---------------------------------------------------------------------------");
  o.line("// ndl_gpu_inject_apply: hold[idx[t]] = cur[t]  (step-current assignment)");
  o.line("// ---------------------------------------------------------------------------");
  o.line(".visible .entry ndl_gpu_inject_apply(");
  o.line("    .param .u64 p0,");  // idx  — u64[count]
  o.line("    .param .u64 p1,");  // cur  — f64[count]
  o.line("    .param .u32 p2,");  // count
  o.line("    .param .u64 p3");   // hold — f64[n]
  o.line(") {");
  emitRegDecls(o);
  emitThreadIndex(o);
  o.blank();
  loadPtrParam(o, "p0", "%rd1", "%rd2");  // idx
  loadPtrParam(o, "p1", "%rd3", "%rd4");  // cur
  loadU32Param(o, "p2", "%r5");           // count
  loadPtrParam(o, "p3", "%rd5", "%rd6");  // hold
  o.blank();
  emitGuardU32(o, "%r4", "%r5", "L1");    // t >= count -> done
  o.blank();
  o.line("  mul.wide.s32 %rd7, %r4, 8;");
  o.line("  add.s64 %rd8, %rd2, %rd7;         // &idx[t]");
  o.line("  ld.global.u64 %rd9, [%rd8];       // neuron = idx[t]");
  o.line("  add.s64 %rd10, %rd4, %rd7;        // &cur[t] (same t*8 offset)");
  o.line("  ld.global.f64 %fd1, [%rd10];");
  o.line("  mul.lo.s64 %rd11, %rd9, 8;        // neuron*8");
  o.line("  add.s64 %rd12, %rd6, %rd11;       // &hold[neuron]");
  o.line("  st.global.f64 [%rd12], %fd1;      // hold[neuron] = cur[t]");
  o.line("L1:");
  o.line("  ret;");
  o.line("}");
}

// ---------------------------------------------------------------------------
// ndl_gpu_hold_apply(u64 syn, u64 hold, u64 i_eff, u32 n)
//   j = tid; if (j < n) i_eff[j] = syn[j] + hold[j]
// ---------------------------------------------------------------------------
void emitHoldApply(PtxOut& o) {
  o.line("// ---------------------------------------------------------------------------");
  o.line("// ndl_gpu_hold_apply: i_eff[j] = syn[j] + hold[j]");
  o.line("// ---------------------------------------------------------------------------");
  o.line(".visible .entry ndl_gpu_hold_apply(");
  o.line("    .param .u64 p0,");  // syn  — f64[n] (decaying synaptic component)
  o.line("    .param .u64 p1,");  // hold — f64[n] (sustained step component)
  o.line("    .param .u64 p2,");  // i_eff — f64[n] (effective current, rewritten)
  o.line("    .param .u32 p3");   // n
  o.line(") {");
  emitRegDecls(o);
  emitThreadIndex(o);
  o.blank();
  loadPtrParam(o, "p0", "%rd1", "%rd2");
  loadPtrParam(o, "p1", "%rd3", "%rd4");
  loadPtrParam(o, "p2", "%rd5", "%rd6");
  loadU32Param(o, "p3", "%r5");
  o.blank();
  emitGuardU32(o, "%r4", "%r5", "L1");
  o.blank();
  o.line("  mul.wide.s32 %rd7, %r4, 8;");
  o.line("  add.s64 %rd8, %rd2, %rd7;");
  o.line("  ld.global.f64 %fd1, [%rd8];       // syn[j]");
  o.line("  add.s64 %rd9, %rd4, %rd7;");
  o.line("  ld.global.f64 %fd2, [%rd9];       // hold[j]");
  o.line("  add.f64 %fd3, %fd1, %fd2;");
  o.line("  add.s64 %rd10, %rd6, %rd7;");
  o.line("  st.global.f64 [%rd10], %fd3;      // i_eff[j] = syn[j] + hold[j]");
  o.line("L1:");
  o.line("  ret;");
  o.line("}");
}

// ---------------------------------------------------------------------------
// ndl_gpu_syn_decay(u64 i_syn, f64 decay, u32 n)
//   i = tid; if (i < n) i_syn[i] = i_syn[i] * decay
// ---------------------------------------------------------------------------
void emitSynDecay(PtxOut& o) {
  o.line("// ---------------------------------------------------------------------------");
  o.line("// ndl_gpu_syn_decay: i_syn[i] = i_syn[i]*decay  (exponential synapse, tau_syn)");
  o.line("// ---------------------------------------------------------------------------");
  o.line(".visible .entry ndl_gpu_syn_decay(");
  o.line("    .param .u64 p0,");  // i_syn — f64[n]
  o.line("    .param .f64 p1,");  // decay
  o.line("    .param .u32 p2");   // n
  o.line(") {");
  emitRegDecls(o);
  emitThreadIndex(o);
  o.blank();
  loadPtrParam(o, "p0", "%rd1", "%rd2");
  loadF64Param(o, "p1", "%fd1");
  loadU32Param(o, "p2", "%r5");
  o.blank();
  emitGuardU32(o, "%r4", "%r5", "L1");
  o.blank();
  o.line("  mul.wide.s32 %rd3, %r4, 8;");
  o.line("  add.s64 %rd4, %rd2, %rd3;");
  o.line("  ld.global.f64 %fd2, [%rd4];      // i_syn[i]");
  o.line("  mul.f64 %fd3, %fd2, %fd1;");
  o.line("  st.global.f64 [%rd4], %fd3;       // i_syn[i] *= decay");
  o.line("L1:");
  o.line("  ret;");
  o.line("}");
}

// ---------------------------------------------------------------------------
// ndl_gpu_stdp_update(u64 w, u64 tr_src, u64 tr_dst, u64 s_prev, u64 s_new,
//                     f64 lr_pot, f64 lr_dep, u32 n_src, u32 n_dst)
//   j = tid; if (j >= n_dst) ret
//   for i in [0, n_src):
//     w[j*n_src + i] += lr_pot*tr_src[i]*f64(s_new[j])
//                     - lr_dep*tr_dst[j]*f64(s_prev[i])
// regs: %rd2 w*  %rd4 tr_src*  %rd6 tr_dst*  %rd8 s_prev*  %rd10 s_new*
// ---------------------------------------------------------------------------
void emitStdpUpdate(PtxOut& o, bool modulated) {
  o.line("// ---------------------------------------------------------------------------");
  if (modulated) {
    o.line("// ndl_gpu_stdp_update (v2.0 modulated 3-factor): w += M(t) * (lr_pot*tr_src[i]*s_new[j]");
    o.line("//                      - lr_dep*tr_dst[j]*s_prev[i])");
  } else {
    o.line("// ndl_gpu_stdp_update: w += lr_pot*tr_src[i]*s_new[j]");
    o.line("//                      - lr_dep*tr_dst[j]*s_prev[i]");
  }
  o.line("// ---------------------------------------------------------------------------");
  o.line(".visible .entry ndl_gpu_stdp_update(");
  o.line("    .param .u64 p0,");  // w      — f64[n_dst*n_src], row-major
  o.line("    .param .u64 p1,");  // tr_src — f64[n_src]
  o.line("    .param .u64 p2,");  // tr_dst — f64[n_dst]
  o.line("    .param .u64 p3,");  // s_prev — u8[n_src]
  o.line("    .param .u64 p4,");  // s_new  — u8[n_dst]
  o.line("    .param .f64 p5,");  // lr_pot
  o.line("    .param .f64 p6,");  // lr_dep
  o.line("    .param .u32 p7,");  // n_src
  o.line("    .param .u32 p8");   // n_dst
  if (modulated) {
    o.line("    .param .f64 p9");   // v2.0: M(t) modulator value
    o.line(") {");
    emitRegDecls(o);
    emitThreadIndex(o);
    o.blank();
    loadPtrParam(o, "p0", "%rd1", "%rd2");   // w
    loadPtrParam(o, "p1", "%rd3", "%rd4");   // tr_src
    loadPtrParam(o, "p2", "%rd5", "%rd6");   // tr_dst
    loadPtrParam(o, "p3", "%rd7", "%rd8");   // s_prev
    loadPtrParam(o, "p4", "%rd9", "%rd10");  // s_new
    loadF64Param(o, "p5", "%fd1");           // lr_pot
    loadF64Param(o, "p6", "%fd2");           // lr_dep
    loadU32Param(o, "p7", "%r5");            // n_src
    loadU32Param(o, "p8", "%r6");            // n_dst
    loadF64Param(o, "p9", "%fd14");          // modulator M(t)
  } else {
    o.line(") {");
    emitRegDecls(o);
    emitThreadIndex(o);
    o.blank();
    loadPtrParam(o, "p0", "%rd1", "%rd2");   // w
    loadPtrParam(o, "p1", "%rd3", "%rd4");   // tr_src
    loadPtrParam(o, "p2", "%rd5", "%rd6");   // tr_dst
    loadPtrParam(o, "p3", "%rd7", "%rd8");   // s_prev
    loadPtrParam(o, "p4", "%rd9", "%rd10");  // s_new
    loadF64Param(o, "p5", "%fd1");           // lr_pot
    loadF64Param(o, "p6", "%fd2");           // lr_dep
    loadU32Param(o, "p7", "%r5");            // n_src
    loadU32Param(o, "p8", "%r6");            // n_dst
  }
  o.blank();
  emitGuardU32(o, "%r4", "%r6", "L3");     // j >= n_dst -> done
  o.blank();
  // Hoist the j-dependent values (loop-invariant) before the i loop.
  o.line("  mul.wide.s32 %rd11, %r4, 1;");
  o.line("  add.s64 %rd16, %rd10, %rd11;      // &s_new[j]");
  o.line("  ld.global.u8 %r10, [%rd16];");
  o.line("  cvt.rn.f64.s32 %fd5, %r10;        // f64(s_new[j])");
  o.line("  mul.wide.s32 %rd11, %r4, 8;");
  o.line("  add.s64 %rd14, %rd6, %rd11;       // &tr_dst[j]");
  o.line("  ld.global.f64 %fd3, [%rd14];      // tr_dst[j]");
  o.line("  mul.f64 %fd4, %fd2, %fd3;         // lr_dep * tr_dst[j]");
  o.line("  mov.u32 %r7, 0;                   // i = 0");
  o.line("L1:");
  o.line("  setp.ge.s32 %p2, %r7, %r5;        // i >= n_src ?");
  o.line("  @%p2 bra L3;");
  o.line("  mad.lo.s32 %r8, %r4, %r5, %r7;    // off = j*n_src + i");
  o.line("  mul.wide.s32 %rd11, %r8, 8;");
  o.line("  add.s64 %rd12, %rd2, %rd11;       // &w[off]");
  o.line("  ld.global.f64 %fd6, [%rd12];      // w");
  o.line("  mul.wide.s32 %rd11, %r7, 8;");
  o.line("  add.s64 %rd13, %rd4, %rd11;       // &tr_src[i]");
  o.line("  ld.global.f64 %fd7, [%rd13];");
  o.line("  mul.wide.s32 %rd11, %r7, 1;");
  o.line("  add.s64 %rd15, %rd8, %rd11;       // &s_prev[i]");
  o.line("  ld.global.u8 %r9, [%rd15];");
  o.line("  cvt.rn.f64.s32 %fd8, %r9;         // f64(s_prev[i])");
  o.line("  mul.f64 %fd9, %fd1, %fd7;         // lr_pot * tr_src[i]");
  o.line("  mul.f64 %fd10, %fd9, %fd5;        // X = ... * s_new[j]");
  o.line("  mul.f64 %fd11, %fd4, %fd8;        // Y = lr_dep*tr_dst[j]*s_prev[i]");
  o.line("  sub.f64 %fd12, %fd10, %fd11;      // X - Y");
  if (modulated) o.line("  mul.f64 %fd12, %fd12, %fd14;      // (X - Y) * M(t)   [v2.0]");
  o.line("  add.f64 %fd13, %fd6, %fd12;       // w + (X - Y)");
  o.line("  st.global.f64 [%rd12], %fd13;     // w[off] = ...");
  o.line("  add.s32 %r7, %r7, 1;");
  o.line("  bra L1;");
  o.line("L3:");
  o.line("  ret;");
  o.line("}");
}

// ---------------------------------------------------------------------------
// v2.0 kernels. The v2.0 simulation-time features execute on the CPU path of
// libndl_rt; these kernels keep the §8 GPU contract complete so that future
// executor work can schedule them without a blob format change.
// ---------------------------------------------------------------------------

// ndl_gpu_oscillator_apply(u64 i_syn, f64 i_osc, u32 n)
//   i_syn[i] += i_osc   (the sinusoidal value is computed host-side per tick)
void emitOscillatorApply(PtxOut& o) {
  o.line("// ndl_gpu_oscillator_apply: i_syn[i] += i_osc  (v2.0 background rhythm)");
  o.line(".visible .entry ndl_gpu_oscillator_apply(");
  o.line("    .param .u64 p0,");  // i_syn — f64[n]
  o.line("    .param .f64 p1,");  // i_osc (scalar for this tick)
  o.line("    .param .u32 p2");   // n
  o.line(") {");
  emitRegDecls(o);
  emitThreadIndex(o);
  o.blank();
  loadPtrParam(o, "p0", "%rd1", "%rd2");
  loadF64Param(o, "p1", "%fd1");
  loadU32Param(o, "p2", "%r5");
  o.blank();
  emitGuardU32(o, "%r4", "%r5", "L1");
  o.line("  mul.wide.s32 %rd11, %r4, 8;");
  o.line("  add.s64 %rd3, %rd2, %rd11;        // &i_syn[i]");
  o.line("  ld.global.f64 %fd2, [%rd3];");
  o.line("  add.f64 %fd3, %fd2, %fd1;");
  o.line("  st.global.f64 [%rd3], %fd3;");
  o.line("L1:");
  o.line("  ret;");
  o.line("}");
}

// ndl_gpu_prune_dense(u64 w, f64 threshold, u32 n_total)
//   if (|w[i]| < threshold) w[i] = 0
void emitPruneDense(PtxOut& o) {
  o.line("// ndl_gpu_prune_dense: w[i] = |w[i]| < threshold ? 0 : w[i]  (v2.0)");
  o.line(".visible .entry ndl_gpu_prune_dense(");
  o.line("    .param .u64 p0,");  // w — f64[n]
  o.line("    .param .f64 p1,");  // threshold
  o.line("    .param .u32 p2");   // n_total
  o.line(") {");
  emitRegDecls(o);
  emitThreadIndex(o);
  o.blank();
  loadPtrParam(o, "p0", "%rd1", "%rd2");
  loadF64Param(o, "p1", "%fd1");
  loadU32Param(o, "p2", "%r5");
  o.blank();
  emitGuardU32(o, "%r4", "%r5", "L1");
  o.line("  mul.wide.s32 %rd11, %r4, 8;");
  o.line("  add.s64 %rd3, %rd2, %rd11;        // &w[i]");
  o.line("  ld.global.f64 %fd2, [%rd3];");
  o.line("  abs.f64 %fd3, %fd2;");
  o.line("  setp.lt.f64 %p2, %fd3, %fd1;");
  o.line("  mov.f64 %fd4, 0d0000000000000000; // +0.0");
  o.line("  @%p2 st.global.f64 [%rd3], %fd4;");
  o.line("L1:");
  o.line("  ret;");
  o.line("}");
}

// ndl_gpu_prune_csr(u64 vals, f64 threshold, u32 nnz)
//   if (|vals[k]| < threshold) vals[k] = 0
void emitPruneCsr(PtxOut& o) {
  o.line("// ndl_gpu_prune_csr: vals[k] = |vals[k]| < threshold ? 0 : vals[k]  (v2.0)");
  o.line(".visible .entry ndl_gpu_prune_csr(");
  o.line("    .param .u64 p0,");  // vals — f64[nnz]
  o.line("    .param .f64 p1,");  // threshold
  o.line("    .param .u32 p2");   // nnz
  o.line(") {");
  emitRegDecls(o);
  emitThreadIndex(o);
  o.blank();
  loadPtrParam(o, "p0", "%rd1", "%rd2");
  loadF64Param(o, "p1", "%fd1");
  loadU32Param(o, "p2", "%r5");
  o.blank();
  emitGuardU32(o, "%r4", "%r5", "L1");
  o.line("  mul.wide.s32 %rd11, %r4, 8;");
  o.line("  add.s64 %rd3, %rd2, %rd11;        // &vals[k]");
  o.line("  ld.global.f64 %fd2, [%rd3];");
  o.line("  abs.f64 %fd3, %fd2;");
  o.line("  setp.lt.f64 %p2, %fd3, %fd1;");
  o.line("  mov.f64 %fd4, 0d0000000000000000; // +0.0");
  o.line("  @%p2 st.global.f64 [%rd3], %fd4;");
  o.line("L1:");
  o.line("  ret;");
  o.line("}");
}

// ---------------------------------------------------------------------------
// ndl_gpu_trace_update(u64 tr, u64 s_new, f64 decay, u32 n)
//   i = tid; if (i >= n) ret
//   tr[i] = tr[i]*decay + f64(s_new[i])
// regs: %rd2 tr*  %rd4 s_new* (global-space)
// ---------------------------------------------------------------------------
void emitTraceUpdate(PtxOut& o) {
  o.line("// ---------------------------------------------------------------------------");
  o.line("// ndl_gpu_trace_update: tr[i] = tr[i]*decay + f64(s_new[i])");
  o.line("// ---------------------------------------------------------------------------");
  o.line(".visible .entry ndl_gpu_trace_update(");
  o.line("    .param .u64 p0,");  // tr    — f64[n]
  o.line("    .param .u64 p1,");  // s_new — u8[n]
  o.line("    .param .f64 p2,");  // decay
  o.line("    .param .u32 p3");   // n
  o.line(") {");
  emitRegDecls(o);
  emitThreadIndex(o);
  o.blank();
  loadPtrParam(o, "p0", "%rd1", "%rd2");  // tr
  loadPtrParam(o, "p1", "%rd3", "%rd4");  // s_new
  loadF64Param(o, "p2", "%fd1");          // decay
  loadU32Param(o, "p3", "%r5");           // n
  o.blank();
  emitGuardU32(o, "%r4", "%r5", "L1");    // i >= n -> done
  o.blank();
  o.line("  mul.wide.s32 %rd5, %r4, 8;");
  o.line("  add.s64 %rd6, %rd2, %rd5;         // &tr[i]");
  o.line("  mul.wide.s32 %rd5, %r4, 1;");
  o.line("  add.s64 %rd7, %rd4, %rd5;         // &s_new[i]");
  o.line("  ld.global.f64 %fd2, [%rd6];       // tr[i]");
  o.line("  ld.global.u8 %r6, [%rd7];");
  o.line("  cvt.rn.f64.s32 %fd3, %r6;         // f64(s_new[i])");
  o.line("  mul.f64 %fd4, %fd2, %fd1;         // tr[i]*decay");
  o.line("  add.f64 %fd5, %fd4, %fd3;");
  o.line("  st.global.f64 [%rd6], %fd5;       // tr[i] = ...");
  o.line("L1:");
  o.line("  ret;");
  o.line("}");
}

// Module header: identification, group inventory from sema, target directives.
void emitHeader(PtxOut& o, const SemaResult& sema) {
  o.line("// NDL v1.0 PTX module generated by ndlc");
  o.line(std::string("// ndlc ") + kNdlcVersion + " (NDL v" + kNdlVersion + ")");
  std::string groups;
  for (const GroupInfo& g : sema.groups) {
    groups += ' ';
    groups += g.name;
    groups += '[';
    groups += std::to_string(g.size);
    groups += ']';
  }
  o.line("// groups:" + (groups.empty() ? std::string(" (none)") : groups));
  o.line("// ABI: INTERNALS.md §8 fixed kernel set; loaded by libndl_rt via");
  o.line("// cuModuleLoadData (driver JIT). 1-D launches, guard tid >= n -> ret.");
  o.line("// layouts: f64 8B/elem; u8 1B/elem (0/1); index arrays u64 8B/elem.");
  o.blank();
  o.line(".version 7.0");
  o.line(".target sm_70");
  o.line(".address_size 64");
}

} // namespace

// ---------------------------------------------------------------------------
// PTXCodeGen
// ---------------------------------------------------------------------------
PTXCodeGen::PTXCodeGen(const Program& program, const SemaResult& sema,
                       DiagnosticEngine& diag)
    : prog_(program), sema_(sema), diag_(diag) {}

std::string PTXCodeGen::generate() {
  // Kernels are the fixed §8 set — the program AST carries no GPU-specific
  // constructs in v1.0 (network topology lives in libndl_rt buffers), so only
  // sema's group table feeds the generated text (header comment).
  PtxOut o;
  emitHeader(o, sema_);
  o.blank();
  emitLifStep(o);
  o.blank();
  emitPropagateDense(o);
  o.blank();
  emitPropagateCsr(o);
  o.blank();
  emitInjectApply(o);
  o.blank();
  emitHoldApply(o);
  o.blank();
  emitSynDecay(o);
  o.blank();
  emitStdpUpdate(o, sema_.hasModulatedStdp);
  o.blank();
  emitTraceUpdate(o);
  if (sema_.hasOscillators) {
    o.blank();
    emitOscillatorApply(o);
  }
  if (sema_.hasPrune) {
    o.blank();
    emitPruneDense(o);
    o.blank();
    emitPruneCsr(o);
  }
  return o.take();
}

} // namespace ndl
