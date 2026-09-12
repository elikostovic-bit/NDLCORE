// NDL v1.0 — LLVM IR (text) code generator.
//
// Emits a complete, standalone LLVM 15+ (opaque pointers) module per
// INTERNALS §7. The compiler has NO LLVM dependency: IR is written as text;
// object emission + optimization run through external tools (clang/llc/opt)
// driven by linker.cpp.
//
// Structure of the emitted module:
//   ; header (ModuleID / source_filename / target triple)
//   declare block        — the full libndl_rt C ABI
//   @grp_<Name>          — one global handle per node group
//   @g_<var>             — one global per direct top-level `let`
//   @.str.N              — interned string constants
//   @ndl_ptx_source      — embedded GPU PTX (when present)
//   @ndl_on_spike.N      — one function per on_spike handler
//   @ndl_user_main       — top-level program statements
//   @main                — entry, calls ndl_user_main, returns 0
//
// Conventions: i64 Int, double Float/Duration, i1 Bool in registers (i8 in
// memory), ptr Str/Tensor/Group. Every variable lives in memory (alloca
// hoisted to the function entry) — mem2reg cleans it up under optimization.
#include "ndl/irgen_llvm.hpp"

#include <cinttypes>
#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ndl {

namespace {

// ---------------------------------------------------------------------------
// LLVM type mapping
// ---------------------------------------------------------------------------
std::string llvmTypeName(const TypeInfo& t) {
  switch (t.k) {
    case TypeInfo::K::Int: return "i64";
    case TypeInfo::K::Float:
    case TypeInfo::K::Duration: return "double";
    case TypeInfo::K::Bool: return "i1";
    case TypeInfo::K::Str:
    case TypeInfo::K::Tensor:
    case TypeInfo::K::Group:
    case TypeInfo::K::Void:
    default: return "ptr";
  }
}

std::string zeroInit(const TypeInfo& t) {
  switch (t.k) {
    case TypeInfo::K::Int: return "0";
    case TypeInfo::K::Float:
    case TypeInfo::K::Duration: return "0.000000e+00";
    case TypeInfo::K::Bool: return "0";
    default: return "null";
  }
}

std::string fmtDouble(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17e", v);
  return buf;
}

std::string llvmEscape(const std::string& s, bool escapeAll) {
  // escapeAll: every byte as \XX (used for the PTX blob).
  std::string out;
  out.reserve(s.size() * 2 + 8);
  for (unsigned char c : s) {
    if (escapeAll) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\%02X", c);
      out += buf;
    } else if (c >= 0x20 && c <= 0x7E && c != '"' && c != '\\') {
      out += (char)c;
    } else {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\%02X", c);
      out += buf;
    }
  }
  return out;
}

bool isTerminatorLine(const std::string& line) {
  return line.rfind("br ", 0) == 0 || line.rfind("ret ", 0) == 0;
}
[[maybe_unused]] bool isTerminator(const std::string& line) { return isTerminatorLine(line); }

// ---------------------------------------------------------------------------
// Function emission context
// ---------------------------------------------------------------------------
struct Slot {
  std::string ref;   // "@g_x" / "%x.1.addr" / "%neuron_index.addr"
  bool isGlobal = false;
  TypeInfo type;
};

struct FnCtx {
  std::string name;
  std::vector<std::string> allocas;                 // hoisted to entry
  std::vector<std::string> lines;                   // instruction stream
  std::vector<std::unordered_map<std::string, Slot>> scopes;
  int tmpN = 0;
  int blkN = 0;
  bool terminated = true;  // nothing emitted yet

  void pushScope() { scopes.emplace_back(); }
  void popScope() { if (scopes.size() > 1) scopes.pop_back(); }

  Slot* find(const std::string& name) {
    for (size_t i = scopes.size(); i-- > 0;) {
      auto it = scopes[i].find(name);
      if (it != scopes[i].end()) return &it->second;
    }
    return nullptr;
  }
  void declare(const std::string& name, const Slot& s) { scopes.back()[name] = s; }

  std::string tmp() { return "%t" + std::to_string(++tmpN); }

  void line(const std::string& s) {
    lines.push_back("  " + s);
    terminated = false;
  }
  void br(const std::string& target) {
    if (terminated) return;  // already has a terminator
    lines.push_back("  br label %" + target);
    terminated = true;
  }
  void brCond(const std::string& cond, const std::string& t, const std::string& f) {
    if (terminated) return;  // defensive
    lines.push_back("  br i1 " + cond + ", label %" + t + ", label %" + f);
    terminated = true;
  }
  void retVoid() {
    if (terminated) return;
    lines.push_back("  ret void");
    terminated = true;
  }
  std::string label(const std::string& hint) {
    if (!terminated) br(hint);  // auto-terminate fallthrough
    std::string name = hint;
    lines.push_back(name + ":");
    terminated = false;
    return name;
  }
  std::string freshBlock(const std::string& hint) {
    return hint + "." + std::to_string(++blkN);
  }
};

class IRGen {
 public:
  IRGen(const Program& p, const SemaResult& s, const LLVMEmitOptions& o,
        DiagnosticEngine& d)
      : prog_(p), sema_(s), opts_(o), diag_(d) {}

  std::string generate() {
    emitHeader();
    emitDeclares();
    emitGlobals();

    // Main function body.
    main_.name = "ndl_user_main";
    main_.pushScope();
    // Global scope: groups + top-level lets (declaration-before-use enforced
    // by sema; pre-population only provides the storage mapping).
    for (const auto& g : sema_.groups)
      main_.declare(g.name, Slot{"@grp_" + g.name, true, TypeInfo::Group()});
    for (const auto& v : sema_.globals)
      main_.declare(v.name, Slot{"@g_" + v.name, true, v.type});
    // v2.0: signals are pre-declared globals (@sig_<Name>, double) so reads
    // and assignments inside handlers/bodies resolve to storage.
    for (const auto& sig : sema_.signals)
      main_.declare(sig.name, Slot{"@sig_" + sig.name, true, TypeInfo::Float()});

    if (!opts_.ptxSource.empty())
      main_.line("call void @ndl_rt_gpu_load_ptx(ptr @ndl_ptx_source)");

    for (const auto& mod : prog_.modules)
      for (const auto& item : mod->items) emitStmt(*item, main_);

    // v2.0: a program that entered continuous mode stops the event loop at
    // program end (mirrors the VM path).
    if (sawRunContinuous_) main_.line("call void @ndl_rt_stop_continuous()");

    main_.retVoid();
    main_.popScope();

    assemble();
    if (failed_) return "";
    return out_.str();
  }

 private:
  const Program& prog_;
  const SemaResult& sema_;
  const LLVMEmitOptions& opts_;
  DiagnosticEngine& diag_;

  std::ostringstream out_;
  std::ostringstream globals_;   // @grp_* and @g_*
  std::ostringstream strings_;   // @.str.*
  std::vector<std::string> handlerFns_;
  FnCtx main_;
  std::map<std::string, std::string> strPool_;
  int strN_ = 0;
  int handlerN_ = 0;
  bool sawRunContinuous_ = false;  // v2.0: emit stop_continuous at exit
  bool failed_ = false;

  void fail(const std::string& msg) {
    if (failed_) return;
    failed_ = true;
    Diagnostic d;
    d.severity = Severity::Error;
    d.code = "E0301";
    d.message = "internal codegen error: " + msg;
    diag_.report(std::move(d));
  }

  // --- module scaffolding ---------------------------------------------------
  void emitHeader() {
    out_ << "; ModuleID = '" << opts_.moduleName << "'\n";
    out_ << "; NDL v1.0 -- generated by ndlc (LLVM IR backend)\n";
    out_ << "source_filename = \"" << opts_.moduleName << "\"\n";
    out_ << "target triple = \"" << opts_.triple << "\"\n\n";
  }

  void emitDeclares() {
    out_ <<
        "declare ptr @ndl_rt_group_create(i64, i32, double, double, double, double, ptr)\n"
        "declare void @ndl_rt_dense_connect(ptr, ptr, i32, double, double, i32)\n"
        "declare void @ndl_rt_sparse_connect(ptr, ptr, double, double, i32)\n"
        "declare void @ndl_rt_one_to_one_connect(ptr, ptr, double, i32)\n"
        "declare void @ndl_rt_configure_stdp(ptr, ptr, double, double, double)\n"
        "declare void @ndl_rt_register_spike_handler(ptr, ptr, ptr)\n"
        "declare void @ndl_rt_schedule_inject(ptr, i64, i64, double, double)\n"
        "declare void @ndl_rt_run(double, double, i32)\n"
        "declare void @ndl_rt_print_begin()\n"
        "declare void @ndl_rt_print_str(ptr)\n"
        "declare void @ndl_rt_print_i64(i64)\n"
        "declare void @ndl_rt_print_f64(double)\n"
        "declare void @ndl_rt_print_bool(i32)\n"
        "declare void @ndl_rt_print_end()\n"
        "declare void @ndl_rt_save_checkpoint(ptr)\n"
        "declare void @ndl_rt_load_checkpoint(ptr)\n"
        "declare void @ndl_rt_export_raster(ptr, ptr)\n"
        "declare void @ndl_rt_gpu_load_ptx(ptr)\n"
        "declare double @ndl_rt_random_uniform(double, double)\n"
        "declare double @ndl_rt_random_gaussian(double, double)\n"
        "declare double @ndl_rt_sin(double)\n"
        "declare double @ndl_rt_cos(double)\n"
        "declare double @ndl_rt_exp(double)\n"
        "declare double @ndl_rt_log(double)\n"
        "declare double @ndl_rt_sqrt(double)\n"
        "declare void @ndl_rt_set_seed(i64)\n"
        "declare i64 @ndl_rt_now_ms()\n"
        "declare i64 @ndl_rt_now_ns()\n"
        "declare ptr @ndl_rt_str_concat(ptr, ptr)\n"
        "declare double @ndl_rt_get_weight(ptr, ptr, i64, i64)\n"
        "declare ptr @ndl_rt_tensor_new(i64, ptr, double)\n"
        "declare double @ndl_rt_tensor_get1(ptr, i64)\n"
        "declare double @ndl_rt_tensor_get2(ptr, i64, i64)\n"
        "declare double @ndl_rt_tensor_get3(ptr, i64, i64, i64)\n"
        "declare void @ndl_rt_tensor_set1(ptr, i64, double)\n"
        "declare void @ndl_rt_tensor_set2(ptr, i64, i64, double)\n"
        "declare void @ndl_rt_tensor_set3(ptr, i64, i64, i64, double)\n"
        "declare void @ndl_rt_dump_tensor(ptr, ptr)\n"
        "declare void @ndl_rt_signal_create(ptr, double)\n"
        "declare void @ndl_rt_signal_set(ptr, double)\n"
        "declare double @ndl_rt_signal_get(ptr)\n"
        "declare void @ndl_rt_oscillator_add(ptr, ptr, double, double, double)\n"
        "declare i64 @ndl_rt_stream_create(ptr, i64)\n"
        "declare void @ndl_rt_bind_input_stream(i64, ptr, i32, double, double)\n"
        "declare void @ndl_rt_stdp_set_modulator(ptr, ptr, ptr)\n"
        "declare void @ndl_rt_set_plasticity(ptr, ptr, i32)\n"
        "declare void @ndl_rt_prune_weights(ptr, ptr, double)\n"
        "declare void @ndl_rt_run_continuous(double, i32)\n"
        "declare void @ndl_rt_stop_continuous()\n"
        "declare void @ndl_rt_wait_continuous()\n"
        "declare ptr @ndl_rt_get_membrane_potentials(ptr)\n"
        "declare ptr @ndl_rt_predict_linear(ptr, ptr)\n"
        "declare double @ndl_rt_vector_l2_norm(ptr)\n"
        "declare void @ndl_rt_stream_set(i64, i64, double)\n"
        "declare i32 @strcmp(ptr, ptr)\n\n";
  }

  void emitGlobals() {
    for (const auto& g : sema_.groups)
      globals_ << "@grp_" << g.name << " = global ptr null\n";

    // v2.0: one global handle per external stream (runtime id) and one double
    // per signal (host-side cache; the runtime signal is authoritative).
    for (const auto& st : sema_.streams)
      globals_ << "@strm_" << st.name << " = global i64 -1\n";
    for (const auto& sig : sema_.signals)
      globals_ << "@sig_" << sig.name << " = global double 0.000000e+00\n";

    for (const auto& v : sema_.globals) {
      const std::string ty = llvmTypeName(v.type);
      if (ty == "ptr")
        globals_ << "@g_" << v.name << " = global " << ty << " null\n";
      else if (ty == "i8")
        globals_ << "@g_" << v.name << " = global i8 0\n";
      else
        globals_ << "@g_" << v.name << " = global " << ty << " " << zeroInit(v.type)
                 << "\n";
    }
    if (!sema_.groups.empty() || !sema_.globals.empty()) globals_ << "\n";
  }

  std::string internStr(const std::string& s) {
    auto it = strPool_.find(s);
    if (it != strPool_.end()) return it->second;
    const std::string name = "@.str." + std::to_string(strN_++);
    strPool_[s] = name;
    strings_ << name << " = private unnamed_addr constant ["
             << (s.size() + 1) << " x i8] c\"" << llvmEscape(s, false) << "\\00\"\n";
    return name;
  }

  std::string fm(const std::string&) { return opts_.fastmath ? "fast " : ""; }

  // --- expression emission ---------------------------------------------------
  struct Val {
    std::string v;  // SSA register name or constant, e.g. "%t3" / "5" / "1.0e+01"
    TypeInfo type;
  };

  TypeInfo typeOf(const Expr& e) {
    auto it = sema_.exprTypes.find(&e);
    if (it != sema_.exprTypes.end()) return it->second;
    return TypeInfo::Error();
  }

  Val emitExpr(const Expr& e, FnCtx& f) {
    Val r;
    switch (e.kind) {
      case ExprKind::IntLit:
        r.v = std::to_string((long long)e.intVal);
        r.type = TypeInfo::Int();
        return r;
      case ExprKind::FloatLit:
        r.v = fmtDouble(e.floatVal);
        r.type = TypeInfo::Float();
        return r;
      case ExprKind::DurationLit:
        r.v = fmtDouble(e.floatVal);
        r.type = TypeInfo::Duration();
        return r;
      case ExprKind::StringLit:
        r.v = internStr(e.strVal);
        r.type = TypeInfo::Str();
        return r;
      case ExprKind::BoolLit:
        r.v = e.boolVal ? "1" : "0";
        r.type = TypeInfo::Bool();
        return r;
      case ExprKind::DeviceLit:
        r.v = (e.name == "GPU") ? "1" : "0";
        r.type = TypeInfo::Int();
        return r;
      case ExprKind::Ident: {
        Slot* s = f.find(e.name);
        if (!s) {
          fail("unknown identifier in codegen: " + e.name);
          r.v = "null";
          r.type = TypeInfo::Error();
          return r;
        }
        const std::string ty = llvmTypeName(s->type);
        if (ty == "i1") {
          // Bool stored as i8 in memory
          const std::string t = f.tmp();
          f.line(t + " = load i8, ptr " + s->ref);
          const std::string t2 = f.tmp();
          f.line(t2 + " = trunc i8 " + t + " to i1");
          r.v = t2;
        } else {
          const std::string t = f.tmp();
          f.line(t + " = load " + ty + ", ptr " + s->ref);
          r.v = t;
        }
        r.type = s->type;
        return r;
      }
      case ExprKind::Unary:
        return emitUnary(e, f);
      case ExprKind::Binary:
        return emitBinary(e, f);
      case ExprKind::Call:
        return emitCall(e, f);
    }
    fail("unknown expression kind");
    r.v = "null";
    r.type = TypeInfo::Error();
    return r;
  }

  // Converts a value to double (sitofp from Int).
  std::string toDouble(const Val& in, FnCtx& f) {
    if (in.type.k == TypeInfo::K::Int) {
      const std::string t = f.tmp();
      f.line(t + " = sitofp i64 " + in.v + " to double");
      return t;
    }
    return in.v;
  }
  // Converts a value to i64 (fptosi from Float/Duration).
  std::string toI64(const Val& in, FnCtx& f) {
    if (in.type.k == TypeInfo::K::Float || in.type.k == TypeInfo::K::Duration) {
      const std::string t = f.tmp();
      f.line(t + " = fptosi double " + in.v + " to i64");
      return t;
    }
    return in.v;
  }
  // Converts a value to the target type for storage/arithmetic.
  std::string coerceTo(const Val& in, const TypeInfo& target, FnCtx& f) {
    const std::string tt = llvmTypeName(target);
    if (tt == "double" && in.type.k == TypeInfo::K::Int) return toDouble(in, f);
    if (tt == "i64" && (in.type.k == TypeInfo::K::Float || in.type.k == TypeInfo::K::Duration))
      return toI64(in, f);
    return in.v;
  }

  Val emitUnary(const Expr& e, FnCtx& f) {
    Val r;
    Val x = emitExpr(*e.lhs, f);
    const TypeInfo& xt = x.type;
    if (e.op == Tok::Minus) {
      if (xt.k == TypeInfo::K::Int) {
        const std::string t = f.tmp();
        f.line(t + " = sub i64 0, " + x.v);
        r = {t, xt};
      } else {
        const std::string t = f.tmp();
        f.line(t + " = fneg double " + x.v);
        r = {t, xt};
      }
    } else {  // Not
      const std::string t = f.tmp();
      f.line(t + " = xor i1 " + x.v + ", 1");
      r = {t, TypeInfo::Bool()};
    }
    return r;
  }

  // Short-circuit && / || via an i8 result alloca (phi-free).
  Val emitShortCircuit(const Expr& e, FnCtx& f, bool isAnd) {
    Val lhs = emitExpr(*e.lhs, f);
    const std::string result = f.tmp() + ".sc.addr";
    f.allocas.push_back("  " + result + " = alloca i8");
    const std::string rhsBlk = f.freshBlock(isAnd ? "land.rhs" : "lor.rhs");
    const std::string endBlk = f.freshBlock(isAnd ? "land.end" : "lor.end");

    // store zext(lhs)
    const std::string zl = f.tmp();
    f.line(zl + " = zext i1 " + lhs.v + " to i8");
    f.line("store i8 " + zl + ", ptr " + result);
    if (isAnd)
      f.brCond(lhs.v, rhsBlk, endBlk);   // lhs true → eval rhs
    else
      f.brCond(lhs.v, endBlk, rhsBlk);   // lhs true → done

    f.label(rhsBlk);
    Val rhs = emitExpr(*e.rhs, f);
    const std::string zr = f.tmp();
    f.line(zr + " = zext i1 " + rhs.v + " to i8");
    f.line("store i8 " + zr + ", ptr " + result);
    f.br(endBlk);

    f.label(endBlk);
    const std::string ld = f.tmp();
    f.line(ld + " = load i8, ptr " + result);
    const std::string tr = f.tmp();
    f.line(tr + " = trunc i8 " + ld + " to i1");
    return {tr, TypeInfo::Bool()};
  }

  Val emitBinary(const Expr& e, FnCtx& f) {
    if (e.op == Tok::AndAnd) return emitShortCircuit(e, f, true);
    if (e.op == Tok::OrOr) return emitShortCircuit(e, f, false);

    Val lhs = emitExpr(*e.lhs, f);
    Val rhs = emitExpr(*e.rhs, f);
    const TypeInfo rt = typeOf(e);
    Val r;
    r.type = rt;

    const bool floatOps = lhs.type.k == TypeInfo::K::Float ||
                          lhs.type.k == TypeInfo::K::Duration ||
                          rhs.type.k == TypeInfo::K::Float ||
                          rhs.type.k == TypeInfo::K::Duration;

    switch (e.op) {
      case Tok::Plus:
      case Tok::Minus:
      case Tok::Star:
      case Tok::Slash:
      case Tok::Percent: {
        if (lhs.type.k == TypeInfo::K::Str && rhs.type.k == TypeInfo::K::Str) {
          // string + string
          const std::string t = f.tmp();
          f.line(t + " = call ptr @ndl_rt_str_concat(ptr " + lhs.v + ", ptr " + rhs.v + ")");
          r = {t, TypeInfo::Str()};
          return r;
        }
        static const char* iops[] = {"add", "sub", "mul", "sdiv", "srem"};
        static const char* fops[] = {"fadd", "fsub", "fmul", "fdiv", "frem"};
        int idx = e.op == Tok::Plus ? 0 : e.op == Tok::Minus ? 1 : e.op == Tok::Star ? 2
                  : e.op == Tok::Slash ? 3 : 4;
        if (floatOps || rt.k == TypeInfo::K::Float || rt.k == TypeInfo::K::Duration) {
          const std::string a = toDouble(lhs, f);
          const std::string b = toDouble(rhs, f);
          const std::string t = f.tmp();
          f.line(t + " = " + fops[idx] + " " + fm("") + "double " + a + ", " + b);
          r = {t, rt};
        } else {
          const std::string t = f.tmp();
          f.line(t + " = " + iops[idx] + " i64 " + lhs.v + ", " + rhs.v);
          r = {t, rt};
        }
        return r;
      }
      case Tok::EqEq:
      case Tok::NotEq: {
        const bool isEq = e.op == Tok::EqEq;
        if (lhs.type.k == TypeInfo::K::Str && rhs.type.k == TypeInfo::K::Str) {
          const std::string c = f.tmp();
          f.line(c + " = call i32 @strcmp(ptr " + lhs.v + ", ptr " + rhs.v + ")");
          const std::string z = f.tmp();
          f.line(z + (isEq ? " = icmp eq i32 " : " = icmp ne i32 ") + c + ", 0");
          r = {z, TypeInfo::Bool()};
          return r;
        }
        if (lhs.type.k == TypeInfo::K::Bool && rhs.type.k == TypeInfo::K::Bool) {
          const std::string z = f.tmp();
          f.line(z + (isEq ? " = icmp eq i1 " : " = icmp ne i1 ") + lhs.v + ", " + rhs.v);
          r = {z, TypeInfo::Bool()};
          return r;
        }
        if (floatOps) {
          const std::string a = toDouble(lhs, f), b = toDouble(rhs, f);
          const std::string z = f.tmp();
          f.line(z + (isEq ? " = fcmp fast oeq double " : " = fcmp fast one double ") + a + ", " + b);
          r = {z, TypeInfo::Bool()};
        } else {
          const std::string z = f.tmp();
          f.line(z + (isEq ? " = icmp eq i64 " : " = icmp ne i64 ") + lhs.v + ", " + rhs.v);
          r = {z, TypeInfo::Bool()};
        }
        return r;
      }
      case Tok::Lt:
      case Tok::Gt:
      case Tok::Le:
      case Tok::Ge: {
        static const char* iops[] = {"slt", "sgt", "sle", "sge"};
        static const char* fops[] = {"olt", "ogt", "ole", "oge"};
        int idx = e.op == Tok::Lt ? 0 : e.op == Tok::Gt ? 1 : e.op == Tok::Le ? 2 : 3;
        if (floatOps) {
          const std::string a = toDouble(lhs, f), b = toDouble(rhs, f);
          const std::string z = f.tmp();
          f.line(z + " = fcmp fast " + fops[idx] + " double " + a + ", " + b);
          r = {z, TypeInfo::Bool()};
        } else {
          const std::string z = f.tmp();
          f.line(z + " = icmp " + iops[idx] + " i64 " + lhs.v + ", " + rhs.v);
          r = {z, TypeInfo::Bool()};
        }
        return r;
      }
      default:
        fail("unknown binary operator");
        r.v = "null";
        return r;
    }
  }

  Val emitCall(const Expr& e, FnCtx& f) {
    Val r;
    const std::string& n = e.name;

    // v2.0 stream_set: argument 1 is a stream NAME — it must not be evaluated
    // as a value expression, so this call is handled before the generic walk.
    if (n == "stream_set") {
      const Expr& first = *e.args[0].value;
      if (first.kind != ExprKind::Ident || !sema_.streamIndex.count(first.name)) {
        fail("stream_set: argument 1 must be a stream name");
        r.v = "null";
        r.type = TypeInfo::Error();
        return r;
      }
      const std::string idL = f.tmp();
      f.line(idL + " = load i64, ptr @strm_" + first.name);
      Val idx = emitExpr(*e.args[1].value, f);
      Val val = emitExpr(*e.args[2].value, f);
      f.line("call void @ndl_rt_stream_set(i64 " + idL + ", i64 " + toI64(idx, f) +
             ", double " + toDouble(val, f) + ")");
      r = {"null", TypeInfo::Void()};
      return r;
    }

    // Evaluate arguments (builtins are positional).
    std::vector<Val> args;
    for (const auto& a : e.args) args.push_back(emitExpr(*a.value, f));

    auto argD = [&](size_t i) { return toDouble(args[i], f); };
    auto argI = [&](size_t i) { return toI64(args[i], f); };

    std::string call;
    if (n == "sin" || n == "cos" || n == "exp" || n == "log" || n == "sqrt") {
      const std::string t = f.tmp();
      f.line(t + " = call double @ndl_rt_" + n + "(double " + argD(0) + ")");
      r = {t, TypeInfo::Float()};
      return r;
    }
    if (n == "random_uniform" || n == "random_gaussian") {
      const std::string t = f.tmp();
      f.line(t + " = call double @ndl_rt_" + n + "(double " + argD(0) + ", double " +
             argD(1) + ")");
      r = {t, TypeInfo::Float()};
      return r;
    }
    if (n == "get_weight") {
      const std::string t = f.tmp();
      f.line(t + " = call double @ndl_rt_get_weight(ptr " + args[0].v + ", ptr " +
             args[1].v + ", i64 " + argI(2) + ", i64 " + argI(3) + ")");
      r = {t, TypeInfo::Float()};
      return r;
    }
    if (n == "tensor_new" || n == "tensor_new2" || n == "tensor_new3") {
      const int rank = n == "tensor_new" ? 1 : n == "tensor_new2" ? 2 : 3;
      const std::string dims = f.tmp() + ".dims";
      f.allocas.push_back("  " + dims + " = alloca [" + std::to_string(rank) + " x i64]");
      for (int k = 0; k < rank; ++k) {
        const std::string p = f.tmp();
        f.line(p + " = getelementptr inbounds [" + std::to_string(rank) +
               " x i64], ptr " + dims + ", i64 0, i64 " + std::to_string(k));
        f.line("store i64 " + argI(k) + ", ptr " + p);
      }
      const std::string t = f.tmp();
      f.line(t + " = call ptr @ndl_rt_tensor_new(i64 " + std::to_string(rank) +
             ", ptr " + dims + ", double 0.000000e+00)");
      r = {t, TypeInfo::Tensor()};
      return r;
    }
    if (n == "tensor_get" || n == "tensor_get2" || n == "tensor_get3") {
      const std::string t = f.tmp();
      if (n == "tensor_get")
        f.line(t + " = call double @ndl_rt_tensor_get1(ptr " + args[0].v + ", i64 " + argI(1) + ")");
      else if (n == "tensor_get2")
        f.line(t + " = call double @ndl_rt_tensor_get2(ptr " + args[0].v + ", i64 " +
               argI(1) + ", i64 " + argI(2) + ")");
      else
        f.line(t + " = call double @ndl_rt_tensor_get3(ptr " + args[0].v + ", i64 " +
               argI(1) + ", i64 " + argI(2) + ", i64 " + argI(3) + ")");
      r = {t, TypeInfo::Float()};
      return r;
    }
    if (n == "tensor_set" || n == "tensor_set2" || n == "tensor_set3") {
      const size_t last = args.size() - 1;
      const std::string v = toDouble(args[last], f);
      if (n == "tensor_set")
        f.line("call void @ndl_rt_tensor_set1(ptr " + args[0].v + ", i64 " + argI(1) +
               ", double " + v + ")");
      else if (n == "tensor_set2")
        f.line("call void @ndl_rt_tensor_set2(ptr " + args[0].v + ", i64 " + argI(1) +
               ", i64 " + argI(2) + ", double " + v + ")");
      else
        f.line("call void @ndl_rt_tensor_set3(ptr " + args[0].v + ", i64 " + argI(1) +
               ", i64 " + argI(2) + ", i64 " + argI(3) + ", double " + v + ")");
      r = {"null", TypeInfo::Void()};
      return r;
    }
    if (n == "dump_tensor") {
      f.line("call void @ndl_rt_dump_tensor(ptr " + args[0].v + ", ptr " + args[1].v + ")");
      r = {"null", TypeInfo::Void()};
      return r;
    }
    // ---- v2.0 — readout & action decoder (RSOL) ----
    if (n == "get_membrane_potentials") {
      const std::string t = f.tmp();
      f.line(t + " = call ptr @ndl_rt_get_membrane_potentials(ptr " + args[0].v + ")");
      r = {t, TypeInfo::Tensor()};
      return r;
    }
    if (n == "predict_linear") {
      const std::string t = f.tmp();
      f.line(t + " = call ptr @ndl_rt_predict_linear(ptr " + args[0].v + ", ptr " +
             args[1].v + ")");
      r = {t, TypeInfo::Tensor()};
      return r;
    }
    if (n == "vector_l2_norm") {
      const std::string t = f.tmp();
      f.line(t + " = call double @ndl_rt_vector_l2_norm(ptr " + args[0].v + ")");
      r = {t, TypeInfo::Float()};
      return r;
    }
    if (n == "stream_set") {
      // First argument is a stream NAME — resolve to its runtime id global.
      const Expr& first = *e.args[0].value;
      if (first.kind != ExprKind::Ident || !sema_.streamIndex.count(first.name)) {
        fail("stream_set: argument 1 must be a stream name");
        r.v = "null";
        r.type = TypeInfo::Error();
        return r;
      }
      const std::string idL = f.tmp();
      f.line(idL + " = load i64, ptr @strm_" + first.name);
      f.line("call void @ndl_rt_stream_set(i64 " + idL + ", i64 " + argI(1) +
             ", double " + toDouble(args[2], f) + ")");
      r = {"null", TypeInfo::Void()};
      return r;
    }
    fail("unknown call target: " + n);
    r.v = "null";
    r.type = TypeInfo::Error();
    return r;
  }

  // --- statement emission -----------------------------------------------------
  void emitStmt(const Stmt& s, FnCtx& f) {
    switch (s.kind) {
      case StmtKind::Import:
      case StmtKind::Use:
        return;
      case StmtKind::NodeGroup:
        return emitNodeGroup(s, f);
      case StmtKind::Connect:
        return emitConnect(s, f);
      case StmtKind::Stdp:
        return emitStdp(s, f);
      case StmtKind::Let:
        return emitLet(s, f);
      case StmtKind::Assign:
        return emitAssign(s, f);
      case StmtKind::Print:
        return emitPrint(s, f);
      case StmtKind::SaveCheckpoint:
        f.line("call void @ndl_rt_save_checkpoint(ptr " + strVal(s.init.get(), f) + ")");
        return;
      case StmtKind::LoadCheckpoint:
        f.line("call void @ndl_rt_load_checkpoint(ptr " + strVal(s.init.get(), f) + ")");
        return;
      case StmtKind::ExportRaster: {
        Slot* g = f.find(s.srcName);
        if (!g) { fail("export_raster: unknown group " + s.srcName); return; }
        const std::string gL = f.tmp();
        f.line(gL + " = load ptr, ptr " + g->ref);
        f.line("call void @ndl_rt_export_raster(ptr " + gL + ", ptr " +
               strVal(s.init.get(), f) + ")");
        return;
      }
      case StmtKind::If:
        return emitIf(s, f);
      case StmtKind::For:
        return emitFor(s, f);
      case StmtKind::While:
        return emitWhile(s, f);
      case StmtKind::Run:
        return emitRun(s, f);
      case StmtKind::AtEmit:
        return;  // handled by emitRun
      case StmtKind::OnSpike:
        return emitOnSpike(s, f);
      // ---- v2.0 — asynchronous continuous-time environment ----
      case StmtKind::SignalDecl:
        return emitSignalDecl(s, f);
      case StmtKind::OscillatorDecl:
        return emitOscillatorDecl(s, f);
      case StmtKind::ExternalStreamDecl:
        return emitExternalStreamDecl(s, f);
      case StmtKind::BindInputStream:
        return emitBindInputStream(s, f);
      case StmtKind::RunContinuous:
        return emitRunContinuous(s, f);
      case StmtKind::StopContinuous:
        f.line("; stop_continuous");
        f.line("call void @ndl_rt_stop_continuous()");
        return;
      case StmtKind::WaitContinuous:
        f.line("; wait_continuous");
        f.line("call void @ndl_rt_wait_continuous()");
        return;
      case StmtKind::SetPlasticity:
        return emitSetPlasticity(s, f);
      case StmtKind::PruneWeights:
        return emitPruneWeights(s, f);
      case StmtKind::ExprStmt:
        if (s.expr) emitExpr(*s.expr, f);
        return;
    }
  }

  std::string strVal(const Expr* e, FnCtx& f) {
    if (!e) return internStr("");
    Val v = emitExpr(*e, f);
    return v.v;
  }

  std::string groupRef(const std::string& name) {
    for (const auto& g : sema_.groups)
      if (g.name == name) return "@grp_" + name;
    fail("unknown group in codegen: " + name);
    return "null";
  }

  // Finds a named option or (defensively) positional.
  const Expr* opt(const Stmt& s, const char* name) {
    for (const auto& a : s.opts)
      if (a.named && a.name == name) return a.value.get();
    return nullptr;
  }

  void emitNodeGroup(const Stmt& s, FnCtx& f) {
    f.line("; node_group " + s.name);
    // Evaluate params in declaration order and map by name.
    std::unordered_map<std::string, std::string> vals;
    std::unordered_map<std::string, TypeInfo> types;
    for (const auto& p : s.params) {
      Val v = emitExpr(*p.value, f);
      vals[p.name] = toDouble(v, f);
      types[p.name] = v.type;
    }
    static const char* names[] = {"tau", "threshold", "rest", "reset"};
    double defaults[] = {0.0, 0.0, 0.0, 0.0};
    std::string args[4];
    for (int k = 0; k < 4; ++k) {
      auto it = vals.find(names[k]);
      args[k] = it != vals.end() ? it->second : fmtDouble(defaults[k]);
    }
    const std::string sizeExpr = [&] {
      const GroupInfo* gi = nullptr;
      for (const auto& g : sema_.groups)
        if (g.name == s.name) { gi = &g; break; }
      return gi ? std::to_string((long long)gi->size) : "1";
    }();
    const std::string h = f.tmp();
    f.line(h + " = call ptr @ndl_rt_group_create(i64 " + sizeExpr + ", i32 " +
           (s.ntype == NeuronType::Inhibitory ? "1" : "0") + ", double " + args[0] +
           ", double " + args[1] + ", double " + args[2] + ", double " + args[3] +
           ", ptr " + internStr(s.name) + ")");
    f.line("store ptr " + h + ", ptr " + groupRef(s.name));
    // refresh slot (the global-scope slot already points at @grp_<Name>)
  }

  void emitConnect(const Stmt& s, FnCtx& f) {
    f.line("; connect " + s.srcName + " -> " + s.dstName);
    const std::string src = f.find(s.srcName) ? f.find(s.srcName)->ref : groupRef(s.srcName);
    const std::string dst = f.find(s.dstName) ? f.find(s.dstName)->ref : groupRef(s.dstName);
    const std::string srcL = f.tmp();
    f.line(srcL + " = load ptr, ptr " + src);
    const std::string dstL = f.tmp();
    f.line(dstL + " = load ptr, ptr " + dst);

    const Expr* plasticE = opt(s, "plastic");
    std::string plastic = "0";
    if (plasticE) {
      Val v = emitExpr(*plasticE, f);
      const std::string z = f.tmp();
      f.line(z + " = zext i1 " + v.v + " to i32");
      plastic = z;
    }

    if (s.ckind == ConnectKind::Sparse) {
      Val density = emitExpr(*opt(s, "density"), f);
      Val weight = emitExpr(*opt(s, "weight"), f);
      f.line("call void @ndl_rt_sparse_connect(ptr " + srcL + ", ptr " + dstL +
             ", double " + toDouble(density, f) + ", double " + toDouble(weight, f) +
             ", i32 " + plastic + ")");
      return;
    }
    if (s.ckind == ConnectKind::OneToOne) {
      Val weight = emitExpr(*opt(s, "weight"), f);
      f.line("call void @ndl_rt_one_to_one_connect(ptr " + srcL + ", ptr " + dstL +
             ", double " + toDouble(weight, f) + ", i32 " + plastic + ")");
      return;
    }

    // Dense: weight | weight_func
    const Expr* weightE = opt(s, "weight");
    const Expr* wfE = opt(s, "weight_func");
    std::string kind = "0", p0 = "0.000000e+00", p1 = "0.000000e+00";
    if (weightE) {
      Val w = emitExpr(*weightE, f);
      kind = "0";
      p0 = toDouble(w, f);
    } else if (wfE && wfE->kind == ExprKind::Call) {
      // random_gaussian(mu, sigma) → kind 1; random_uniform(a, b) → kind 2
      Val a0 = emitExpr(*wfE->args[0].value, f);
      Val a1 = emitExpr(*wfE->args[1].value, f);
      kind = wfE->name == "random_uniform" ? "2" : "1";
      p0 = toDouble(a0, f);
      p1 = toDouble(a1, f);
    }
    f.line("call void @ndl_rt_dense_connect(ptr " + srcL + ", ptr " + dstL + ", i32 " +
           kind + ", double " + p0 + ", double " + p1 + ", i32 " + plastic + ")");
  }

  void emitStdp(const Stmt& s, FnCtx& f) {
    f.line("; configure_stdp " + s.srcName + " -> " + s.dstName);
    const std::string srcL = f.tmp();
    f.line(srcL + " = load ptr, ptr " + groupRef(s.srcName));
    const std::string dstL = f.tmp();
    f.line(dstL + " = load ptr, ptr " + groupRef(s.dstName));
    Val pot = emitExpr(*opt(s, "lr_pot"), f);
    Val dep = emitExpr(*opt(s, "lr_dep"), f);
    Val win = emitExpr(*opt(s, "window_ms"), f);
    f.line("call void @ndl_rt_configure_stdp(ptr " + srcL + ", ptr " + dstL +
           ", double " + toDouble(pot, f) + ", double " + toDouble(dep, f) +
           ", double " + toDouble(win, f) + ")");
    // v2.0: bind the 3-factor modulator signal when present.
    if (const Expr* modE = opt(s, "modulator")) {
      if (modE->kind == ExprKind::Ident) {
        f.line("call void @ndl_rt_stdp_set_modulator(ptr " + srcL + ", ptr " + dstL +
               ", ptr " + internStr(modE->name) + ")");
      }
    }
  }

  // ---- v2.0 — asynchronous continuous-time environment ----

  void emitSignalDecl(const Stmt& s, FnCtx& f) {
    f.line("; signal " + s.name);
    Val init = emitExpr(*s.init, f);
    const std::string d = toDouble(init, f);
    f.line("call void @ndl_rt_signal_create(ptr " + internStr(s.name) + ", double " +
           d + ")");
    f.line("store double " + d + ", ptr @sig_" + s.name);
  }

  void emitOscillatorDecl(const Stmt& s, FnCtx& f) {
    f.line("; oscillator " + s.name);
    double phaseD = 0.0;
    const Expr* freqE = nullptr;
    const Expr* ampE = nullptr;
    const Expr* phaseE = nullptr;
    const Expr* targetE = nullptr;
    for (const auto& a : s.opts) {
      if (!a.named || !a.value) continue;
      if (a.name == "frequency") freqE = a.value.get();
      else if (a.name == "amplitude") ampE = a.value.get();
      else if (a.name == "phase") phaseE = a.value.get();
      else if (a.name == "target") targetE = a.value.get();
    }
    if (!targetE || targetE->kind != ExprKind::Ident) {
      fail("oscillator without target");
      return;
    }
    const std::string gL = f.tmp();
    f.line(gL + " = load ptr, ptr " + groupRef(targetE->name));
    Val freq = emitExpr(*freqE, f);
    Val amp = emitExpr(*ampE, f);
    std::string phase = fmtDouble(phaseD);
    if (phaseE) phase = toDouble(emitExpr(*phaseE, f), f);
    f.line("call void @ndl_rt_oscillator_add(ptr " + internStr(s.name) + ", ptr " + gL +
           ", double " + toDouble(freq, f) + ", double " + toDouble(amp, f) +
           ", double " + phase + ")");
  }

  void emitExternalStreamDecl(const Stmt& s, FnCtx& f) {
    f.line("; external stream " + s.name);
    // Size: const-evaluated by sema (StreamInfo.size).
    uint64_t size = 1;
    for (const auto& st : sema_.streams)
      if (st.name == s.name) { size = st.size; break; }
    const std::string id = f.tmp();
    f.line(id + " = call i64 @ndl_rt_stream_create(ptr " + internStr(s.name) +
           ", i64 " + std::to_string((long long)size) + ")");
    f.line("store i64 " + id + ", ptr @strm_" + s.name);
  }

  void emitBindInputStream(const Stmt& s, FnCtx& f) {
    f.line("; bind_input_stream " + s.srcName + " -> " + s.dstName);
    const std::string idL = f.tmp();
    f.line(idL + " = load i64, ptr @strm_" + s.srcName);
    const std::string gL = f.tmp();
    f.line(gL + " = load ptr, ptr " + groupRef(s.dstName));
    std::string maxFreq = fmtDouble(100.0);
    std::string kick = fmtDouble(1.0);
    for (const auto& a : s.opts) {
      if (!a.named || !a.value) continue;
      if (a.name == "max_freq")
        maxFreq = toDouble(emitExpr(*a.value, f), f);
      else if (a.name == "kick")
        kick = toDouble(emitExpr(*a.value, f), f);
    }
    // encoding: 0 = Poisson (sema rejects anything else)
    f.line("call void @ndl_rt_bind_input_stream(i64 " + idL + ", ptr " + gL +
           ", i32 0, double " + maxFreq + ", double " + kick + ")");
  }

  void emitRunContinuous(const Stmt& s, FnCtx& f) {
    f.line("; run_continuous");
    f.pushScope();
    for (const auto& st : s.body) emitStmt(*st, f);
    f.popScope();
    std::string dtD = fmtDouble(1.0);
    if (s.hasDt && s.dtExpr) dtD = toDouble(emitExpr(*s.dtExpr, f), f);
    const std::string dev = s.hasDevice ? (s.device == DeviceKind::GPU ? "1" : "0") : "0";
    f.line("call void @ndl_rt_run_continuous(double " + dtD + ", i32 " + dev + ")");
    sawRunContinuous_ = true;
  }

  void emitSetPlasticity(const Stmt& s, FnCtx& f) {
    f.line("; set_plasticity " + s.srcName + " -> " + s.dstName);
    const std::string srcL = f.tmp();
    f.line(srcL + " = load ptr, ptr " + groupRef(s.srcName));
    const std::string dstL = f.tmp();
    f.line(dstL + " = load ptr, ptr " + groupRef(s.dstName));
    Val en = emitExpr(*opt(s, "enabled"), f);
    const std::string z = f.tmp();
    f.line(z + " = zext i1 " + en.v + " to i32");
    f.line("call void @ndl_rt_set_plasticity(ptr " + srcL + ", ptr " + dstL +
           ", i32 " + z + ")");
  }

  void emitPruneWeights(const Stmt& s, FnCtx& f) {
    f.line("; prune_weights " + s.srcName + " -> " + s.dstName);
    const std::string srcL = f.tmp();
    f.line(srcL + " = load ptr, ptr " + groupRef(s.srcName));
    const std::string dstL = f.tmp();
    f.line(dstL + " = load ptr, ptr " + groupRef(s.dstName));
    Val thr = emitExpr(*opt(s, "threshold"), f);
    f.line("call void @ndl_rt_prune_weights(ptr " + srcL + ", ptr " + dstL +
           ", double " + toDouble(thr, f) + ")");
  }

  void emitLet(const Stmt& s, FnCtx& f) {
    f.line("; let " + s.name);
    // Determine the variable's type from sema.
    TypeInfo vt;
    auto it = sema_.varTypes.find(&s);
    if (it != sema_.varTypes.end()) vt = it->second;

    Val init = emitExpr(*s.init, f);
    const std::string st = llvmTypeName(vt);
    const std::string value = coerceTo(init, vt, f);

    // Top-level lets (direct module children) map to @g_<name>. We detect it
    // by checking whether a global with this name was declared for this stmt.
    bool globalSlot = false;
    for (const auto& gv : sema_.globals)
      if (gv.name == s.name && gv.decl == &s) { globalSlot = true; break; }

    if (globalSlot) {
      const std::string ref = "@g_" + s.name;
      if (st == "i1") {
        const std::string z = f.tmp();
        f.line(z + " = zext i1 " + value + " to i8");
        f.line("store i8 " + z + ", ptr " + ref);
      } else {
        f.line("store " + st + " " + value + ", ptr " + ref);
      }
      Slot* existingSlot = f.find(s.name);
      if (!existingSlot || existingSlot->ref != ref) f.declare(s.name, Slot{ref, true, vt});
      return;
    }

    // Block-scoped: fresh alloca (hoisted to entry).
    const std::string addr = "%" + s.name + "." + std::to_string(f.tmpN + 1000) + ".addr";
    f.tmpN++;
    f.allocas.push_back("  " + addr + " = alloca " + (st == "i1" ? "i8" : st));
    if (st == "i1") {
      const std::string z = f.tmp();
      f.line(z + " = zext i1 " + value + " to i8");
      f.line("store i8 " + z + ", ptr " + addr);
    } else {
      f.line("store " + st + " " + value + ", ptr " + addr);
    }
    f.declare(s.name, Slot{addr, false, vt});
  }

  void emitAssign(const Stmt& s, FnCtx& f) {
    f.line("; assign " + s.name);
    Slot* slot = f.find(s.name);
    if (!slot) { fail("assign to unknown variable: " + s.name); return; }
    Val v = emitExpr(*s.init, f);
    const std::string st = llvmTypeName(slot->type);
    const std::string value = coerceTo(v, slot->type, f);
    if (st == "i1" || (slot->type.k == TypeInfo::K::Bool)) {
      const std::string z = f.tmp();
      f.line(z + " = zext i1 " + value + " to i8");
      f.line("store i8 " + z + ", ptr " + slot->ref);
    } else {
      f.line("store " + st + " " + value + ", ptr " + slot->ref);
    }
    // v2.0: signal assignment also updates the runtime neuromodulator M(t).
    if (sema_.signalIndex.count(s.name)) {
      const std::string d = toDouble(v, f);
      f.line("call void @ndl_rt_signal_set(ptr " + internStr(s.name) + ", double " +
             d + ")");
    }
  }

  void emitPrint(const Stmt& s, FnCtx& f) {
    f.line("; print");
    f.line("call void @ndl_rt_print_begin()");
    for (const auto& e : s.printArgs) {
      Val v = emitExpr(*e, f);
      switch (v.type.k) {
        case TypeInfo::K::Int:
          f.line("call void @ndl_rt_print_i64(i64 " + v.v + ")");
          break;
        case TypeInfo::K::Float:
        case TypeInfo::K::Duration:
          f.line("call void @ndl_rt_print_f64(double " + v.v + ")");
          break;
        case TypeInfo::K::Bool: {
          const std::string z = f.tmp();
          f.line(z + " = zext i1 " + v.v + " to i32");
          f.line("call void @ndl_rt_print_bool(i32 " + z + ")");
          break;
        }
        case TypeInfo::K::Str:
          f.line("call void @ndl_rt_print_str(ptr " + v.v + ")");
          break;
        case TypeInfo::K::Group: {
          // print the constant "<group Name>" (name is static)
          std::string gname;
          if (e->kind == ExprKind::Ident) gname = e->name;
          f.line("call void @ndl_rt_print_str(ptr " + internStr("<group " + gname + ">") + ")");
          break;
        }
        default:
          f.line("call void @ndl_rt_print_str(ptr " + internStr("<tensor>") + ")");
          break;
      }
    }
    f.line("call void @ndl_rt_print_end()");
  }

  void emitIf(const Stmt& s, FnCtx& f) {
    f.line("; if");
    Val c = emitExpr(*s.cond, f);
    const std::string thenB = f.freshBlock("if.then");
    const std::string endB = f.freshBlock("if.end");
    const std::string elseB = f.freshBlock("if.else");
    if (!s.elseBody.empty())
      f.brCond(c.v, thenB, elseB);
    else
      f.brCond(c.v, thenB, endB);

    f.label(thenB);
    f.pushScope();
    for (const auto& st : s.body) emitStmt(*st, f);
    f.popScope();
    f.br(endB);

    if (!s.elseBody.empty()) {
      f.label(elseB);
      f.pushScope();
      for (const auto& st : s.elseBody) emitStmt(*st, f);
      f.popScope();
      f.br(endB);
    }
    f.label(endB);
  }

  void emitFor(const Stmt& s, FnCtx& f) {
    f.line("; for " + s.loopVar);
    Val lo = emitExpr(*s.lo, f);
    Val hi = emitExpr(*s.hi, f);
    const std::string loI = toI64(lo, f);
    const std::string hiI = toI64(hi, f);

    const std::string addr = "%" + s.loopVar + "." + std::to_string(f.tmpN + 1000) + ".addr";
    f.tmpN++;
    f.allocas.push_back("  " + addr + " = alloca i64");
    f.line("store i64 " + loI + ", ptr " + addr);

    const std::string condB = f.freshBlock("for.cond");
    const std::string bodyB = f.freshBlock("for.body");
    const std::string incB = f.freshBlock("for.inc");
    const std::string endB = f.freshBlock("for.end");
    f.br(condB);

    f.label(condB);
    const std::string i1 = f.tmp();
    f.line(i1 + " = load i64, ptr " + addr);
    const std::string c = f.tmp();
    f.line(c + " = icmp slt i64 " + i1 + ", " + hiI);
    f.brCond(c, bodyB, endB);

    f.label(bodyB);
    f.pushScope();
    f.declare(s.loopVar, Slot{addr, false, TypeInfo::Int()});
    for (const auto& st : s.body) emitStmt(*st, f);
    f.popScope();
    f.br(incB);

    f.label(incB);
    const std::string i2 = f.tmp();
    f.line(i2 + " = load i64, ptr " + addr);
    const std::string i3 = f.tmp();
    f.line(i3 + " = add i64 " + i2 + ", 1");
    f.line("store i64 " + i3 + ", ptr " + addr);
    f.br(condB);

    f.label(endB);
  }

  void emitWhile(const Stmt& s, FnCtx& f) {
    f.line("; while");
    const std::string condB = f.freshBlock("while.cond");
    const std::string bodyB = f.freshBlock("while.body");
    const std::string endB = f.freshBlock("while.end");
    f.br(condB);
    f.label(condB);
    Val c = emitExpr(*s.cond, f);
    f.brCond(c.v, bodyB, endB);
    f.label(bodyB);
    f.pushScope();
    for (const auto& st : s.body) emitStmt(*st, f);
    f.popScope();
    f.br(condB);
    f.label(endB);
  }

  void emitRun(const Stmt& s, FnCtx& f) {
    f.line("; run");
    // Schedule all at-emit statements first (source order).
    for (const auto& item : s.body) {
      if (item->kind != StmtKind::AtEmit) continue;
      const Stmt& at = *item;
      f.line("; at-emit " + at.name);
      const std::string gL = f.tmp();
      f.line(gL + " = load ptr, ptr " + groupRef(at.name));

      // Bounds
      std::string lo, hi;
      const GroupInfo* gi = nullptr;
      for (const auto& g : sema_.groups)
        if (g.name == at.name) { gi = &g; break; }
      const int64_t gsize = gi ? (int64_t)gi->size : 1;
      if (!at.sliceHasIndex && !at.sliceHasRange) {
        lo = "0";
        hi = std::to_string((long long)gsize);
      } else if (at.sliceHasIndex && !at.sliceHasRange) {
        Val i = emitExpr(*at.sliceLo, f);
        lo = toI64(i, f);
        const std::string one = f.tmp();
        f.line(one + " = add i64 " + lo + ", 1");
        hi = one;
      } else {
        Val a = emitExpr(*at.sliceLo, f);
        Val b = emitExpr(*at.sliceHi, f);
        lo = toI64(a, f);
        hi = toI64(b, f);
      }

      Val t = emitExpr(*at.timeExpr, f);
      Val cur = emitExpr(*at.currentExpr, f);
      f.line("call void @ndl_rt_schedule_inject(ptr " + gL + ", i64 " + lo +
             ", i64 " + hi + ", double " + toDouble(t, f) + ", double " +
             toDouble(cur, f) + ")");
    }

    // duration
    Val dur = emitExpr(*s.durationExpr, f);
    const std::string durD = toDouble(dur, f);
    std::string dtD = fmtDouble(1.0);
    if (s.hasDt && s.dtExpr) {
      Val dt = emitExpr(*s.dtExpr, f);
      dtD = toDouble(dt, f);
    }
    const std::string dev = s.hasDevice ? (s.device == DeviceKind::GPU ? "1" : "0") : "0";
    f.line("call void @ndl_rt_run(double " + durD + ", double " + dtD + ", i32 " + dev + ")");
  }

  void emitOnSpike(const Stmt& s, FnCtx& f) {
    // 1. Register in main at declaration point.
    f.line("; on_spike " + s.name);
    const std::string gL = f.tmp();
    f.line(gL + " = load ptr, ptr " + groupRef(s.name));
    const std::string fname = "@ndl_on_spike." + std::to_string(handlerN_);
    f.line("call void @ndl_rt_register_spike_handler(ptr " + gL + ", ptr " + fname +
           ", ptr null)");

    // 2. Emit the handler function.
    FnCtx h;
    h.name = fname;
    h.pushScope();
    // global scope for handlers: groups + globals + signals
    for (const auto& g : sema_.groups)
      h.declare(g.name, Slot{"@grp_" + g.name, true, TypeInfo::Group()});
    for (const auto& v : sema_.globals)
      h.declare(v.name, Slot{"@g_" + v.name, true, v.type});
    for (const auto& sig : sema_.signals)
      h.declare(sig.name, Slot{"@sig_" + sig.name, true, TypeInfo::Float()});

    std::ostringstream hdr;
    hdr << "define internal void " << fname
        << "(i64 %neuron_param, double %time_param, ptr %user_param) {\n";
    // entry allocas for the two builtins
    h.allocas.push_back("  %neuron_index.addr = alloca i64");
    h.allocas.push_back("  %spike_time.addr = alloca double");
    h.lines.push_back("  store i64 %neuron_param, ptr %neuron_index.addr");
    h.lines.push_back("  store double %time_param, ptr %spike_time.addr");
    h.terminated = false;
    h.declare("neuron_index", Slot{"%neuron_index.addr", false, TypeInfo::Int()});
    h.declare("spike_time", Slot{"%spike_time.addr", false, TypeInfo::Float()});

    h.pushScope();
    for (const auto& st : s.body) emitStmt(*st, h);
    h.popScope();
    h.retVoid();
    h.popScope();

    // Assemble handler text.
    std::ostringstream ss;
    ss << hdr.str();
    for (const auto& a : h.allocas) ss << a << "\n";
    for (const auto& l : h.lines) ss << l << "\n";
    ss << "}\n\n";
    handlerFns_.push_back(ss.str());
    ++handlerN_;
  }

  // --- assembly ---------------------------------------------------------------
  void assemble() {
    out_ << globals_.str();
    out_ << strings_.str();

    if (!opts_.ptxSource.empty()) {
      out_ << "@ndl_ptx_source = private unnamed_addr constant ["
           << (opts_.ptxSource.size() + 1) << " x i8] c\""
           << llvmEscape(opts_.ptxSource, true) << "\\00\"\n\n";
    }

    for (const auto& hf : handlerFns_) out_ << hf;

    // ndl_user_main
    out_ << "define void @ndl_user_main() {\n";
    out_ << "entry:\n";
    for (const auto& a : main_.allocas) out_ << a << "\n";
    for (const auto& l : main_.lines) out_ << l << "\n";
    out_ << "}\n\n";

    out_ << "define i32 @main() {\n"
         << "entry:\n"
         << "  call void @ndl_user_main()\n"
         << "  ret i32 0\n"
         << "}\n";
  }
};

}  // namespace

LLVMCodeGen::LLVMCodeGen(const Program& program, const SemaResult& sema,
                         const LLVMEmitOptions& opts, DiagnosticEngine& diag)
    : prog_(program), sema_(sema), opts_(opts), diag_(diag) {}

std::string LLVMCodeGen::generate() {
  IRGen g(prog_, sema_, opts_, diag_);
  return g.generate();
}

}  // namespace ndl
