// NDL v1.0 — Tree-walking VM (src/vm.cpp)
//
// Interprets the sema-checked AST. Value model, scoping rules and the order
// of libndl_rt calls follow the VM contract (INTERNALS §9) and the type /
// statement semantics (§4/§5), mirroring the LLVM codegen path (§7)
// call-for-call: every heavy operation — simulation, connectivity, STDP,
// checkpoints, raster export, RNG, math — goes through the frozen C ABI in
// runtime/ndl_rt.h, so VM execution is semantically identical to a natively
// compiled binary.
//
// The AST arrives fully type-checked (sema runs before execution), therefore
// the ndl_rt_panic("type error: ...") guards below are defensive only.

#include "ndl/vm.hpp"

#include "ndl_rt.h"

#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ndl {

// Forward declaration: the C trampoline is registered with libndl_rt and
// defined after the interpreter (it needs the complete Interp type).
extern "C" void ndl_vm_spike_trampoline(int64_t neuron, double t_ms, void* user);

namespace {

// ---------------------------------------------------------------------------
// Value model (§9): Void | Int | Float | Bool | String | Duration | Group | Tensor
// ---------------------------------------------------------------------------
struct VoidV {};                        // unit value (tensor_set, ExprStmt result)
struct DurationV { double ms = 0.0; };  // duration, milliseconds
struct GroupV { NdlGroup* p = nullptr; };
struct TensorV { NdlTensor* p = nullptr; };

using Value = std::variant<VoidV, int64_t, double, bool, std::string,
                           DurationV, GroupV, TensorV>;

// A scope maps names to values; the interpreter keeps a stack of scopes
// with scopes_[0] being the global one.
using Scope = std::map<std::string, Value>;

// Index map: 0=VoidV 1=int64_t 2=double 3=bool 4=string 5=DurationV 6=GroupV 7=TensorV
const char* kindName(const Value& v) {
  switch (v.index()) {
    case 1: return "int";
    case 2: return "float";
    case 3: return "bool";
    case 4: return "string";
    case 5: return "duration";
    case 6: return "group";
    case 7: return "tensor";
    default: return "void";
  }
}

[[noreturn]] void typePanic(const std::string& what) {
  ndl_rt_panic(("type error: " + what).c_str());
  std::abort(); // not reached: ndl_rt_panic exits(70); placates -Wreturn-type
}

bool isInt(const Value& v)   { return std::holds_alternative<int64_t>(v); }
bool isFloat(const Value& v) { return std::holds_alternative<double>(v); }
bool isNum(const Value& v)   { return isInt(v) || isFloat(v); }
bool isDur(const Value& v)   { return std::holds_alternative<DurationV>(v); }
bool isStr(const Value& v)   { return std::holds_alternative<std::string>(v); }

double numAsDouble(const Value& v) {
  if (const int64_t* i = std::get_if<int64_t>(&v)) return static_cast<double>(*i);
  return std::get<double>(v);
}

// Guarantees a sub-expression exists (defensive against contract violations).
const Expr& needExpr(const ExprPtr& p, const char* what) {
  if (!p) ndl_rt_panic((std::string("internal error: missing ") + what).c_str());
  return *p;
}

// ---------------------------------------------------------------------------
// Spike-handler plumbing
// ---------------------------------------------------------------------------
struct Interp; // fwd

// Registered with ndl_rt_register_spike_handler as the user pointer; the
// trampoline unwraps it and dispatches to the right handler slot.
struct HandlerCtx {
  Interp* interp = nullptr;
  size_t slot = 0;
};

// One OnSpike statement, in registration order.
struct HandlerDesc {
  const Stmt* stmt = nullptr;
};

// ---------------------------------------------------------------------------
// Interpreter state + execution
// ---------------------------------------------------------------------------
struct Interp {
  const Program& prog_;
  DiagnosticEngine& diag_;
  VMOptions opts_;

  std::vector<Scope> scopes_;      // scopes_[0] = global; back() = innermost
  std::vector<HandlerDesc> handlers_;
  std::vector<std::unique_ptr<HandlerCtx>> handlerCtxs_; // keeps user ptrs alive
  std::map<std::string, int64_t> streams_;  // v2.0: stream name -> runtime id
  std::set<std::string> signals_;           // v2.0: declared signal names
  bool ranContinuous_ = false;              // v2.0: stop the loop at exit

  Interp(const Program& prog, DiagnosticEngine& diag, VMOptions opts)
      : prog_(prog), diag_(diag), opts_(std::move(opts)) {}

  // --- entry (§9 start) ------------------------------------------------------
  int run() {
    ndl_rt_set_seed(opts_.seed);
    if (!opts_.ptxSource.empty()) ndl_rt_gpu_load_ptx(opts_.ptxSource.c_str());

    scopes_.emplace_back(); // global scope
    // Modules in import order (main module last), items in source order —
    // same execution order as the LLVM path.
    for (const std::unique_ptr<Module>& mod : prog_.modules)
      for (const StmtPtr& item : mod->items)
        execStmt(*item);
    // v2.0: a program that entered continuous mode must stop the event loop
    // before the interpreter tears down the AST (handlers run on the loop
    // thread and reference the AST).
    if (ranContinuous_) ndl_rt_stop_continuous();
    return 0;
  }

  // --- scopes ----------------------------------------------------------------
  Value* findVar(const std::string& name) {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      auto f = it->find(name);
      if (f != it->end()) return &f->second;
    }
    return nullptr;
  }

  void declare(const std::string& name, Value v) {
    scopes_.back()[name] = std::move(v); // let binds in the CURRENT scope
  }

  void assign(const std::string& name, Value v) {
    Value* slot = findVar(name); // innermost binding wins
    if (!slot) typePanic("assignment to undefined variable '" + name + "'");
    *slot = std::move(v);
  }

  void execBlock(const std::vector<StmtPtr>& body) {
    for (const StmtPtr& st : body)
      if (st) execStmt(*st);
  }

  // A block body runs in a fresh scope (fresh `let` inside must not clobber
  // outer bindings — matches the distinct allocas of the LLVM path).
  void execBlockScoped(const std::vector<StmtPtr>& body) {
    scopes_.emplace_back();
    execBlock(body);
    scopes_.pop_back();
  }

  // --- expressions -------------------------------------------------------------
  Value evalExpr(const Expr& e) {
    switch (e.kind) {
      case ExprKind::IntLit:     return Value{static_cast<int64_t>(e.intVal)};
      case ExprKind::FloatLit:   return Value{e.floatVal};
      case ExprKind::DurationLit:return Value{DurationV{e.floatVal}};
      case ExprKind::StringLit:  return Value{e.strVal};
      case ExprKind::BoolLit:    return Value{e.boolVal};
      case ExprKind::DeviceLit:
        typePanic("device literal ('CPU'/'GPU') is not a value");
      case ExprKind::Ident: {
        const Value* v = findVar(e.name);
        if (!v) typePanic("undefined variable '" + e.name + "'");
        return *v;
      }
      case ExprKind::Unary:  return evalUnary(e);
      case ExprKind::Binary: return evalBinary(e);
      case ExprKind::Call:   return evalCall(e);
    }
    typePanic("unknown expression kind");
  }

  Value evalUnary(const Expr& e) {
    Value v = evalExpr(needExpr(e.lhs, "unary operand"));
    if (e.op == Tok::Not) {
      const bool* b = std::get_if<bool>(&v);
      if (!b) typePanic(std::string("operand of '!' must be bool, got ") + kindName(v));
      return Value{!*b};
    }
    if (e.op == Tok::Minus) {
      if (const int64_t* i = std::get_if<int64_t>(&v)) return Value{-*i};
      if (const double* f = std::get_if<double>(&v))   return Value{-*f};
      if (const DurationV* d = std::get_if<DurationV>(&v))
        return Value{DurationV{-d->ms}};
      typePanic(std::string("operand of unary '-' must be int/float/duration, got ") +
                kindName(v));
    }
    typePanic("unknown unary operator");
  }

  Value evalBinary(const Expr& e) {
    const Tok op = e.op;

    // Logical operators: strict bool, short-circuit.
    if (op == Tok::AndAnd || op == Tok::OrOr) {
      Value lv = evalExpr(needExpr(e.lhs, "operand"));
      const bool* lb = std::get_if<bool>(&lv);
      if (!lb)
        typePanic(std::string("'&&'/'||' requires bool operands, got ") + kindName(lv));
      if (op == Tok::AndAnd && !*lb) return Value{false};
      if (op == Tok::OrOr && *lb)    return Value{true};
      Value rv = evalExpr(needExpr(e.rhs, "operand"));
      const bool* rb = std::get_if<bool>(&rv);
      if (!rb)
        typePanic(std::string("'&&'/'||' requires bool operands, got ") + kindName(rv));
      return Value{*rb};
    }

    Value l = evalExpr(needExpr(e.lhs, "operand"));
    Value r = evalExpr(needExpr(e.rhs, "operand"));

    switch (op) {
      case Tok::Plus: {
        if (isDur(l) && isDur(r))   // D + D -> D
          return Value{DurationV{std::get<DurationV>(l).ms +
                                 std::get<DurationV>(r).ms}};
        if (isNum(l) && isNum(r)) { // int/int -> i64; float participation -> f64
          if (isInt(l) && isInt(r))
            return Value{std::get<int64_t>(l) + std::get<int64_t>(r)};
          return Value{numAsDouble(l) + numAsDouble(r)};
        }
        if (isStr(l) && isStr(r)) { // string + string -> concat (via libndl_rt)
          const char* s = ndl_rt_str_concat(std::get<std::string>(l).c_str(),
                                            std::get<std::string>(r).c_str());
          return Value{std::string(s)};
        }
        typePanic(std::string("unsupported operand types for '+' (") +
                  kindName(l) + ", " + kindName(r) + ")");
      }
      case Tok::Minus: {
        if (isDur(l) && isDur(r))   // D - D -> D
          return Value{DurationV{std::get<DurationV>(l).ms -
                                 std::get<DurationV>(r).ms}};
        if (isNum(l) && isNum(r)) {
          if (isInt(l) && isInt(r))
            return Value{std::get<int64_t>(l) - std::get<int64_t>(r)};
          return Value{numAsDouble(l) - numAsDouble(r)};
        }
        typePanic(std::string("unsupported operand types for '-' (") +
                  kindName(l) + ", " + kindName(r) + ")");
      }
      case Tok::Star: {
        if (isDur(l) && isNum(r))   // D * num -> D
          return Value{DurationV{std::get<DurationV>(l).ms * numAsDouble(r)}};
        if (isNum(l) && isDur(r))   // num * D -> D
          return Value{DurationV{numAsDouble(l) * std::get<DurationV>(r).ms}};
        if (isNum(l) && isNum(r)) {
          if (isInt(l) && isInt(r))
            return Value{std::get<int64_t>(l) * std::get<int64_t>(r)};
          return Value{numAsDouble(l) * numAsDouble(r)};
        }
        typePanic(std::string("unsupported operand types for '*' (") +
                  kindName(l) + ", " + kindName(r) + ")");
      }
      case Tok::Slash: {
        if (isDur(l) && isDur(r)) { // D / D -> Float
          double denom = std::get<DurationV>(r).ms;
          if (denom == 0.0) ndl_rt_panic("duration division by zero");
          return Value{std::get<DurationV>(l).ms / denom};
        }
        if (isDur(l) && isNum(r))   // D / num -> D
          return Value{DurationV{std::get<DurationV>(l).ms / numAsDouble(r)}};
        if (isNum(l) && isNum(r)) {
          if (isInt(l) && isInt(r)) { // truncating division, % as in C++
            int64_t denom = std::get<int64_t>(r);
            if (denom == 0) ndl_rt_panic("integer division by zero");
            return Value{std::get<int64_t>(l) / denom};
          }
          return Value{numAsDouble(l) / numAsDouble(r)};
        }
        typePanic(std::string("unsupported operand types for '/' (") +
                  kindName(l) + ", " + kindName(r) + ")");
      }
      case Tok::Percent: {
        if (isInt(l) && isInt(r)) {
          int64_t denom = std::get<int64_t>(r);
          if (denom == 0) ndl_rt_panic("integer modulo by zero");
          return Value{std::get<int64_t>(l) % denom};
        }
        typePanic(std::string("'%' requires int operands (") +
                  kindName(l) + ", " + kindName(r) + ")");
      }
      case Tok::EqEq:
      case Tok::NotEq: {
        bool eq = false;
        if (isNum(l) && isNum(r)) {
          if (isInt(l) && isInt(r))
            eq = std::get<int64_t>(l) == std::get<int64_t>(r);
          else
            eq = numAsDouble(l) == numAsDouble(r);
        } else if (isDur(l) && isDur(r)) {
          eq = std::get<DurationV>(l).ms == std::get<DurationV>(r).ms;
        } else if (std::holds_alternative<bool>(l) &&
                   std::holds_alternative<bool>(r)) {
          eq = std::get<bool>(l) == std::get<bool>(r);
        } else if (isStr(l) && isStr(r)) {
          eq = std::get<std::string>(l) == std::get<std::string>(r);
        } else {
          typePanic(std::string("unsupported operand types for '=='/'!=' (") +
                    kindName(l) + ", " + kindName(r) + ")");
        }
        return Value{op == Tok::EqEq ? eq : !eq};
      }
      case Tok::Lt:
      case Tok::Gt:
      case Tok::Le:
      case Tok::Ge: {
        auto cmp = [&](auto a, auto b) -> bool {
          switch (op) {
            case Tok::Lt: return a < b;
            case Tok::Gt: return a > b;
            case Tok::Le: return a <= b;
            default:      return a >= b; // Ge
          }
        };
        if (isNum(l) && isNum(r)) { // numeric comparison, Int/Float mix allowed
          if (isInt(l) && isInt(r))
            return Value{cmp(std::get<int64_t>(l), std::get<int64_t>(r))};
          return Value{cmp(numAsDouble(l), numAsDouble(r))};
        }
        if (isDur(l) && isDur(r))   // Duration vs Duration compares ms
          return Value{cmp(std::get<DurationV>(l).ms, std::get<DurationV>(r).ms)};
        typePanic(std::string("relational comparison requires two numbers or "
                              "two durations (") +
                  kindName(l) + ", " + kindName(r) + ")");
      }
      default:
        typePanic("unsupported binary operator");
    }
  }

  // --- builtin calls -----------------------------------------------------------
  NdlGroup* evalGroupExpr(const Expr& e) {
    Value v = evalExpr(e);
    const GroupV* g = std::get_if<GroupV>(&v);
    if (!g) typePanic(std::string("expected a group, got ") + kindName(v));
    return g->p;
  }

  NdlTensor* evalTensorExpr(const Expr& e) {
    Value v = evalExpr(e);
    const TensorV* t = std::get_if<TensorV>(&v);
    if (!t) typePanic(std::string("expected a tensor, got ") + kindName(v));
    return t->p;
  }

  Value evalCall(const Expr& e) {
    const std::string& n = e.name;
    const size_t arity = e.args.size();
    auto arg = [&](size_t i) -> const Expr& {
      if (i >= arity) typePanic("call '" + n + "': missing argument");
      return needExpr(e.args[i].value, "call argument");
    };

    // Math: 1 numeric argument, result Float.
    if (n == "sin" || n == "cos" || n == "exp" || n == "log" || n == "sqrt") {
      if (arity != 1) typePanic("call '" + n + "' expects 1 argument, got " +
                                std::to_string(arity));
      double x = evalDouble(arg(0));
      if (n == "sin")  return Value{ndl_rt_sin(x)};
      if (n == "cos")  return Value{ndl_rt_cos(x)};
      if (n == "exp")  return Value{ndl_rt_exp(x)};
      if (n == "log")  return Value{ndl_rt_log(x)};
      return Value{ndl_rt_sqrt(x)};
    }

    // RNG: 2 numeric arguments, result Float.
    if (n == "random_uniform" || n == "random_gaussian") {
      if (arity != 2) typePanic("call '" + n + "' expects 2 arguments, got " +
                                std::to_string(arity));
      double a = evalDouble(arg(0));
      double b = evalDouble(arg(1));
      return Value{n == "random_uniform" ? ndl_rt_random_uniform(a, b)
                                         : ndl_rt_random_gaussian(a, b)};
    }

    // Introspection: get_weight(src: Group, dst: Group, i: Int, j: Int) -> Float
    if (n == "get_weight") {
      if (arity != 4) typePanic("call 'get_weight' expects 4 arguments, got " +
                                std::to_string(arity));
      NdlGroup* src = evalGroupExpr(arg(0));
      NdlGroup* dst = evalGroupExpr(arg(1));
      int64_t i = evalInt(arg(2));
      int64_t j = evalInt(arg(3));
      return Value{ndl_rt_get_weight(src, dst, i, j)};
    }

    // Tensors: tensor_new(d0) / tensor_new2(d0,d1) / tensor_new3(d0,d1,d2)
    if (n == "tensor_new" || n == "tensor_new2" || n == "tensor_new3") {
      const size_t rank = (n == "tensor_new") ? 1 : (n == "tensor_new2") ? 2 : 3;
      if (arity != rank)
        typePanic("call '" + n + "' expects " + std::to_string(rank) +
                  " argument(s), got " + std::to_string(arity));
      int64_t dims[3] = {1, 1, 1};
      for (size_t d = 0; d < rank; ++d) dims[d] = evalInt(arg(d));
      return Value{TensorV{ndl_rt_tensor_new(static_cast<int64_t>(rank), dims,
                                             0.0)}};
    }

    // tensor_get(t,i) / tensor_get2(t,i,j) / tensor_get3(t,i,j,k) -> Float
    if (n == "tensor_get" || n == "tensor_get2" || n == "tensor_get3") {
      const size_t want = (n == "tensor_get") ? 2 : (n == "tensor_get2") ? 3 : 4;
      if (arity != want)
        typePanic("call '" + n + "' expects " + std::to_string(want) +
                  " arguments, got " + std::to_string(arity));
      NdlTensor* t = evalTensorExpr(arg(0));
      int64_t i0 = evalInt(arg(1));
      if (want == 2) return Value{ndl_rt_tensor_get1(t, i0)};
      int64_t i1 = evalInt(arg(2));
      if (want == 3) return Value{ndl_rt_tensor_get2(t, i0, i1)};
      int64_t i2 = evalInt(arg(3));
      return Value{ndl_rt_tensor_get3(t, i0, i1, i2)};
    }

    // tensor_set(t,i,v) / tensor_set2(t,i,j,v) / tensor_set3(t,i,j,k,v) -> Void
    if (n == "tensor_set" || n == "tensor_set2" || n == "tensor_set3") {
      const size_t want = (n == "tensor_set") ? 3 : (n == "tensor_set2") ? 4 : 5;
      if (arity != want)
        typePanic("call '" + n + "' expects " + std::to_string(want) +
                  " arguments, got " + std::to_string(arity));
      NdlTensor* t = evalTensorExpr(arg(0));
      int64_t i0 = evalInt(arg(1));
      if (want == 3) {
        ndl_rt_tensor_set1(t, i0, evalDouble(arg(2)));
        return Value{VoidV{}};
      }
      int64_t i1 = evalInt(arg(2));
      if (want == 4) {
        ndl_rt_tensor_set2(t, i0, i1, evalDouble(arg(3)));
        return Value{VoidV{}};
      }
      int64_t i2 = evalInt(arg(3));
      ndl_rt_tensor_set3(t, i0, i1, i2, evalDouble(arg(4)));
      return Value{VoidV{}};
    }

    // dump_tensor(name: String, t: Tensor) -> Void
    if (n == "dump_tensor") {
      if (arity != 2) typePanic("call 'dump_tensor' expects 2 arguments, got " +
                                std::to_string(arity));
      std::string name = evalString(arg(0));
      NdlTensor* t = evalTensorExpr(arg(1));
      ndl_rt_dump_tensor(name.c_str(), t);
      return Value{VoidV{}};
    }

    // ---- v2.0 — readout & action decoder (RSOL) ----
    // get_membrane_potentials(g: Group) -> Tensor[size]
    if (n == "get_membrane_potentials") {
      if (arity != 1) typePanic("call 'get_membrane_potentials' expects 1 argument");
      NdlGroup* g = evalGroupExpr(arg(0));
      return Value{TensorV{ndl_rt_get_membrane_potentials(g)}};
    }
    // predict_linear(x: Tensor[N], w: Tensor[M,N]) -> Tensor[M]
    if (n == "predict_linear") {
      if (arity != 2) typePanic("call 'predict_linear' expects 2 arguments");
      NdlTensor* x = evalTensorExpr(arg(0));
      NdlTensor* w = evalTensorExpr(arg(1));
      return Value{TensorV{ndl_rt_predict_linear(x, w)}};
    }
    // vector_l2_norm(t: Tensor) -> Float
    if (n == "vector_l2_norm") {
      if (arity != 1) typePanic("call 'vector_l2_norm' expects 1 argument");
      NdlTensor* t = evalTensorExpr(arg(0));
      return Value{ndl_rt_vector_l2_norm(t)};
    }
    // stream_set(streamName, index: Int, value: Float) -> Void
    if (n == "stream_set") {
      if (arity != 3) typePanic("call 'stream_set' expects 3 arguments");
      const Expr& first = arg(0);
      if (first.kind != ExprKind::Ident)
        typePanic("stream_set: argument 1 must be a stream name");
      auto sid = streams_.find(first.name);
      if (sid == streams_.end())
        typePanic("stream_set: unknown stream '" + first.name + "'");
      int64_t idx = evalInt(arg(1));
      double v = evalDouble(arg(2));
      ndl_rt_stream_set(sid->second, idx, v);
      return Value{VoidV{}};
    }

    typePanic("unknown function '" + n + "'");
  }

  // --- typed evaluation helpers (post-sema these only assert) -------------------
  int64_t evalInt(const Expr& e) {
    Value v = evalExpr(e);
    if (const int64_t* i = std::get_if<int64_t>(&v)) return *i;
    typePanic(std::string("expected int, got ") + kindName(v));
  }

  // Int, or Float truncated to int (used for `Group[a..b]` slice bounds).
  int64_t evalIntCoerced(const Expr& e) {
    Value v = evalExpr(e);
    if (const int64_t* i = std::get_if<int64_t>(&v)) return *i;
    if (const double* f = std::get_if<double>(&v)) return static_cast<int64_t>(*f);
    typePanic(std::string("expected int (or float for slice bounds), got ") +
              kindName(v));
  }

  double evalDouble(const Expr& e) { // Int or Float
    Value v = evalExpr(e);
    if (isNum(v)) return numAsDouble(v);
    typePanic(std::string("expected a number, got ") + kindName(v));
  }

  bool evalBool(const Expr& e) {
    Value v = evalExpr(e);
    if (const bool* b = std::get_if<bool>(&v)) return *b;
    typePanic(std::string("expected bool, got ") + kindName(v));
  }

  std::string evalString(const Expr& e) {
    Value v = evalExpr(e);
    if (const std::string* s = std::get_if<std::string>(&v)) return *s;
    typePanic(std::string("expected string, got ") + kindName(v));
  }

  // Duration -> ms | Float -> ms | Int -> ms (run/at time quantities).
  double evalRunTime(const Expr& e) {
    Value v = evalExpr(e);
    if (const DurationV* d = std::get_if<DurationV>(&v)) return d->ms;
    if (isNum(v)) return numAsDouble(v);
    typePanic(std::string("expected a duration/number, got ") + kindName(v));
  }

  NdlGroup* lookupGroup(const std::string& name) {
    const Value* v = findVar(name);
    if (!v) typePanic("undefined group '" + name + "'");
    const GroupV* g = std::get_if<GroupV>(v);
    if (!g) typePanic("'" + name + "' is not a group (got " + kindName(*v) + ")");
    return g->p;
  }

  // --- statements ----------------------------------------------------------------
  void execStmt(const Stmt& s) {
    switch (s.kind) {
      case StmtKind::Import:
      case StmtKind::Use:
        break; // import/use resolved by the driver; merged program already ordered

      case StmtKind::NodeGroup:
        execNodeGroup(s);
        break;

      case StmtKind::Connect:
        execConnect(s);
        break;

      case StmtKind::Stdp:
        execStdp(s);
        break;

      case StmtKind::Let: {
        Value v = evalExpr(needExpr(s.init, "let initializer"));
        declare(s.name, std::move(v));
        break;
      }

      case StmtKind::Assign: {
        Value v = evalExpr(needExpr(s.init, "assignment value"));
        // v2.0: assignment to a signal updates the runtime neuromodulator too.
        if (signals_.count(s.name)) {
          double dv = 0.0;
          if (const int64_t* i = std::get_if<int64_t>(&v)) dv = (double)*i;
          else if (const double* f = std::get_if<double>(&v)) dv = *f;
          else typePanic("signal '" + s.name + "' must be assigned a number, got " +
                         kindName(v));
          ndl_rt_signal_set(s.name.c_str(), dv);
        }
        assign(s.name, std::move(v));
        break;
      }

      case StmtKind::Print:
        execPrint(s);
        break;

      case StmtKind::SaveCheckpoint:
        ndl_rt_save_checkpoint(evalString(needExpr(s.init, "checkpoint path")).c_str());
        break;

      case StmtKind::LoadCheckpoint:
        ndl_rt_load_checkpoint(evalString(needExpr(s.init, "checkpoint path")).c_str());
        break;

      case StmtKind::ExportRaster: {
        NdlGroup* g = lookupGroup(s.srcName);
        std::string path = evalString(needExpr(s.init, "raster path"));
        ndl_rt_export_raster(g, path.c_str());
        break;
      }

      case StmtKind::If: {
        bool c = evalBool(needExpr(s.cond, "if condition"));
        if (c) execBlockScoped(s.body);
        else   execBlockScoped(s.elseBody);
        break;
      }

      case StmtKind::For: {
        int64_t lo = evalInt(needExpr(s.lo, "for lower bound"));
        int64_t hi = evalInt(needExpr(s.hi, "for upper bound"));
        for (int64_t v = lo; v < hi; ++v) { // range [lo, hi)
          scopes_.emplace_back();
          declare(s.loopVar, Value{v});
          execBlock(s.body);
          scopes_.pop_back();
        }
        break;
      }

      case StmtKind::While: {
        while (evalBool(needExpr(s.cond, "while condition"))) {
          execBlockScoped(s.body); // fresh scope per iteration
        }
        break;
      }

      case StmtKind::Run:
        execRun(s);
        break;

      case StmtKind::AtEmit:
        // Only legal inside a run block, where AtEmits are collected and
        // scheduled by execRun (sema rejects stray ones; defensive).
        typePanic("'at ... emit' outside of a run block");
        break;

      case StmtKind::OnSpike:
        execOnSpike(s);
        break;

      // ---- v2.0 — asynchronous continuous-time environment ----
      case StmtKind::SignalDecl: {
        double init = 0.0;
        if (s.init) init = evalDouble(*s.init);
        ndl_rt_signal_create(s.name.c_str(), init);
        signals_.insert(s.name);
        declare(s.name, Value{init}); // readable/assignable as a normal Float
        break;
      }

      case StmtKind::OscillatorDecl: {
        double freq = 0.0, amp = 0.0, phase = 0.0;
        NdlGroup* target = nullptr;
        for (const CallArg& a : s.opts) {
          if (!a.value) typePanic("oscillator: empty parameter");
          if (a.name == "frequency") freq = evalDouble(*a.value);
          else if (a.name == "amplitude") amp = evalDouble(*a.value);
          else if (a.name == "phase") phase = evalDouble(*a.value);
          else if (a.name == "target") {
            if (a.value->kind != ExprKind::Ident)
              typePanic("oscillator 'target' must be a group name");
            target = lookupGroup(a.value->name);
          } else {
            typePanic("unknown oscillator parameter '" + a.name + "'");
          }
        }
        if (!target) typePanic("oscillator '" + s.name + "' is missing 'target'");
        ndl_rt_oscillator_add(s.name.c_str(), target, freq, amp, phase);
        break;
      }

      case StmtKind::ExternalStreamDecl: {
        const int64_t size = evalInt(needExpr(s.sizeExpr, "stream size"));
        const int64_t id = ndl_rt_stream_create(s.name.c_str(), size);
        streams_[s.name] = id;
        declare(s.name, Value{id}); // stream handle (Int)
        break;
      }

      case StmtKind::BindInputStream: {
        auto sid = streams_.find(s.srcName);
        if (sid == streams_.end())
          typePanic("bind_input_stream: unknown stream '" + s.srcName + "'");
        NdlGroup* target = lookupGroup(s.dstName);
        int32_t encoding = 0; // Poisson
        double maxFreq = 100.0;
        double kick = 1.0;
        for (const CallArg& a : s.opts) {
          if (!a.value) typePanic("bind_input_stream: empty option");
          if (a.name == "encoding") {
            // `encoding=Poisson` is an identifier, not a value expression.
            std::string enc;
            if (a.value->kind == ExprKind::Ident)
              enc = a.value->name;
            else
              enc = evalString(*a.value);
            if (enc != "Poisson") typePanic("bind_input_stream: unknown encoding '" +
                                            enc + "' (only Poisson is supported)");
          } else if (a.name == "max_freq") {
            maxFreq = evalDouble(*a.value);
          } else if (a.name == "kick") {
            kick = evalDouble(*a.value);
          } else {
            typePanic("unknown bind_input_stream option '" + a.name + "'");
          }
        }
        ndl_rt_bind_input_stream(sid->second, target, encoding, maxFreq, kick);
        break;
      }

      case StmtKind::RunContinuous:
        execRunContinuous(s);
        break;

      case StmtKind::StopContinuous:
        ndl_rt_stop_continuous();
        break;

      case StmtKind::WaitContinuous:
        ndl_rt_wait_continuous();
        break;

      case StmtKind::SetPlasticity:
      case StmtKind::PruneWeights: {
        NdlGroup* src = lookupGroup(s.srcName);
        NdlGroup* dst = lookupGroup(s.dstName);
        if (s.kind == StmtKind::SetPlasticity) {
          bool enabled = false;
          for (const CallArg& a : s.opts)
            if (a.name == "enabled") enabled = evalBool(needExpr(a.value, "enabled"));
          ndl_rt_set_plasticity(src, dst, enabled ? 1 : 0);
        } else {
          double thr = 0.0;
          for (const CallArg& a : s.opts)
            if (a.name == "threshold") thr = evalDouble(needExpr(a.value, "threshold"));
          ndl_rt_prune_weights(src, dst, thr);
        }
        break;
      }

      case StmtKind::ExprStmt:
        (void)evalExpr(needExpr(s.expr, "expression statement"));
        break;
    }
  }

  // node_group Name[size] : Type(tau=.., threshold=.., rest=.., reset=..);
  void execNodeGroup(const Stmt& s) {
    const int64_t size = evalInt(needExpr(s.sizeExpr, "group size"));

    // LIF defaults mirror the reference programs; sema requires explicit
    // values, so these only serve as a defensive fallback.
    double tau = 20.0, threshold = -55.0, rest = -70.0, reset = -75.0;
    static const char* const kNames[4] = {"tau", "threshold", "rest", "reset"};
    double* slots[4] = {&tau, &threshold, &rest, &reset};

    size_t positional = 0;
    for (const CallArg& a : s.params) {
      if (!a.value) typePanic("node_group '" + s.name + "': empty parameter");
      double val = evalDouble(*a.value);
      if (a.named) {
        bool matched = false;
        for (int k = 0; k < 4; ++k) {
          if (a.name == kNames[k]) { *slots[k] = val; matched = true; break; }
        }
        if (!matched) typePanic("unknown node_group parameter '" + a.name + "'");
      } else { // positional fallback: tau, threshold, rest, reset
        if (positional >= 4) typePanic("too many node_group parameters");
        *slots[positional++] = val;
      }
    }

    const int32_t ntype = (s.ntype == NeuronType::Inhibitory) ? 1 : 0;
    NdlGroup* g = ndl_rt_group_create(size, ntype, tau, threshold, rest, reset,
                                      s.name.c_str());
    declare(s.name, Value{GroupV{g}}); // visible by name from now on
  }

  void execConnect(const Stmt& s) {
    NdlGroup* src = lookupGroup(s.srcName);
    NdlGroup* dst = lookupGroup(s.dstName);

    double weight = 1.0;      // default weight when no distribution given
    int32_t distKind = 0;     // 0 = constant, 1 = gaussian, 2 = uniform
    double p0 = weight, p1 = 0.0;
    double density = 1.0;     // sparse default: fully connected
    int32_t plastic = 0;      // default false

    for (const CallArg& a : s.opts) {
      if (!a.value) typePanic("connect: empty option");
      if (!a.named) typePanic("connect options must be named (weight=..., ...)");
      if (a.name == "weight") {
        weight = evalDouble(*a.value);
      } else if (a.name == "weight_func") {
        // weight_func is a Call node: random_gaussian(mu, sigma) or
        // random_uniform(a, b). The two numeric arguments are evaluated NOW;
        // per-synapse sampling happens inside libndl_rt (do not call the
        // RNG here — that would draw a single shared sample).
        const Expr& fn = *a.value;
        if (fn.kind != ExprKind::Call)
          typePanic("weight_func must be random_gaussian(..) or random_uniform(..)");
        if (fn.args.size() != 2)
          typePanic("weight_func distribution expects 2 numeric arguments");
        if (fn.name == "random_gaussian") {
          distKind = 1;
          p0 = evalDouble(needExpr(fn.args[0].value, "weight_func arg")); // mu
          p1 = evalDouble(needExpr(fn.args[1].value, "weight_func arg")); // sigma
        } else if (fn.name == "random_uniform") {
          distKind = 2;
          p0 = evalDouble(needExpr(fn.args[0].value, "weight_func arg")); // a
          p1 = evalDouble(needExpr(fn.args[1].value, "weight_func arg")); // b
        } else {
          typePanic("unsupported weight_func '" + fn.name + "'");
        }
      } else if (a.name == "density") {
        density = evalDouble(*a.value);
      } else if (a.name == "plastic") {
        plastic = evalBool(*a.value) ? 1 : 0;
      } else {
        typePanic("unknown connect option '" + a.name + "'");
      }
    }

    switch (s.ckind) {
      case ConnectKind::Dense:
        if (distKind == 0) { p0 = weight; p1 = 0.0; } // constant weight
        ndl_rt_dense_connect(src, dst, distKind, p0, p1, plastic);
        break;
      case ConnectKind::Sparse:
        ndl_rt_sparse_connect(src, dst, density, weight, plastic);
        break;
      case ConnectKind::OneToOne:
        ndl_rt_one_to_one_connect(src, dst, weight, plastic);
        break;
    }
  }

  void execStdp(const Stmt& s) {
    NdlGroup* src = lookupGroup(s.srcName);
    NdlGroup* dst = lookupGroup(s.dstName);

    // Defaults mirror the reference program; sema requires explicit values.
    double lrPot = 0.01, lrDep = 0.005, windowMs = 20.0;
    static const char* const kNames[3] = {"lr_pot", "lr_dep", "window_ms"};
    double* slots[3] = {&lrPot, &lrDep, &windowMs};
    std::string modulator; // v2.0: modulator signal name (may be empty)

    size_t positional = 0;
    for (const CallArg& a : s.opts) {
      if (!a.value) typePanic("configure_stdp: empty option");
      if (a.name == "modulator" && a.named) {
        // `modulator=Dopamine` is an identifier naming a signal.
        if (a.value->kind != ExprKind::Ident)
          typePanic("configure_stdp 'modulator' must name a signal");
        modulator = a.value->name;
        continue;
      }
      double val = evalDouble(*a.value);
      if (a.named) {
        bool matched = false;
        for (int k = 0; k < 3; ++k) {
          if (a.name == kNames[k]) { *slots[k] = val; matched = true; break; }
        }
        if (!matched) typePanic("unknown configure_stdp option '" + a.name + "'");
      } else {
        if (positional >= 3) typePanic("too many configure_stdp options");
        *slots[positional++] = val;
      }
    }

    ndl_rt_configure_stdp(src, dst, lrPot, lrDep, windowMs);
    if (!modulator.empty())
      ndl_rt_stdp_set_modulator(src, dst, modulator.c_str());
  }

  void execPrint(const Stmt& s) {
    ndl_rt_print_begin();
    for (const ExprPtr& a : s.printArgs) {
      if (!a) continue;
      Value v = evalExpr(*a);
      switch (v.index()) {
        case 1: // int
          ndl_rt_print_i64(std::get<int64_t>(v));
          break;
        case 2: // float
          ndl_rt_print_f64(std::get<double>(v));
          break;
        case 3: // bool
          ndl_rt_print_bool(std::get<bool>(v));
          break;
        case 4: // string
          ndl_rt_print_str(std::get<std::string>(v).c_str());
          break;
        case 5: // duration -> printed as its millisecond count
          ndl_rt_print_f64(std::get<DurationV>(v).ms);
          break;
        case 6: { // group -> "<group NAME>"
          std::string txt =
              std::string("<group ") + ndl_rt_group_name(std::get<GroupV>(v).p) + ">";
          ndl_rt_print_str(txt.c_str());
          break;
        }
        case 7: // tensor
          ndl_rt_print_str("<tensor>");
          break;
        default:
          typePanic("cannot print a void value");
      }
    }
    ndl_rt_print_end();
  }

  // run (duration = .., dt = .., device = ..) { at .. emit ..; }
  void execRun(const Stmt& s) {
    double dur = 0.0;
    double dt = 1.0; // per contract: missing dt means 1 ms
    if (s.hasDuration && s.durationExpr) dur = evalRunTime(*s.durationExpr);
    if (s.hasDt && s.dtExpr) dt = evalRunTime(*s.dtExpr);
    const int32_t device = s.hasDevice ? (s.device == DeviceKind::GPU ? 1 : 0) : 0;

    // Collect every AtEmit of the body in source order (the run body contains
    // only AtEmits; the walk is defensive and skips nested run blocks, which
    // sema forbids anyway), schedule them, then run the simulation.
    std::vector<const Stmt*> emits;
    collectAtEmits(s.body, emits);
    for (const Stmt* at : emits) execAtEmit(*at);

    ndl_rt_run(dur, dt, device);
  }

  static void collectAtEmits(const std::vector<StmtPtr>& body,
                             std::vector<const Stmt*>& out) {
    for (const StmtPtr& st : body) {
      if (!st) continue;
      if (st->kind == StmtKind::AtEmit) { out.push_back(st.get()); continue; }
      if (st->kind == StmtKind::Run) continue; // nested run owns its own emits
      collectAtEmits(st->body, out);
      collectAtEmits(st->elseBody, out);
    }
  }

  // at <time> emit Group[<slice>](current = ..);
  void execAtEmit(const Stmt& s) {
    NdlGroup* g = lookupGroup(s.name);

    int64_t lo = 0;
    int64_t hi = ndl_rt_group_size(g); // bare group -> whole population
    if (s.sliceHasIndex) {             // Group[i] -> neuron i only
      int64_t i = evalIntCoerced(needExpr(s.sliceLo, "slice index"));
      lo = i;
      hi = i + 1;
    } else if (s.sliceHasRange) {      // Group[a..b] -> neurons [a, b)
      lo = evalIntCoerced(needExpr(s.sliceLo, "slice lower bound"));
      hi = evalIntCoerced(needExpr(s.sliceHi, "slice upper bound"));
    }

    double t = evalRunTime(needExpr(s.timeExpr, "emit time"));
    double current = evalDouble(needExpr(s.currentExpr, "emit current"));
    ndl_rt_schedule_inject(g, lo, hi, t, current);
  }

  // run_continuous (dt = .., device = ..) { body } — execute the body once
  // (setup), then start the asynchronous event loop thread.
  void execRunContinuous(const Stmt& s) {
    double dt = 1.0; // per contract: missing dt means 1 ms
    if (s.hasDt && s.dtExpr) dt = evalRunTime(*s.dtExpr);
    const int32_t device = s.hasDevice ? (s.device == DeviceKind::GPU ? 1 : 0) : 0;

    execBlockScoped(s.body);
    ndl_rt_run_continuous(dt, device);
    ranContinuous_ = true;
  }

  // on_spike Group { ... } — register handler; body runs later from the
  // runtime through ndl_vm_spike_trampoline.
  void execOnSpike(const Stmt& s) {
    NdlGroup* g = lookupGroup(s.name);
    handlers_.push_back(HandlerDesc{&s});
    const size_t slot = handlers_.size() - 1;

    auto ctx = std::make_unique<HandlerCtx>();
    ctx->interp = this;
    ctx->slot = slot;
    HandlerCtx* raw = ctx.get();
    handlerCtxs_.push_back(std::move(ctx));

    ndl_rt_register_spike_handler(g, &ndl_vm_spike_trampoline, raw);
  }

  // Handler body executes in a fresh scope exposing the built-in bindings
  // `neuron_index: Int` and `spike_time: Float` (ms).
  void invokeHandler(size_t slot, int64_t neuron, double tMs) {
    if (slot >= handlers_.size())
      ndl_rt_panic("internal error: bad spike handler slot");
    const Stmt& st = *handlers_[slot].stmt;
    scopes_.emplace_back();
    declare("neuron_index", Value{neuron});
    declare("spike_time", Value{tMs});
    execBlock(st.body);
    scopes_.pop_back();
  }
};

} // namespace

// ---------------------------------------------------------------------------
// C trampoline: libndl_rt calls this on every spike of a registered group.
// ---------------------------------------------------------------------------
extern "C" void ndl_vm_spike_trampoline(int64_t neuron, double t_ms, void* user) {
  auto* ctx = static_cast<HandlerCtx*>(user);
  ctx->interp->invokeHandler(ctx->slot, neuron, t_ms);
}

// ---------------------------------------------------------------------------
// VM entry points (contract header)
// ---------------------------------------------------------------------------
VM::VM(const Program& program, DiagnosticEngine& diag, VMOptions opts)
    : prog_(program), diag_(diag), opts_(std::move(opts)) {}

int VM::run() {
  Interp interp(prog_, diag_, opts_);
  return interp.run();
}

} // namespace ndl
