// NDL v1.0 — Semantic analyzer (two-pass): symbols, placement, types.
// Implements the §4 contract of docs/INTERNALS.md against include/ndl/sema.hpp.
//
// Pass 1 collects the global declaration environment: node_group declarations
// (names, const-evaluated sizes, parameter sets) and top-level let bindings
// with constant initializers (usable in later group size expressions).
//
// Pass 2 walks all modules in import order with a scope stack and checks:
//   - statement placement (E0207 / E0210 / E0211 / E0212 / E0213 / E0214),
//   - node_group / connect / configure_stdp option sets (E0205),
//   - plastic-connection tracking for configure_stdp (E0209),
//   - expression typing with Int→Float promotion (E0206),
//   - name resolution with "did you mean" suggestions (E0201),
//   - duplicate declarations (E0202),
//   - builtin call signatures (sin/cos/exp/log/sqrt, random_*, get_weight,
//     tensor_new*/get*/set*, dump_tensor, get_membrane_potentials,
//     predict_linear, vector_l2_norm),
//   - run / at_emit / on_spike details, constant slice bounds (E0203),
//     empty slice ranges (W0201),
//   - v2.0: signal/oscillator/external-stream declarations, bind_input_stream,
//     set_plasticity / prune_weights (E0216), run() vs run_continuous() mixing
//     (E0215), modulated 3-factor STDP (configure_stdp modulator=<signal>).
//
// Error recovery: a failed subexpression yields TypeInfo::Error(); parents
// suppress cascading reports and traversal always continues so a single run
// collects as many diagnostics as possible. All types of every expression
// node (after promotion rules) are recorded in SemaResult::exprTypes for the
// code generators.

#include "ndl/sema.hpp"

#include "ndl/ast.hpp"
#include "ndl/diag.hpp"
#include "ndl/tokens.hpp"

#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ndl {

// ---------------------------------------------------------------------------
// TypeInfo::toString / SemaResult::findGroup (declared in sema.hpp)
// ---------------------------------------------------------------------------

std::string TypeInfo::toString() const {
  switch (k) {
    case K::Error:    return "<error>";
    case K::Int:      return "Int";
    case K::Float:    return "Float";
    case K::Bool:     return "Bool";
    case K::Str:      return "Str";
    case K::Duration: return "Duration";
    case K::Group:    return "Group";
    case K::Void:     return "Void";
    case K::Tensor:   break;
  }
  if (tensorDims.empty()) return "Tensor";
  std::string s = "Tensor[";
  for (size_t i = 0; i < tensorDims.size(); ++i) {
    if (i) s += ",";
    s += std::to_string(tensorDims[i]);
  }
  s += "]";
  return s;
}

const GroupInfo* SemaResult::findGroup(const std::string& name) const {
  auto it = groupIndex.find(name);
  if (it == groupIndex.end()) return nullptr;
  return &groups[it->second];
}

namespace {

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const std::vector<std::string>& builtinNames() {
  static const std::vector<std::string> names{
      "sin", "cos", "exp", "log", "sqrt", "random_uniform", "random_gaussian",
      "get_weight", "tensor_new", "tensor_new2", "tensor_new3",
      "tensor_get", "tensor_get2", "tensor_get3",
      "tensor_set", "tensor_set2", "tensor_set3", "dump_tensor",
      // v2.0 — readout & action decoder (RSOL)
      "get_membrane_potentials", "predict_linear", "vector_l2_norm",
      // v2.0 — stream mailbox write (first arg is a stream name, not a value)
      "stream_set"};
  return names;
}

bool isBuiltinName(const std::string& n) {
  for (const std::string& b : builtinNames())
    if (b == n) return true;
  return false;
}

const char* opSpelling(Tok op) {
  switch (op) {
    case Tok::Plus:    return "+";
    case Tok::Minus:   return "-";
    case Tok::Star:    return "*";
    case Tok::Slash:   return "/";
    case Tok::Percent: return "%";
    case Tok::EqEq:    return "==";
    case Tok::NotEq:   return "!=";
    case Tok::Lt:      return "<";
    case Tok::Gt:      return ">";
    case Tok::Le:      return "<=";
    case Tok::Ge:      return ">=";
    case Tok::AndAnd:  return "&&";
    case Tok::OrOr:    return "||";
    case Tok::Not:     return "!";
    default:           return "?";
  }
}

// Traversal context for pass 2.
//   Top        — directly at module top level.
//   RunBody    — directly inside a `run` body (only `at ... emit` is legal).
//   Handler    — anywhere inside an `on_spike` body.
//   ContBody   — anywhere inside a `run_continuous` body.
//   Block      — inside if/for/while bodies (outside run/handler).
//   BlockNoTop — like Block, but the "only allowed at top level" diagnostic
//                for this statement was already emitted (E0210 demotion).
enum class Ctx { Top, RunBody, Handler, ContBody, Block, BlockNoTop };

Ctx innerCtx(Ctx c) {
  return (c == Ctx::Handler || c == Ctx::ContBody) ? c : Ctx::Block;
}

// ---------------------------------------------------------------------------
// Analyzer driver
// ---------------------------------------------------------------------------

struct Impl {
  Program& prog;
  DiagnosticEngine& diag;
  SemaResult res;

  // ---- pass 1 state ----
  std::unordered_set<const Stmt*> topGroupStmts;   // registered top-level node_groups
  std::unordered_set<const Stmt*> topLetOwner;     // top-level lets owning their name
  std::unordered_map<std::string, SourceRange> topLetRange;
  std::unordered_map<std::string, int64_t> constInts;   // const top-level int lets
  std::unordered_map<std::string, double> constFloats;  // const top-level numeric lets
  std::unordered_set<std::string> badSize;              // groups with unknown size

  // ---- pass 2 state ----
  std::vector<std::unordered_map<std::string, TypeInfo>> scopes; // scopes[0] = global
  std::set<std::pair<std::string, std::string>> plasticPairs;    // src -> dst
  std::set<std::pair<std::string, std::string>> allConnPairs;    // src -> dst (any kind)
  bool sawRun = false;        // a run() statement was seen
  bool sawContinuous = false; // a run_continuous statement was seen
  bool mixReported = false;   // E0215 emitted once

  Impl(Program& p, DiagnosticEngine& d) : prog(p), diag(d) {}

  // ---- diagnostics helpers ----
  void err(const char* code, SourceRange r, std::string msg,
           std::vector<DiagLabel> labels = {}, std::vector<std::string> notes = {}) {
    diag.error(code, r, std::move(msg), std::move(labels), std::move(notes));
  }
  void warn(const char* code, SourceRange r, std::string msg) {
    diag.warning(code, r, std::move(msg));
  }
  void suggest(const std::string& word, const std::vector<std::string>& candidates,
               std::vector<std::string>& notes) {
    std::string m = bestMatch(word, candidates);
    if (!m.empty()) notes.push_back("help: did you mean '" + m + "'?");
  }

  std::vector<std::string> visibleNames() const {
    std::vector<std::string> v;
    v.reserve(res.groups.size() + builtinNames().size());
    for (const GroupInfo& g : res.groups) v.push_back(g.name);
    for (const auto& sc : scopes)
      for (const auto& kv : sc) v.push_back(kv.first);
    for (const std::string& b : builtinNames()) v.push_back(b);
    return v;
  }

  // ---- const evaluation (pass 1 environment) ----
  bool constEvalInt(const Expr* e, int64_t& out) const {
    if (!e) return false;
    switch (e->kind) {
      case ExprKind::IntLit:
        out = static_cast<int64_t>(e->intVal);
        return true;
      case ExprKind::Unary:
        if (e->op == Tok::Minus && constEvalInt(e->lhs.get(), out)) {
          out = -out;
          return true;
        }
        return false;
      case ExprKind::Binary: {
        int64_t a = 0, b = 0;
        if (!constEvalInt(e->lhs.get(), a) || !constEvalInt(e->rhs.get(), b))
          return false;
        switch (e->op) {
          case Tok::Plus:    out = a + b; return true;
          case Tok::Minus:   out = a - b; return true;
          case Tok::Star:    out = a * b; return true;
          case Tok::Slash:   if (b == 0) return false; out = a / b; return true;
          case Tok::Percent: if (b == 0) return false; out = a % b; return true;
          default: return false;
        }
      }
      case ExprKind::Ident: {
        auto it = constInts.find(e->name);
        if (it != constInts.end()) { out = it->second; return true; }
        return false;
      }
      default:
        return false;
    }
  }

  bool constEvalFloat(const Expr* e, double& out) const {
    if (!e) return false;
    switch (e->kind) {
      case ExprKind::IntLit:
        out = static_cast<double>(e->intVal);
        return true;
      case ExprKind::FloatLit:
      case ExprKind::DurationLit:
        out = e->floatVal;
        return true;
      case ExprKind::Unary:
        if (e->op == Tok::Minus && constEvalFloat(e->lhs.get(), out)) {
          out = -out;
          return true;
        }
        return false;
      case ExprKind::Binary: {
        double a = 0, b = 0;
        if (!constEvalFloat(e->lhs.get(), a) || !constEvalFloat(e->rhs.get(), b))
          return false;
        switch (e->op) {
          case Tok::Plus:  out = a + b; return true;
          case Tok::Minus: out = a - b; return true;
          case Tok::Star:  out = a * b; return true;
          case Tok::Slash: if (b == 0) return false; out = a / b; return true;
          default: return false;
        }
      }
      case ExprKind::Ident: {
        auto ii = constInts.find(e->name);
        if (ii != constInts.end()) { out = static_cast<double>(ii->second); return true; }
        auto fi = constFloats.find(e->name);
        if (fi != constFloats.end()) { out = fi->second; return true; }
        return false;
      }
      default:
        return false;
    }
  }

  // ---- scope helpers ----
  const TypeInfo* lookupVar(const std::string& name) const {
    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
      auto f = it->find(name);
      if (f != it->end()) return &f->second;
    }
    return nullptr;
  }

  static TypeInfo typeRefToInfo(const TypeRef& tr) {
    switch (tr.k) {
      case TypeRef::K::Int:    return TypeInfo::Int();
      case TypeRef::K::Float:  return TypeInfo::Float();
      case TypeRef::K::Bool:   return TypeInfo::Bool();
      case TypeRef::K::String: return TypeInfo::Str();
      case TypeRef::K::Tensor: {
        TypeInfo t = TypeInfo::Tensor();
        t.tensorDims = tr.tensorDims;
        return t;
      }
      case TypeRef::K::None:   break;
    }
    return TypeInfo::Error();
  }

  // May a value of type `src` be used where `target` is expected?
  static bool assignable(const TypeInfo& target, const TypeInfo& src) {
    if (target.isError() || src.isError()) return true; // recovery: no cascade
    if (target.k == src.k) return true;                 // Tensor: dims ignored
    if (target.k == TypeInfo::K::Float && src.k == TypeInfo::K::Int) return true;
    return false;
  }

  // ==========================================================================
  // Pass 1 — global declarations
  // ==========================================================================
  void runPass1() {
    for (const auto& mod : prog.modules) {
      if (!mod) continue;
      for (const auto& item : mod->items) {
        if (!item) continue;
        if (item->kind == StmtKind::NodeGroup)
          pass1Group(*mod, *item);
        else if (item->kind == StmtKind::Let)
          pass1Let(*item);
        else if (item->kind == StmtKind::SignalDecl)
          pass1Signal(*item);
        else if (item->kind == StmtKind::OscillatorDecl)
          pass1Oscillator(*mod, *item);
        else if (item->kind == StmtKind::ExternalStreamDecl)
          pass1Stream(*mod, *item);
      }
    }
  }

  void dupDiag(const std::string& what, const std::string& name, SourceRange r) {
    const SourceRange* prev = nullptr;
    auto gi = res.groupIndex.find(name);
    if (gi != res.groupIndex.end())
      prev = &res.groups[gi->second].declRange;
    else {
      auto li = topLetRange.find(name);
      if (li != topLetRange.end()) prev = &li->second;
    }
    std::vector<DiagLabel> labels;
    if (prev) labels.push_back({"previous declaration is here", *prev});
    err("E0202", r, "duplicate " + what + " '" + name + "'", std::move(labels));
  }

  void pass1Group(const Module& mod, Stmt& s) {
    if (res.groupIndex.count(s.name) || topLetRange.count(s.name)) {
      dupDiag("node_group", s.name, s.range);
      return; // first declaration wins
    }
    GroupInfo gi;
    gi.name = s.name;
    gi.type = s.ntype;
    gi.module = mod.path;
    gi.declRange = s.range;
    checkGroupSize(s, gi.size);
    res.groupIndex[s.name] = res.groups.size();
    res.groups.push_back(std::move(gi));
    topGroupStmts.insert(&s);
    checkGroupParams(s);
  }

  void checkGroupSize(Stmt& s, uint64_t& outSize) {
    int64_t sz = 0;
    bool isConst = s.sizeExpr && constEvalInt(s.sizeExpr.get(), sz);
    if (isConst && sz <= 0) {
      err("E0204", s.sizeExpr->range, "group size of '" + s.name +
          "' must be positive (got " + std::to_string(sz) + ")");
      badSize.insert(s.name);
    } else if (!isConst) {
      std::vector<std::string> notes{
          "note: the size must be an integer literal or an expression over "
          "integers and previously declared constant top-level 'let' bindings"};
      err("E0204", s.sizeExpr ? s.sizeExpr->range : s.range,
          "group size of '" + s.name + "' must be a compile-time constant", {}, notes);
      badSize.insert(s.name);
    } else {
      outSize = static_cast<uint64_t>(sz);
    }
  }

  void checkGroupParams(Stmt& s) {
    static const char* kReq[4] = {"tau", "threshold", "rest", "reset"};
    std::unordered_set<std::string> seen;
    for (const CallArg& p : s.params) {
      if (!p.named) {
        err("E0205", p.range,
            "node_group parameters must be named "
            "(tau=..., threshold=..., rest=..., reset=...)");
        continue;
      }
      bool known = false;
      for (const char* r : kReq)
        if (p.name == r) { known = true; break; }
      if (!known) {
        std::vector<std::string> notes;
        suggest(p.name, std::vector<std::string>(kReq, kReq + 4), notes);
        err("E0205", p.range, "unknown node_group parameter '" + p.name + "'", {}, notes);
        continue;
      }
      if (seen.count(p.name))
        err("E0205", p.range, "duplicate node_group parameter '" + p.name + "'");
      else
        seen.insert(p.name);
    }
    std::vector<std::string> missing;
    for (const char* r : kReq)
      if (!seen.count(r)) missing.push_back(r);
    if (!missing.empty()) {
      std::string list;
      for (size_t i = 0; i < missing.size(); ++i) {
        if (i) list += ", ";
        list += missing[i];
      }
      err("E0205", s.range,
          "node_group '" + s.name + "' is missing parameter(s): " + list);
    }
  }

  void pass1Let(Stmt& s) {
    if (res.groupIndex.count(s.name) || topLetRange.count(s.name) ||
        res.signalIndex.count(s.name) || res.streamIndex.count(s.name)) {
      dupDiag("declaration", s.name, s.range);
      return; // first declaration wins
    }
    topLetRange[s.name] = s.range;
    topLetOwner.insert(&s);
    if (s.init) {
      int64_t iv;
      if (constEvalInt(s.init.get(), iv)) constInts[s.name] = iv;
      double fv;
      if (constEvalFloat(s.init.get(), fv)) constFloats[s.name] = fv;
    }
  }

  // ---- v2.0 pass-1 declarations ----
  void pass1Signal(Stmt& s) {
    if (res.groupIndex.count(s.name) || topLetRange.count(s.name) ||
        res.signalIndex.count(s.name) || res.streamIndex.count(s.name)) {
      dupDiag("signal", s.name, s.range);
      return;
    }
    SignalInfo si;
    si.name = s.name;
    si.declRange = s.range;
    res.signalIndex[s.name] = res.signals.size();
    res.signals.push_back(std::move(si));
  }

  void pass1Oscillator(const Module& mod, Stmt& s) {
    if (topLetRange.count(s.name) || res.signalIndex.count(s.name) ||
        res.streamIndex.count(s.name)) {
      dupDiag("oscillator", s.name, s.range);
      return; // oscillator names may repeat group names (separate namespace)
    }
    OscillatorInfo oi;
    oi.name = s.name;
    oi.declRange = s.range;
    res.oscillators.push_back(std::move(oi));
    (void)mod;
  }

  void pass1Stream(const Module& mod, Stmt& s) {
    if (res.groupIndex.count(s.name) || topLetRange.count(s.name) ||
        res.signalIndex.count(s.name) || res.streamIndex.count(s.name)) {
      dupDiag("external stream", s.name, s.range);
      return;
    }
    StreamInfo si;
    si.name = s.name;
    si.declRange = s.range;
    int64_t sz = 0;
    if (s.sizeExpr && constEvalInt(s.sizeExpr.get(), sz) && sz > 0) {
      si.size = static_cast<uint64_t>(sz);
    } else {
      err("E0204", s.sizeExpr ? s.sizeExpr->range : s.range,
          "size of external stream '" + s.name +
              "' must be a positive compile-time constant");
      badSize.insert(s.name);
    }
    res.streamIndex[s.name] = res.streams.size();
    res.streams.push_back(std::move(si));
    (void)mod;
  }

  // ==========================================================================
  // Pass 2 — full traversal
  // ==========================================================================
  void runPass2() {
    for (const auto& mod : prog.modules) {
      if (!mod) continue;
      for (const auto& item : mod->items) stmt(item, Ctx::Top);
    }
  }

  void stmt(const StmtPtr& s, Ctx ctx) {
    if (!s) return;

    // Statements directly inside a run body: only `at ... emit` (plus the
    // constructs that carry their own, more specific codes).
    if (ctx == Ctx::RunBody && s->kind != StmtKind::AtEmit &&
        s->kind != StmtKind::Run && s->kind != StmtKind::OnSpike) {
      err("E0210", s->range,
          "only 'at <time> emit ...' statements are allowed directly inside a run body");
      if (s->kind == StmtKind::Let) { // do not bind into the enclosing scope
        if (s->init) expr(s->init);
        return;
      }
      ctx = Ctx::BlockNoTop; // placement already reported; keep checking contents
    }

    switch (s->kind) {
      case StmtKind::Import:
      case StmtKind::Use:
        // Imports are resolved before sema; only placement is checked here.
        if (ctx == Ctx::Handler || ctx == Ctx::Block || ctx == Ctx::BlockNoTop)
          err("E0207", s->range,
              std::string(s->kind == StmtKind::Import ? "'import'" : "'use'") +
                  " is only allowed at top level");
        break;
      case StmtKind::NodeGroup: nodeGroup(*s, ctx); break;
      case StmtKind::Connect:   connectStmt(*s, ctx); break;
      case StmtKind::Stdp:      stdpStmt(*s, ctx); break;
      case StmtKind::Let:       letStmt(*s, ctx); break;
      case StmtKind::Assign:    assignStmt(*s); break;
      case StmtKind::Print:     printStmt(*s); break;
      case StmtKind::SaveCheckpoint:
      case StmtKind::LoadCheckpoint: saveLoad(*s); break;
      case StmtKind::ExportRaster:   exportRaster(*s); break;
      case StmtKind::If:        ifStmt(*s, ctx); break;
      case StmtKind::For:       forStmt(*s, ctx); break;
      case StmtKind::While:     whileStmt(*s, ctx); break;
      case StmtKind::Run:       runStmt(*s, ctx); break;
      case StmtKind::AtEmit:    atEmit(*s, ctx); break;
      case StmtKind::OnSpike:   onSpike(*s, ctx); break;
      // v2.0 — asynchronous continuous-time environment
      case StmtKind::SignalDecl:         signalDecl(*s, ctx); break;
      case StmtKind::OscillatorDecl:     oscillatorDecl(*s, ctx); break;
      case StmtKind::ExternalStreamDecl: externalStreamDecl(*s, ctx); break;
      case StmtKind::BindInputStream:    bindInputStream(*s, ctx); break;
      case StmtKind::RunContinuous:      runContinuousStmt(*s, ctx); break;
      case StmtKind::StopContinuous:     stopContinuousStmt(*s, ctx); break;
      case StmtKind::WaitContinuous:     waitContinuousStmt(*s, ctx); break;
      case StmtKind::SetPlasticity:      structuralStmt(*s, ctx, /*isPrune=*/false); break;
      case StmtKind::PruneWeights:       structuralStmt(*s, ctx, /*isPrune=*/true); break;
      case StmtKind::ExprStmt:
        if (s->expr) expr(s->expr);
        break;
    }
  }

  // ---- node_group ----
  void nodeGroup(Stmt& s, Ctx ctx) {
    if (ctx == Ctx::Handler || ctx == Ctx::Block)
      err("E0207", s.range, "'node_group' is only allowed at top level");
    if (ctx != Ctx::Top) {
      // Not seen by pass 1: check its shape here as well.
      checkGroupParams(s);
      uint64_t unused = 0;
      checkGroupSize(s, unused);
    } else if (!topGroupStmts.count(&s)) {
      // Duplicate top-level declaration: pass 1 already reported it; the
      // first declaration owns the name. Still walk the expressions below.
    }
    if (s.sizeExpr) expr(s.sizeExpr);
    for (const CallArg& p : s.params) {
      if (!p.value) continue;
      TypeInfo t = expr(p.value);
      if (!t.isError() && !t.isNumeric())
        err("E0206", p.value->range,
            "type mismatch: node_group parameter '" + p.name +
                "' must be Float, got " + t.toString());
    }
  }

  // ---- shared option helpers ----
  const GroupInfo* resolveGroup(const std::string& name, SourceRange r) {
    const GroupInfo* g = res.findGroup(name);
    if (!g) {
      std::vector<std::string> cands;
      cands.reserve(res.groups.size());
      for (const GroupInfo& x : res.groups) cands.push_back(x.name);
      std::vector<std::string> notes;
      suggest(name, cands, notes);
      err("E0201", r, "unknown node group '" + name + "'", {}, notes);
    }
    return g;
  }

  TypeInfo checkOptNumeric(const CallArg& o, const std::string& kw) {
    if (!o.value) return TypeInfo::Error();
    TypeInfo t = expr(o.value);
    if (!t.isError() && !t.isNumeric())
      err("E0206", o.value->range,
          "type mismatch: " + kw + " option '" + o.name + "' must be Float, got " +
              t.toString());
    return t;
  }

  // ---- connect ----
  void connectStmt(Stmt& s, Ctx ctx) {
    const char* kw = s.ckind == ConnectKind::Dense     ? "dense_connect"
                     : s.ckind == ConnectKind::Sparse  ? "sparse_connect"
                                                       : "one_to_one_connect";
    if (ctx == Ctx::Handler || ctx == Ctx::Block)
      err("E0207", s.range, std::string("'") + kw + "' is only allowed at top level");

    resolveGroup(s.srcName, s.range);
    resolveGroup(s.dstName, s.range);

    std::unordered_set<std::string> seen;
    bool hasWeight = false, hasWeightFunc = false, hasDensity = false;
    const CallArg* plasticArg = nullptr;
    for (const CallArg& o : s.opts) {
      if (!o.named) {
        err("E0205", o.range, std::string(kw) + " options must be named");
        if (o.value) expr(o.value);
        continue;
      }
      bool knownOpt = o.name == "weight" || o.name == "weight_func" ||
                      o.name == "density" || o.name == "plastic";
      if (!knownOpt) {
        std::vector<std::string> notes;
        suggest(o.name, {"weight", "weight_func", "density", "plastic"}, notes);
        err("E0205", o.range, "unknown " + std::string(kw) + " option '" + o.name + "'",
            {}, notes);
        if (o.value) expr(o.value);
        continue;
      }
      if (seen.count(o.name)) {
        err("E0205", o.range,
            "duplicate " + std::string(kw) + " option '" + o.name + "'");
        if (o.value) expr(o.value);
        continue;
      }
      seen.insert(o.name);

      if (o.name == "weight") {
        hasWeight = true;
        checkOptNumeric(o, kw);
      } else if (o.name == "weight_func") {
        hasWeightFunc = true;
        if (!o.value || o.value->kind != ExprKind::Call ||
            (o.value->name != "random_gaussian" && o.value->name != "random_uniform")) {
          err("E0206", o.value ? o.value->range : o.range,
              "type mismatch: " + std::string(kw) +
                  " option 'weight_func' must be a call to "
                  "random_gaussian(mu, sigma) or random_uniform(a, b)");
        }
        if (o.value) expr(o.value); // validates the call arguments as usual
      } else if (o.name == "density") {
        hasDensity = true;
        checkOptNumeric(o, kw);
        double d;
        if (o.value && constEvalFloat(o.value.get(), d) && !(d > 0.0 && d <= 1.0))
          err("E0205", o.value->range,
              std::string(kw) + " option 'density' must be in (0, 1]");
      } else { // plastic
        plasticArg = &o;
        if (o.value) {
          TypeInfo t = expr(o.value);
          if (!t.isError() && t.k != TypeInfo::K::Bool)
            err("E0206", o.value->range,
                "type mismatch: " + std::string(kw) + " option 'plastic' must be Bool, got " +
                    t.toString());
        }
      }
    }

    if (s.ckind == ConnectKind::Dense) {
      if (hasWeight && hasWeightFunc)
        err("E0205", s.range,
            "dense_connect requires exactly one of 'weight' or 'weight_func'");
      else if (!hasWeight && !hasWeightFunc)
        err("E0205", s.range, "dense_connect requires a 'weight' or 'weight_func' option");
    } else if (s.ckind == ConnectKind::Sparse) {
      if (!hasDensity) err("E0205", s.range, "sparse_connect requires a 'density' option");
      if (!hasWeight)  err("E0205", s.range, "sparse_connect requires a 'weight' option");
    } else if (!hasWeight) {
      err("E0205", s.range, "one_to_one_connect requires a 'weight' option");
    }

    // Track plastic links (declaration order matters for configure_stdp).
    if (plasticArg && plasticArg->value && plasticArg->value->kind == ExprKind::BoolLit &&
        plasticArg->value->boolVal)
      plasticPairs.insert({s.srcName, s.dstName});
    allConnPairs.insert({s.srcName, s.dstName});
  }

  // ---- configure_stdp ----
  void stdpStmt(Stmt& s, Ctx ctx) {
    if (ctx == Ctx::Handler || ctx == Ctx::Block)
      err("E0207", s.range, "'configure_stdp' is only allowed at top level");

    resolveGroup(s.srcName, s.range);
    resolveGroup(s.dstName, s.range);

    std::unordered_set<std::string> seen;
    bool hasPot = false, hasDep = false, hasWin = false;
    const CallArg* modArg = nullptr;
    for (const CallArg& o : s.opts) {
      if (!o.named) {
        err("E0205", o.range, "configure_stdp options must be named");
        if (o.value) expr(o.value);
        continue;
      }
      if (o.name != "lr_pot" && o.name != "lr_dep" && o.name != "window_ms" &&
          o.name != "modulator") {
        std::vector<std::string> notes;
        suggest(o.name, {"lr_pot", "lr_dep", "window_ms", "modulator"}, notes);
        err("E0205", o.range, "unknown configure_stdp option '" + o.name + "'", {}, notes);
        if (o.value) expr(o.value);
        continue;
      }
      if (seen.count(o.name)) {
        err("E0205", o.range, "duplicate configure_stdp option '" + o.name + "'");
        if (o.value) expr(o.value);
        continue;
      }
      seen.insert(o.name);
      if (o.name == "lr_pot") hasPot = true;
      if (o.name == "lr_dep") hasDep = true;
      if (o.name == "window_ms") hasWin = true;
      if (o.name == "modulator") {
        modArg = &o;
        // v2.0 3-factor STDP: the modulator must name a declared signal.
        if (o.value && o.value->kind == ExprKind::Ident) {
          if (!res.signalIndex.count(o.value->name)) {
            std::vector<std::string> cands;
            for (const auto& sig : res.signals) cands.push_back(sig.name);
            std::vector<std::string> notes;
            suggest(o.value->name, cands, notes);
            err("E0201", o.value->range,
                "unknown signal '" + o.value->name + "' (modulator must name a "
                "signal declared with 'signal <Name> : float = ...')", {}, notes);
          }
        } else {
          err("E0205", o.value ? o.value->range : o.range,
              "configure_stdp 'modulator' must name a signal");
        }
      } else {
        checkOptNumeric(o, "configure_stdp");
      }
    }

    std::vector<std::string> missing;
    if (!hasPot) missing.push_back("lr_pot");
    if (!hasDep) missing.push_back("lr_dep");
    if (!hasWin) missing.push_back("window_ms");
    if (!missing.empty()) {
      std::string list;
      for (size_t i = 0; i < missing.size(); ++i) {
        if (i) list += ", ";
        list += missing[i];
      }
      err("E0205", s.range, "configure_stdp requires option(s): " + list);
    }

    if (!plasticPairs.count({s.srcName, s.dstName}))
      err("E0209", s.range,
          "configure_stdp requires a plastic connection '" + s.srcName + " -> " +
              s.dstName + "' declared earlier with a connect statement (plastic=true)");

    // Backend flag: a modulated 3-factor STDP exists in the program.
    if (modArg && modArg->value && modArg->value->kind == ExprKind::Ident &&
        res.signalIndex.count(modArg->value->name))
      res.hasModulatedStdp = true;
  }

  // ---- let ----
  void letStmt(Stmt& s, Ctx ctx) {
    TypeInfo initT = TypeInfo::Error();
    if (s.init) initT = expr(s.init);

    TypeInfo declT = initT;
    if (s.typeAnno.k != TypeRef::K::None) {
      TypeInfo annoT = typeRefToInfo(s.typeAnno);
      if (!assignable(annoT, initT))
        err("E0206", s.init ? s.init->range : s.range,
            "type mismatch: cannot initialize '" + s.name + "' of type " +
                annoT.toString() + " with " + initT.toString());
      declT = annoT;
    } else if (s.init && initT.k == TypeInfo::K::Void) {
      err("E0206", s.init->range,
          "type mismatch: cannot initialize '" + s.name + "' with a Void value");
    }

    if (ctx == Ctx::Top) {
      if (!topLetOwner.count(&s))
        return; // duplicate top-level let, already reported in pass 1
      GlobalVarInfo g;
      g.name = s.name;
      g.type = declT;
      g.decl = &s;
      res.globals.push_back(std::move(g));
      scopes[0][s.name] = declT;
    } else {
      auto& cur = scopes.back();
      if (cur.count(s.name))
        err("E0202", s.range, "duplicate declaration of '" + s.name + "' in this scope");
      cur[s.name] = declT;
    }
    res.varTypes[&s] = declT;
  }

  // ---- assign ----
  void assignStmt(Stmt& s) {
    const TypeInfo* varT = lookupVar(s.name);
    if (varT) {
      // ok
    } else if (res.signalIndex.count(s.name)) {
      // v2.0: `set <signal> = expr;` updates the neuromodulator value.
      TypeInfo valT = TypeInfo::Error();
      if (s.init) valT = expr(s.init);
      if (!valT.isError() && !valT.isNumeric())
        err("E0206", s.init ? s.init->range : s.range,
            "type mismatch: signal '" + s.name + "' must be assigned a Float, got " +
                valT.toString());
      res.varTypes[&s] = TypeInfo::Float();
      return;
    } else if (res.findGroup(s.name)) {
      err("E0206", s.range,
          "type mismatch: cannot assign to node group '" + s.name + "'");
    } else if (isBuiltinName(s.name)) {
      err("E0206", s.range,
          "type mismatch: cannot assign to builtin function '" + s.name + "'");
    } else {
      std::vector<std::string> notes;
      suggest(s.name, visibleNames(), notes);
      err("E0201", s.range, "unknown identifier '" + s.name + "'", {}, notes);
    }

    TypeInfo valT = TypeInfo::Error();
    if (s.init) valT = expr(s.init);

    if (varT) {
      if (!assignable(*varT, valT))
        err("E0206", s.init ? s.init->range : s.range,
            "type mismatch: cannot assign " + valT.toString() + " to variable '" +
                s.name + "' of type " + varT->toString());
      res.varTypes[&s] = *varT;
    }
  }

  // ---- print / checkpoint / raster ----
  void printStmt(Stmt& s) {
    for (const ExprPtr& a : s.printArgs) {
      if (!a) continue;
      TypeInfo t = expr(a);
      if (t.k == TypeInfo::K::Void)
        err("E0206", a->range, "type mismatch: cannot print a value of type Void");
    }
  }

  // Parser may attach the path expression to `init` (most likely) or `expr`;
  // fall back to the plain `path` string otherwise.
  const ExprPtr& pathExpr(Stmt& s) const { return s.init ? s.init : s.expr; }

  void saveLoad(Stmt& s) {
    const ExprPtr& p = pathExpr(s);
    if (p) {
      TypeInfo t = expr(p);
      if (!t.isError() && t.k != TypeInfo::K::Str)
        err("E0206", p->range,
            "type mismatch: checkpoint path must be Str, got " + t.toString());
    }
    // else: plain string in s.path (or parser-recovered) — nothing to check.
  }

  void exportRaster(Stmt& s) {
    const std::string& gname = !s.srcName.empty() ? s.srcName : s.name;
    resolveGroup(gname, s.range);
    const ExprPtr& p = pathExpr(s);
    if (p) {
      TypeInfo t = expr(p);
      if (!t.isError() && t.k != TypeInfo::K::Str)
        err("E0206", p->range,
            "type mismatch: export path must be Str, got " + t.toString());
    }
  }

  // ---- control flow ----
  void ifStmt(Stmt& s, Ctx ctx) {
    if (s.cond) {
      TypeInfo t = expr(s.cond);
      if (!t.isError() && t.k != TypeInfo::K::Bool)
        err("E0206", s.cond->range,
            "type mismatch: 'if' condition must be Bool, got " + t.toString());
    }
    Ctx inner = innerCtx(ctx);
    scopes.emplace_back();
    for (const StmtPtr& x : s.body) stmt(x, inner);
    scopes.pop_back();
    scopes.emplace_back();
    for (const StmtPtr& x : s.elseBody) stmt(x, inner);
    scopes.pop_back();
  }

  void forStmt(Stmt& s, Ctx ctx) {
    if (s.lo) {
      TypeInfo t = expr(s.lo);
      if (!t.isError() && t.k != TypeInfo::K::Int)
        err("E0206", s.lo->range,
            "type mismatch: for loop lower bound must be Int, got " + t.toString());
    }
    if (s.hi) {
      TypeInfo t = expr(s.hi);
      if (!t.isError() && t.k != TypeInfo::K::Int)
        err("E0206", s.hi->range,
            "type mismatch: for loop upper bound must be Int, got " + t.toString());
    }
    Ctx inner = innerCtx(ctx);
    scopes.emplace_back();
    scopes.back()[s.loopVar] = TypeInfo::Int();
    for (const StmtPtr& x : s.body) stmt(x, inner);
    scopes.pop_back();
    res.varTypes[&s] = TypeInfo::Int();
  }

  void whileStmt(Stmt& s, Ctx ctx) {
    if (s.cond) {
      TypeInfo t = expr(s.cond);
      if (!t.isError() && t.k != TypeInfo::K::Bool)
        err("E0206", s.cond->range,
            "type mismatch: 'while' condition must be Bool, got " + t.toString());
    }
    Ctx inner = innerCtx(ctx);
    scopes.emplace_back();
    for (const StmtPtr& x : s.body) stmt(x, inner);
    scopes.pop_back();
  }

  // ---- run / at_emit / on_spike ----
  void checkDurationLike(const ExprPtr& e, const char* what) {
    TypeInfo t = expr(e);
    if (!t.isError() && t.k != TypeInfo::K::Duration && !t.isNumeric())
      err("E0206", e->range,
          std::string("type mismatch: run '") + what +
              "' must be Duration or Float, got " + t.toString());
    double v;
    if (constEvalFloat(e.get(), v) && !(v > 0.0))
      err("E0206", e->range, std::string("run '") + what + "' must be positive");
  }

  void runStmt(Stmt& s, Ctx ctx) {
    if (ctx == Ctx::RunBody)
      err("E0211", s.range, "'run' cannot be nested inside another 'run' body");
    else if (ctx == Ctx::Handler)
      err("E0211", s.range, "'run' cannot be nested inside a spike handler");
    else if (ctx == Ctx::ContBody)
      err("E0211", s.range, "'run' cannot be nested inside a run_continuous body");
    // Ctx::Top and Ctx::Block (e.g. inside a for loop) are fine.

    // E0215: run() and run_continuous() cannot be mixed in one program.
    if (!mixReported && sawContinuous) {
      mixReported = true;
      err("E0215", s.range,
          "run() and run_continuous() cannot be mixed: a program is either "
          "discrete-time (run) or continuous-time (run_continuous)");
    }
    sawRun = true;

    if (!s.hasDuration || !s.durationExpr)
      err("E0205", s.range, "run requires a 'duration' option");
    else
      checkDurationLike(s.durationExpr, "duration");
    if (s.hasDt && s.dtExpr)
      checkDurationLike(s.dtExpr, "dt");

    for (const StmtPtr& x : s.body) stmt(x, Ctx::RunBody);
  }

  void atEmit(Stmt& s, Ctx ctx) {
    if (ctx == Ctx::Handler)
      err("E0214", s.range, "spike handler bodies cannot contain 'at ... emit' statements");
    else if (ctx != Ctx::RunBody)
      err("E0212", s.range, "'at ... emit' is only allowed directly inside a run body");

    const GroupInfo* g = resolveGroup(s.name, s.range);
    bool sizeKnown = g && !badSize.count(g->name) && g->size > 0;

    if (s.timeExpr) {
      TypeInfo t = expr(s.timeExpr);
      if (!t.isError() && t.k != TypeInfo::K::Duration && !t.isNumeric())
        err("E0206", s.timeExpr->range,
            "type mismatch: 'at' time must be Duration or Float, got " + t.toString());
    }

    auto typeCheckInt = [&](const ExprPtr& e, const char* what) {
      if (!e) return;
      TypeInfo t = expr(e);
      if (!t.isError() && t.k != TypeInfo::K::Int)
        err("E0206", e->range,
            std::string("type mismatch: slice ") + what + " must be Int, got " +
                t.toString());
    };

    if (s.sliceHasIndex) {
      typeCheckInt(s.sliceLo, "index");
      int64_t iv;
      if (sizeKnown && s.sliceLo && constEvalInt(s.sliceLo.get(), iv) &&
          (iv < 0 || static_cast<uint64_t>(iv) >= g->size))
        err("E0203", s.sliceLo->range,
            "slice index " + std::to_string(iv) + " is out of bounds for group '" +
                g->name + "' (size " + std::to_string(g->size) + ")");
    } else if (s.sliceHasRange) {
      typeCheckInt(s.sliceLo, "lower bound");
      typeCheckInt(s.sliceHi, "upper bound");
      int64_t lo = 0, hi = 0;
      bool loC = s.sliceLo && constEvalInt(s.sliceLo.get(), lo);
      bool hiC = s.sliceHi && constEvalInt(s.sliceHi.get(), hi);
      if (loC && hiC && lo > hi)
        warn("W0201", s.sliceRange,
             "slice range " + std::to_string(lo) + ".." + std::to_string(hi) +
                 " is empty (lo > hi)");
      if (sizeKnown) {
        if (loC && (lo < 0 || static_cast<uint64_t>(lo) > g->size))
          err("E0203", s.sliceLo->range,
              "slice lower bound " + std::to_string(lo) + " is out of bounds for group '" +
                  g->name + "' (size " + std::to_string(g->size) + ")");
        if (hiC && (hi < 0 || static_cast<uint64_t>(hi) > g->size))
          err("E0203", s.sliceHi->range,
              "slice upper bound " + std::to_string(hi) + " is out of bounds for group '" +
                  g->name + "' (size " + std::to_string(g->size) + ")");
      }
    }

    if (!s.currentExpr)
      err("E0205", s.range, "at_emit requires a 'current' option");
    else {
      TypeInfo t = expr(s.currentExpr);
      if (!t.isError() && !t.isNumeric())
        err("E0206", s.currentExpr->range,
            "type mismatch: 'current' must be Float, got " + t.toString());
    }
  }

  void onSpike(Stmt& s, Ctx ctx) {
    if (ctx == Ctx::Handler)
      err("E0214", s.range, "'on_spike' cannot be nested inside another spike handler");
    else if (ctx != Ctx::Top && ctx != Ctx::ContBody)
      err("E0213", s.range, "'on_spike' is only allowed at top level");

    resolveGroup(s.name, s.range);

    scopes.emplace_back();
    scopes.back()["neuron_index"] = TypeInfo::Int();
    scopes.back()["spike_time"] = TypeInfo::Float();
    for (const StmtPtr& x : s.body) stmt(x, Ctx::Handler);
    scopes.pop_back();
  }

  // ==========================================================================
  // v2.0 — asynchronous continuous-time environment
  // ==========================================================================

  // Top-level-only declarations report E0207 when nested.
  bool checkTopOnly(Stmt& s, Ctx ctx, const char* kw) {
    if (ctx == Ctx::Handler || ctx == Ctx::Block || ctx == Ctx::ContBody ||
        ctx == Ctx::BlockNoTop) {
      err("E0207", s.range, std::string("'") + kw + "' is only allowed at top level");
      return false;
    }
    return ctx == Ctx::Top;
  }

  void signalDecl(Stmt& s, Ctx ctx) {
    const bool topLevel = checkTopOnly(s, ctx, "signal");
    (void)topLevel; // pass 1 owns registration; duplicates already reported
    if (s.init) {
      TypeInfo t = expr(s.init);
      if (!t.isError() && !t.isNumeric())
        err("E0206", s.init->range,
            "type mismatch: signal '" + s.name +
                "' must be initialized with a Float, got " + t.toString());
    }
    // Signals are readable as Float everywhere after this point.
    if (topLevel) scopes[0][s.name] = TypeInfo::Float();
  }

  void oscillatorDecl(Stmt& s, Ctx ctx) {
    checkTopOnly(s, ctx, "oscillator");
    res.hasOscillators = true;

    std::unordered_set<std::string> seen;
    bool hasFreq = false, hasAmp = false, hasTarget = false;
    for (const CallArg& o : s.opts) {
      if (!o.named) {
        err("E0205", o.range, "oscillator parameters must be named "
                               "(frequency=..., amplitude=..., target=...)");
        if (o.value) expr(o.value);
        continue;
      }
      if (o.name != "frequency" && o.name != "amplitude" && o.name != "target" &&
          o.name != "phase") {
        std::vector<std::string> notes;
        suggest(o.name, {"frequency", "amplitude", "target", "phase"}, notes);
        err("E0205", o.range, "unknown oscillator parameter '" + o.name + "'", {}, notes);
        if (o.value) expr(o.value);
        continue;
      }
      if (!seen.insert(o.name).second) {
        err("E0205", o.range, "duplicate oscillator parameter '" + o.name + "'");
        if (o.value) expr(o.value);
        continue;
      }
      if (o.name == "target") {
        hasTarget = true;
        if (o.value && o.value->kind == ExprKind::Ident) {
          resolveGroup(o.value->name, o.value->range);
        } else {
          err("E0205", o.range, "oscillator 'target' must be a node group name");
        }
      } else {
        if (o.name == "frequency") hasFreq = true;
        if (o.name == "amplitude") hasAmp = true;
        checkOptNumeric(o, "oscillator");
        if (o.name == "frequency" && o.value) {
          double f;
          if (constEvalFloat(o.value.get(), f) && !(f > 0.0))
            err("E0205", o.value->range, "oscillator 'frequency' must be positive");
        }
      }
    }
    if (!hasFreq || !hasAmp || !hasTarget) {
      std::vector<std::string> missing;
      if (!hasFreq) missing.push_back("frequency");
      if (!hasAmp) missing.push_back("amplitude");
      if (!hasTarget) missing.push_back("target");
      std::string list;
      for (size_t i = 0; i < missing.size(); ++i) {
        if (i) list += ", ";
        list += missing[i];
      }
      err("E0205", s.range, "oscillator '" + s.name + "' is missing parameter(s): " + list);
    }
  }

  void externalStreamDecl(Stmt& s, Ctx ctx) {
    checkTopOnly(s, ctx, "external stream");
    res.hasStreams = true;
    if (s.sizeExpr) expr(s.sizeExpr);
  }

  void bindInputStream(Stmt& s, Ctx ctx) {
    // Legal at top level and inside run_continuous bodies (live re-binding).
    if (ctx == Ctx::RunBody)
      err("E0210", s.range, "bind_input_stream is not allowed directly inside a run body");

    resolveGroup(s.dstName, s.range);
    auto si = res.streamIndex.find(s.srcName);
    if (si == res.streamIndex.end()) {
      std::vector<std::string> cands;
      for (const auto& st : res.streams) cands.push_back(st.name);
      std::vector<std::string> notes;
      suggest(s.srcName, cands, notes);
      err("E0201", s.range, "unknown external stream '" + s.srcName + "'", {}, notes);
    }

    std::unordered_set<std::string> seen;
    for (const CallArg& o : s.opts) {
      if (!o.named) {
        err("E0205", o.range, "bind_input_stream options must be named "
                               "(encoding=..., max_freq=...)");
        if (o.value) expr(o.value);
        continue;
      }
      if (o.name != "encoding" && o.name != "max_freq" && o.name != "kick") {
        std::vector<std::string> notes;
        suggest(o.name, {"encoding", "max_freq", "kick"}, notes);
        err("E0205", o.range, "unknown bind_input_stream option '" + o.name + "'", {}, notes);
        if (o.value) expr(o.value);
        continue;
      }
      if (!seen.insert(o.name).second) {
        err("E0205", o.range, "duplicate bind_input_stream option '" + o.name + "'");
        if (o.value) expr(o.value);
        continue;
      }
      if (o.name == "encoding") {
        if (!o.value || o.value->kind != ExprKind::Ident || o.value->name != "Poisson") {
          err("E0205", o.value ? o.value->range : o.range,
              "bind_input_stream 'encoding' must be Poisson (the only v2.0 encoder)");
        }
      } else { // max_freq | kick
        checkOptNumeric(o, "bind_input_stream");
        if (o.value) {
          double f;
          if (constEvalFloat(o.value.get(), f) && o.name == "max_freq" && !(f > 0.0))
            err("E0205", o.value->range, "bind_input_stream 'max_freq' must be positive");
        }
      }
    }
  }

  void runContinuousStmt(Stmt& s, Ctx ctx) {
    if (ctx != Ctx::Top)
      err("E0207", s.range, "'run_continuous' is only allowed at top level");

    // E0215: run() and run_continuous() cannot be mixed in one program.
    if (!mixReported && (sawRun || sawContinuous)) {
      mixReported = true;
      err("E0215", s.range,
          "run() and run_continuous() cannot be mixed: a program is either "
          "discrete-time (run) or continuous-time (run_continuous)");
    }
    sawContinuous = true;
    res.hasContinuous = true;

    if (s.hasDt && s.dtExpr) checkDurationLike(s.dtExpr, "dt");

    for (const StmtPtr& x : s.body) stmt(x, Ctx::ContBody);
  }

  // stop_continuous — legal wherever the loop can be running from.
  void stopContinuousStmt(Stmt& s, Ctx ctx) {
    if (ctx == Ctx::RunBody)
      err("E0210", s.range, "stop_continuous is not allowed directly inside a run body");
  }

  // wait_continuous — main-thread only (waiting inside the loop or a handler
  // would deadlock by construction).
  void waitContinuousStmt(Stmt& s, Ctx ctx) {
    if (ctx == Ctx::RunBody || ctx == Ctx::ContBody || ctx == Ctx::Handler)
      err("E0210", s.range,
          "wait_continuous blocks the main thread and is not allowed inside "
          "run/handler/continuous bodies");
  }

  // set_plasticity / prune_weights — structural plasticity commands.
  void structuralStmt(Stmt& s, Ctx ctx, bool isPrune) {
    const char* kw = isPrune ? "prune_weights" : "set_plasticity";
    if (ctx == Ctx::RunBody)
      err("E0210", s.range,
          std::string(kw) + " is not allowed directly inside a run body");

    resolveGroup(s.srcName, s.range);
    resolveGroup(s.dstName, s.range);

    if (isPrune) res.hasPrune = true;

    std::unordered_set<std::string> seen;
    bool hasRequired = false;
    for (const CallArg& o : s.opts) {
      if (!o.named) {
        err("E0205", o.range,
            std::string(kw) + " options must be named");
        if (o.value) expr(o.value);
        continue;
      }
      const char* req = isPrune ? "threshold" : "enabled";
      if (o.name != req) {
        std::vector<std::string> notes;
        suggest(o.name, {req}, notes);
        err("E0205", o.range, "unknown " + std::string(kw) + " option '" + o.name + "'",
            {}, notes);
        if (o.value) expr(o.value);
        continue;
      }
      if (!seen.insert(o.name).second) {
        err("E0205", o.range,
            "duplicate " + std::string(kw) + " option '" + o.name + "'");
        if (o.value) expr(o.value);
        continue;
      }
      if (isPrune) {
        hasRequired = true;
        checkOptNumeric(o, kw);
      } else {
        hasRequired = true;
        if (o.value) {
          TypeInfo t = expr(o.value);
          if (!t.isError() && t.k != TypeInfo::K::Bool)
            err("E0206", o.value->range,
                "type mismatch: set_plasticity 'enabled' must be Bool, got " +
                    t.toString());
        }
      }
    }
    if (!hasRequired) {
      err("E0205", s.range,
          std::string(isPrune ? "prune_weights requires a 'threshold' option"
                              : "set_plasticity requires an 'enabled' option"));
    }

    // The connection must have been declared earlier (any plasticity flag).
    if (!allConnPairs.count({s.srcName, s.dstName}))
      err("E0216", s.range,
          std::string(kw) + " requires a connection '" + s.srcName + " -> " +
              s.dstName + "' declared earlier with a connect statement");
  }

  // ==========================================================================
  // Expressions
  // ==========================================================================
  TypeInfo expr(const ExprPtr& e) { return e ? expr(*e) : TypeInfo::Error(); }

  TypeInfo expr(const Expr& e) {
    TypeInfo t = exprImpl(e);
    res.exprTypes[&e] = t;
    return t;
  }

  TypeInfo exprImpl(const Expr& e) {
    switch (e.kind) {
      case ExprKind::IntLit:      return TypeInfo::Int();
      case ExprKind::FloatLit:    return TypeInfo::Float();
      case ExprKind::DurationLit: return TypeInfo::Duration();
      case ExprKind::StringLit:   return TypeInfo::Str();
      case ExprKind::BoolLit:     return TypeInfo::Bool();
      case ExprKind::DeviceLit:   return TypeInfo::Error(); // only in run(device = ...)
      case ExprKind::Ident:  return identExpr(e);
      case ExprKind::Unary:  return unaryExpr(e);
      case ExprKind::Binary: return binaryExpr(e);
      case ExprKind::Call:   return callExpr(e);
    }
    return TypeInfo::Error();
  }

  TypeInfo identExpr(const Expr& e) {
    if (const TypeInfo* v = lookupVar(e.name)) return *v;
    if (res.findGroup(e.name)) return TypeInfo::Group();
    if (res.signalIndex.count(e.name)) return TypeInfo::Float(); // v2.0 signal read
    if (isBuiltinName(e.name)) {
      err("E0206", e.range,
          "type mismatch: builtin function '" + e.name + "' cannot be used as a value");
      return TypeInfo::Error();
    }
    std::vector<std::string> notes;
    suggest(e.name, visibleNames(), notes);
    err("E0201", e.range, "unknown identifier '" + e.name + "'", {}, notes);
    return TypeInfo::Error();
  }

  TypeInfo unaryExpr(const Expr& e) {
    TypeInfo t = expr(e.lhs);
    if (t.isError()) return TypeInfo::Error();
    if (e.op == Tok::Not) {
      if (t.k == TypeInfo::K::Bool) return TypeInfo::Bool();
      err("E0206", e.range,
          "type mismatch: operator '!' requires Bool, got " + t.toString());
      return TypeInfo::Error();
    }
    if (e.op == Tok::Minus) {
      if (t.isNumeric()) return t;
      if (t.k == TypeInfo::K::Duration) return TypeInfo::Duration();
      err("E0206", e.range,
          "type mismatch: unary '-' requires a numeric operand, got " + t.toString());
      return TypeInfo::Error();
    }
    return TypeInfo::Error();
  }

  TypeInfo binaryExpr(const Expr& e) {
    TypeInfo l = expr(e.lhs);
    TypeInfo r = expr(e.rhs);
    switch (e.op) {
      case Tok::AndAnd:
      case Tok::OrOr: {
        if (l.isError() || r.isError()) return TypeInfo::Error();
        if (l.k == TypeInfo::K::Bool && r.k == TypeInfo::K::Bool) return TypeInfo::Bool();
        err("E0206", e.range,
            std::string("type mismatch: operator '") + opSpelling(e.op) +
                "' requires Bool operands, got " + l.toString() + " and " + r.toString());
        return TypeInfo::Error();
      }
      case Tok::EqEq:
      case Tok::NotEq:
      case Tok::Lt:
      case Tok::Gt:
      case Tok::Le:
      case Tok::Ge:
        return compareExpr(e, l, r);
      case Tok::Plus:
      case Tok::Minus:
      case Tok::Star:
      case Tok::Slash:
      case Tok::Percent:
        return arithExpr(e, l, r);
      default:
        return TypeInfo::Error();
    }
  }

  // Numeric result type after Int→Float promotion.
  TypeInfo promotedNumeric(const TypeInfo& l, const TypeInfo& r) const {
    return (l.k == TypeInfo::K::Float || r.k == TypeInfo::K::Float) ? TypeInfo::Float()
                                                                    : TypeInfo::Int();
  }

  TypeInfo arithExpr(const Expr& e, const TypeInfo& l, const TypeInfo& r) {
    if (l.isError() || r.isError()) return TypeInfo::Error();
    bool ld = l.k == TypeInfo::K::Duration, rd = r.k == TypeInfo::K::Duration;
    bool ln = l.isNumeric(), rn = r.isNumeric();
    switch (e.op) {
      case Tok::Plus:
        if (ld && rd) return TypeInfo::Duration();
        if (ln && rn) return promotedNumeric(l, r);
        if (l.k == TypeInfo::K::Str && r.k == TypeInfo::K::Str) return TypeInfo::Str();
        break;
      case Tok::Minus:
        if (ld && rd) return TypeInfo::Duration();
        if (ln && rn) return promotedNumeric(l, r);
        break;
      case Tok::Star:
        if ((ld && rn) || (ln && rd)) return TypeInfo::Duration();
        if (ln && rn) return promotedNumeric(l, r);
        break;
      case Tok::Slash:
        if (ld && rn) return TypeInfo::Duration();
        if (ld && rd) return TypeInfo::Float(); // duration ratio
        if (ln && rn) return promotedNumeric(l, r);
        break;
      case Tok::Percent:
        if (l.k == TypeInfo::K::Int && r.k == TypeInfo::K::Int) return TypeInfo::Int();
        break;
      default:
        return TypeInfo::Error();
    }
    err("E0206", e.range,
        std::string("type mismatch: cannot apply '") + opSpelling(e.op) + "' to " +
            l.toString() + " and " + r.toString());
    return TypeInfo::Error();
  }

  TypeInfo compareExpr(const Expr& e, const TypeInfo& l, const TypeInfo& r) {
    if (l.isError() || r.isError()) return TypeInfo::Error();
    bool eq = e.op == Tok::EqEq || e.op == Tok::NotEq;
    if (l.isNumeric() && r.isNumeric()) return TypeInfo::Bool();
    if (l.k == TypeInfo::K::Duration && r.k == TypeInfo::K::Duration)
      return TypeInfo::Bool();
    if (eq && l.k == r.k && (l.k == TypeInfo::K::Str || l.k == TypeInfo::K::Bool))
      return TypeInfo::Bool();
    err("E0206", e.range,
        std::string("type mismatch: cannot compare ") + l.toString() + " and " +
            r.toString() + " with '" + opSpelling(e.op) + "'");
    return TypeInfo::Error();
  }

  // ---- builtin calls ----
  TypeInfo arityErr(const Expr& e, size_t n) {
    err("E0206", e.range,
        "call to '" + e.name + "' expects " + std::to_string(n) + " argument(s), got " +
            std::to_string(e.args.size()));
    return TypeInfo::Error();
  }

  TypeInfo argTypeErr(const Expr& e, size_t idx, const char* want, const TypeInfo& got) {
    if (!got.isError())
      err("E0206", e.range,
          "type mismatch: call to '" + e.name + "': argument " + std::to_string(idx) +
              " must be " + want + ", got " + got.toString());
    return TypeInfo::Error();
  }

  TypeInfo callExpr(const Expr& e) {
    const std::string& fn = e.name;

    // stream_set(streamName, index: Int, value: Float) -> Void
    // Special-cased BEFORE the generic argument walk: argument 1 names a
    // stream and must not be evaluated as a value expression.
    if (fn == "stream_set") {
      if (e.args.size() != 3)
        return arityErr(e, 3);
      for (const CallArg& a : e.args)
        if (a.named) {
          err("E0206", e.range,
              "call to 'stream_set': arguments must be positional");
          return TypeInfo::Error();
        }
      const Expr* first = e.args[0].value.get();
      if (!first || first->kind != ExprKind::Ident ||
          !res.streamIndex.count(first->name)) {
        err("E0201", first ? first->range : e.range,
            "stream_set: argument 1 must be an external stream name");
        return TypeInfo::Error();
      }
      TypeInfo i1 = expr(e.args[1].value);
      if (!i1.isError() && i1.k != TypeInfo::K::Int)
        return argTypeErr(e, 2, "Int", i1);
      TypeInfo i2 = expr(e.args[2].value);
      if (!i2.isError() && !i2.isNumeric())
        return argTypeErr(e, 3, "Float", i2);
      return TypeInfo::Void();
    }

    std::vector<TypeInfo> at;
    at.reserve(e.args.size());
    bool hasNamed = false;
    for (const CallArg& a : e.args) {
      if (a.named) hasNamed = true;
      at.push_back(expr(a.value));
    }

    if (hasNamed) {
      err("E0206", e.range,
          "call to '" + fn + "': builtin function arguments must be positional");
      return TypeInfo::Error();
    }

    if (fn == "sin" || fn == "cos" || fn == "exp" || fn == "log" || fn == "sqrt") {
      if (at.size() != 1) return arityErr(e, 1);
      if (!at[0].isNumeric()) return argTypeErr(e, 1, "Float", at[0]);
      return TypeInfo::Float();
    }
    if (fn == "random_uniform" || fn == "random_gaussian") {
      if (at.size() != 2) return arityErr(e, 2);
      if (!at[0].isNumeric()) return argTypeErr(e, 1, "Float", at[0]);
      if (!at[1].isNumeric()) return argTypeErr(e, 2, "Float", at[1]);
      return TypeInfo::Float();
    }
    if (fn == "get_weight") {
      if (at.size() != 4) return arityErr(e, 4);
      if (at[0].k != TypeInfo::K::Group) return argTypeErr(e, 1, "Group", at[0]);
      if (at[1].k != TypeInfo::K::Group) return argTypeErr(e, 2, "Group", at[1]);
      if (at[2].k != TypeInfo::K::Int)   return argTypeErr(e, 3, "Int", at[2]);
      if (at[3].k != TypeInfo::K::Int)   return argTypeErr(e, 4, "Int", at[3]);
      return TypeInfo::Float();
    }
    if (fn == "tensor_new" || fn == "tensor_new2" || fn == "tensor_new3") {
      size_t n = fn == "tensor_new" ? 1 : fn == "tensor_new2" ? 2 : 3;
      if (at.size() != n) return arityErr(e, n);
      for (size_t i = 0; i < n; ++i)
        if (at[i].k != TypeInfo::K::Int) return argTypeErr(e, i + 1, "Int", at[i]);
      TypeInfo t = TypeInfo::Tensor();
      for (size_t i = 0; i < n; ++i) {
        int64_t d;
        if (constEvalInt(e.args[i].value.get(), d) && d > 0)
          t.tensorDims.push_back(static_cast<uint64_t>(d));
        else {
          t.tensorDims.clear(); // dims not statically known
          break;
        }
      }
      return t;
    }
    if (fn == "tensor_get" || fn == "tensor_get2" || fn == "tensor_get3") {
      size_t n = fn == "tensor_get" ? 2 : fn == "tensor_get2" ? 3 : 4;
      if (at.size() != n) return arityErr(e, n);
      if (at[0].k != TypeInfo::K::Tensor) return argTypeErr(e, 1, "Tensor", at[0]);
      for (size_t i = 1; i < n; ++i)
        if (at[i].k != TypeInfo::K::Int) return argTypeErr(e, i + 1, "Int", at[i]);
      return TypeInfo::Float();
    }
    if (fn == "tensor_set" || fn == "tensor_set2" || fn == "tensor_set3") {
      size_t n = fn == "tensor_set" ? 3 : fn == "tensor_set2" ? 4 : 5;
      if (at.size() != n) return arityErr(e, n);
      if (at[0].k != TypeInfo::K::Tensor) return argTypeErr(e, 1, "Tensor", at[0]);
      for (size_t i = 1; i + 1 < n; ++i)
        if (at[i].k != TypeInfo::K::Int) return argTypeErr(e, i + 1, "Int", at[i]);
      if (!at[n - 1].isNumeric()) return argTypeErr(e, n, "Float", at[n - 1]);
      return TypeInfo::Void();
    }
    if (fn == "dump_tensor") {
      if (at.size() != 2) return arityErr(e, 2);
      if (at[0].k != TypeInfo::K::Str)    return argTypeErr(e, 1, "Str", at[0]);
      if (at[1].k != TypeInfo::K::Tensor) return argTypeErr(e, 2, "Tensor", at[1]);
      return TypeInfo::Void();
    }

    // ---- v2.0 — readout & action decoder (RSOL) ----
    if (fn == "get_membrane_potentials") {
      if (at.size() != 1) return arityErr(e, 1);
      if (at[0].k != TypeInfo::K::Group) return argTypeErr(e, 1, "Group", at[0]);
      TypeInfo t = TypeInfo::Tensor();
      // Dims known when the group size is a compile-time constant.
      if (e.args[0].value && e.args[0].value->kind == ExprKind::Ident) {
        const GroupInfo* g = res.findGroup(e.args[0].value->name);
        if (g && !badSize.count(g->name) && g->size > 0)
          t.tensorDims.push_back(g->size);
      }
      return t;
    }
    if (fn == "predict_linear") {
      // out[j] = sum_i w[j, i] * x[i];  x: Tensor[N], w: Tensor[M,N] -> Tensor[M]
      if (at.size() != 2) return arityErr(e, 2);
      if (at[0].k != TypeInfo::K::Tensor) return argTypeErr(e, 1, "Tensor", at[0]);
      if (at[1].k != TypeInfo::K::Tensor) return argTypeErr(e, 2, "Tensor", at[1]);
      TypeInfo t = TypeInfo::Tensor();
      const auto& xd = at[0].tensorDims;
      const auto& wd = at[1].tensorDims;
      if (wd.size() == 2 && wd[1] > 0) {
        t.tensorDims.push_back(wd[0]);
        if (xd.size() == 1 && xd[0] > 0 && xd[0] != wd[1])
          err("E0206", e.range,
              "type mismatch: predict_linear input length (" +
                  std::to_string(xd[0]) + ") does not match weight matrix columns (" +
                  std::to_string(wd[1]) + ")");
      }
      return t;
    }
    if (fn == "vector_l2_norm") {
      if (at.size() != 1) return arityErr(e, 1);
      if (at[0].k != TypeInfo::K::Tensor) return argTypeErr(e, 1, "Tensor", at[0]);
      return TypeInfo::Float();
    }
    // Unknown function.
    std::vector<std::string> cands = builtinNames();
    for (const auto& sc : scopes)
      for (const auto& kv : sc) cands.push_back(kv.first);
    std::vector<std::string> notes;
    suggest(fn, cands, notes);
    err("E0201", e.range, "unknown function '" + fn + "'", {}, notes);
    return TypeInfo::Error();
  }

  // ==========================================================================
  // Entry
  // ==========================================================================
  SemaResult run() {
    runPass1();
    scopes.emplace_back(); // scopes[0] = global scope
    runPass2();
    return std::move(res);
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Sema entry points
// ---------------------------------------------------------------------------

Sema::Sema(Program& program, DiagnosticEngine& diag) : prog_(program), diag_(diag) {}

SemaResult Sema::run() {
  Impl impl(prog_, diag_);
  res_ = impl.run();
  return res_;
}

} // namespace ndl
