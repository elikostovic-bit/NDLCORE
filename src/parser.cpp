// NDL v1.0 — Recursive descent parser (implementation of include/ndl/parser.hpp).
//
// Grammar (docs/INTERNALS.md §3, reconstructed):
//   module       := top_item* EOF
//   top_item     := import_decl | use_decl | node_group_decl | stmt
//   import_decl  := 'import' STRING ';'
//   use_decl     := 'use' IDENT ';'
//   node_group   := 'node_group' IDENT '[' expr ']' ':' ('Excitatory'|'Inhibitory')
//                   '(' [arg {',' arg}] ')' ';'
//   connect      := ('dense_connect'|'sparse_connect'|'one_to_one_connect')
//                   '(' [arg {',' arg}] ')' ';'
//   stdp         := 'configure_stdp' '(' IDENT '->' IDENT {',' IDENT '=' expr} ')' ';'
//   stmt         := let | print | save_checkpoint | load_checkpoint | export_raster
//                 | if | for | while | run | at_emit | on_spike | simple
//   let          := 'let' IDENT [':' type] '=' expr ';'
//   print        := 'print' '(' [expr {',' expr}] ')' ';'
//   save/load    := ('save_checkpoint'|'load_checkpoint') '(' expr ')' ';'
//   export       := 'export_raster' '(' IDENT ',' expr ')' ';'
//   if           := 'if' '(' expr ')' block ['else' (if | block)]
//   for          := 'for' '(' IDENT 'in' expr '..' expr ')' block
//   while        := 'while' '(' expr ')' block
//   run          := 'run' '(' run_param {',' run_param} ')' block
//   run_param    := ('duration'|'dt') '=' expr | 'device' '=' ('CPU'|'GPU')
//   at_emit      := 'at' expr 'emit' IDENT ['[' (expr | expr '..' expr) ']']
//                   '(' 'current' '=' expr ')' ';'
//   on_spike     := 'on_spike' '(' IDENT ')' block
//   simple       := IDENT '=' expr ';'   (Assign)
//                 | expr ';'             (ExprStmt)
//   expr         := '||' < '&&' < ('=='|'!=') < ('<'|'>'|'<='|'>=')
//                   < ('+'|'-') < ('*'|'/'|'%') < unary < primary   (left-assoc)
//   primary      := INT | FLOAT | DURATION | STRING | 'true' | 'false'
//                 | 'CPU' | 'GPU' | IDENT ['(' args ')'] | '(' expr ')'
//
// Error strategy: every syntax error is reported as E0100
// ("expected <X>, found <Y>"; the expect() context is attached as a note).
// expect() never throws; on mismatch the statement parser panics via
// synchronize(), which skips to the next ';' (consumed), '}' or statement
// start, so the whole file is always parsed and the maximal number of
// independent errors is collected.
//
// Documented conventions for downstream passes:
//   * SaveCheckpoint / LoadCheckpoint / ExportRaster keep their path
//     expression in Stmt::init (the AST has no dedicated path field).
//   * ExportRaster::srcName is the group identifier; its path expr is in init.
//   * Run::body contains only AtEmit statements (extra statements are
//     reported as E0100 and dropped, per the ast.hpp field table).
//   * Extra positional arguments of connect(...) beyond src/dst are kept in
//     Connect::opts as unnamed CallArgs so sema can reject them.

#include "ndl/parser.hpp"

#include <algorithm>
#include <utility>

namespace ndl {
namespace {

// ---------------------------------------------------------------------------
// Small helpers (anonymous namespace)
// ---------------------------------------------------------------------------

ExprPtr makeExpr(ExprKind k, SourceRange r) {
  auto e = std::make_unique<Expr>();
  e->kind = k;
  e->range = std::move(r);
  return e;
}

StmtPtr makeStmt(StmtKind k, const Token& first) {
  auto s = std::make_unique<Stmt>();
  s->kind = k;
  s->range = SourceRange(first.range.begin, first.range.end);
  return s;
}

// End location of the last consumed token (callers guarantee pos > 0 or that
// the token vector is non-empty).
SourceLoc prevEnd(const std::vector<Token>& toks, size_t pos) {
  if (pos > 0 && pos <= toks.size()) return toks[pos - 1].range.end;
  if (!toks.empty()) return toks[0].range.end;
  return SourceLoc();
}

// Range of a statement: kw..';' on success, kw..last-consumed-token otherwise.
SourceRange endRange(const std::vector<Token>& toks, size_t pos,
                     const Token& first, const Token& semi) {
  if (semi.kind == Tok::Semi) return SourceRange(first.range.begin, semi.range.end);
  return SourceRange(first.range.begin, prevEnd(toks, pos));
}

std::string spelling(Tok t) { return std::string(tokSpelling(t)); }

const char* exprKindName(ExprKind k) {
  switch (k) {
  case ExprKind::IntLit: return "integer literal";
  case ExprKind::FloatLit: return "float literal";
  case ExprKind::DurationLit: return "duration literal";
  case ExprKind::StringLit: return "string literal";
  case ExprKind::BoolLit: return "boolean literal";
  case ExprKind::DeviceLit: return "device literal";
  case ExprKind::Ident: return "identifier";
  case ExprKind::Unary: return "unary expression";
  case ExprKind::Binary: return "binary expression";
  case ExprKind::Call: return "call expression";
  }
  return "expression";
}

} // namespace

// ---------------------------------------------------------------------------
// Construction / token access
// ---------------------------------------------------------------------------

Parser::Parser(std::string file, std::vector<Token> tokens, DiagnosticEngine& diag)
    : file_(std::move(file)), toks_(std::move(tokens)), diag_(diag) {
  // Guarantee: at least one token, stream always terminated by Eof.
  if (toks_.empty()) toks_.emplace_back();
  if (toks_.back().kind != Tok::Eof) toks_.emplace_back(); // default Token is Eof
}

const Token& Parser::cur() const {
  const size_t i = pos_ < toks_.size() ? pos_ : toks_.size() - 1;
  return toks_[i];
}

const Token& Parser::peek(size_t n) const {
  size_t i = pos_ + n;
  if (i >= toks_.size()) i = toks_.size() - 1;
  return toks_[i];
}

const Token& Parser::advance() {
  const Token& t = cur();
  if (pos_ + 1 < toks_.size()) ++pos_; // never step past the final Eof
  return t;
}

bool Parser::check(Tok k) const { return cur().kind == k; }

bool Parser::match(Tok k) {
  if (!check(k)) return false;
  advance();
  return true;
}

const Token& Parser::expect(Tok k, const std::string& context) {
  if (check(k)) return advance();
  const std::string msg = "expected " + spelling(k) + ", found " + spelling(cur().kind);
  if (context.empty())
    diag_.error("E0100", cur().range, msg);
  else
    diag_.error("E0100", cur().range, msg, {}, {context});
  return cur();
}

void Parser::parseError(const std::string& msg, SourceRange range) {
  diag_.error("E0100", range, msg);
}

bool Parser::atStmtStart() const {
  switch (cur().kind) {
  case Tok::Kw_node_group:
  case Tok::Kw_dense_connect:
  case Tok::Kw_sparse_connect:
  case Tok::Kw_one_to_one_connect:
  case Tok::Kw_configure_stdp:
  case Tok::Kw_run:
  case Tok::Kw_at:
  case Tok::Kw_on_spike:
  case Tok::Kw_if:
  case Tok::Kw_for:
  case Tok::Kw_while:
  case Tok::Kw_let:
  case Tok::Kw_print:
  case Tok::Kw_save_checkpoint:
  case Tok::Kw_load_checkpoint:
  case Tok::Kw_export_raster:
  case Tok::Kw_import:
  case Tok::Kw_use:
  case Tok::Kw_signal:
  case Tok::Kw_oscillator:
  case Tok::Kw_external:
  case Tok::Kw_run_continuous:
  case Tok::Kw_stop_continuous:
  case Tok::Kw_wait_continuous:
  case Tok::Kw_set_plasticity:
  case Tok::Kw_prune_weights:
  case Tok::Kw_bind_input_stream:
    return true;
  default:
    return false;
  }
}

void Parser::synchronize() {
  // Panic mode: skip tokens until a recovery point. A ';' is consumed so the
  // next statement starts clean; '}' and statement starts are left in place.
  while (!check(Tok::Eof)) {
    if (check(Tok::Semi)) {
      advance();
      return;
    }
    if (check(Tok::RBrace)) return;
    if (atStmtStart()) return;
    advance();
  }
}

// ---------------------------------------------------------------------------
// Module / dispatch
// ---------------------------------------------------------------------------

std::unique_ptr<Module> Parser::parse() {
  auto m = std::make_unique<Module>();
  while (!check(Tok::Eof)) {
    const size_t before = pos_;
    if (!parseTopItem(*m)) break;
    // Progress guarantee: never loop forever on a token nobody consumed.
    if (pos_ == before && !check(Tok::Eof)) advance();
  }
  return m;
}

bool Parser::parseTopItem(Module& m) {
  if (check(Tok::Eof)) return false;
  StmtPtr s;
  switch (cur().kind) {
  case Tok::Kw_import:     s = parseImport(); break;
  case Tok::Kw_use:        s = parseUse(); break;
  case Tok::Kw_node_group: s = parseNodeGroup(); break;
  default:                 s = parseBlockStmt(); break;
  }
  if (s) m.items.push_back(std::move(s));
  return true;
}

// Dispatch for statements (used inside blocks and, via parseTopItem, at the
// top level for everything that is not import/use/node_group).
StmtPtr Parser::parseBlockStmt() {
  switch (cur().kind) {
  case Tok::Kw_dense_connect:       return parseConnect(ConnectKind::Dense);
  case Tok::Kw_sparse_connect:      return parseConnect(ConnectKind::Sparse);
  case Tok::Kw_one_to_one_connect:  return parseConnect(ConnectKind::OneToOne);
  case Tok::Kw_configure_stdp:      return parseStdp();
  case Tok::Kw_let:                 return parseLet();
  case Tok::Kw_print:               return parsePrint();
  case Tok::Kw_save_checkpoint:     return parseSaveLoad(true);
  case Tok::Kw_load_checkpoint:     return parseSaveLoad(false);
  case Tok::Kw_export_raster:       return parseExportRaster();
  case Tok::Kw_if:                  return parseIf();
  case Tok::Kw_for:                 return parseFor();
  case Tok::Kw_while:               return parseWhile();
  case Tok::Kw_run:                 return parseRun();
  case Tok::Kw_at:                  return parseAtEmit();
  case Tok::Kw_on_spike:            return parseOnSpike();
  case Tok::Kw_signal:              return parseSignal();
  case Tok::Kw_oscillator:          return parseOscillator();
  case Tok::Kw_external:            return parseExternalStream();
  case Tok::Kw_set:                 return parseSetAssign();
  case Tok::Kw_run_continuous:      return parseRunContinuous();
  case Tok::Kw_stop_continuous:     return parseStopContinuous();
  case Tok::Kw_wait_continuous:     return parseWaitContinuous();
  case Tok::Kw_set_plasticity:      return parseConnArrowStmt(StmtKind::SetPlasticity);
  case Tok::Kw_prune_weights:       return parseConnArrowStmt(StmtKind::PruneWeights);
  case Tok::Kw_bind_input_stream:   return parseBindInputStream();
  case Tok::Ident:                  return parseSimpleOrExprStmt();
  case Tok::Semi:
    advance(); // tolerate stray ';'
    return nullptr;
  case Tok::LBrace:
    // A bare block is not part of the grammar; report and let the caller's
    // progress guarantee step over '{' so the contents still get parsed.
    parseError("expected statement, found '{' (bare blocks are not allowed; "
               "blocks only attach to if/else/for/while/run/on_spike)",
               cur().range);
    return nullptr;
  case Tok::Kw_import:
  case Tok::Kw_use:
  case Tok::Kw_node_group:
  case Tok::Kw_stream: // 'stream' only ever follows 'external', which is handled above
    parseError("expected statement, found " + spelling(cur().kind) +
                   " (" + spelling(cur().kind) + " is only allowed at the top level)",
               cur().range);
    synchronize();
    return nullptr;
  default:
    parseError("expected statement, found " + spelling(cur().kind), cur().range);
    synchronize();
    return nullptr;
  }
}

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------

StmtPtr Parser::parseImport() {
  const Token& kw = advance(); // 'import'
  auto s = makeStmt(StmtKind::Import, kw);
  const Token& pathTok = expect(Tok::StringLit, "after 'import'");
  if (pathTok.kind == Tok::StringLit) {
    s->path = pathTok.text;
    s->pathRange = pathTok.range;
  } else {
    // synchronize() consumed the recovery ';' already — do not expect it again.
    synchronize();
    s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
    return s;
  }
  const Token& semi = expect(Tok::Semi, "after 'import' path");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

StmtPtr Parser::parseUse() {
  const Token& kw = advance(); // 'use'
  auto s = makeStmt(StmtKind::Use, kw);
  const Token& nameTok = expect(Tok::Ident, "after 'use'");
  if (nameTok.kind == Tok::Ident) {
    s->name = nameTok.text;
  } else {
    synchronize(); // recovery point reached; do not re-error on the same token
    s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
    return s;
  }
  const Token& semi = expect(Tok::Semi, "after 'use' package name");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

StmtPtr Parser::parseNodeGroup() {
  const Token& kw = advance(); // 'node_group'
  auto s = makeStmt(StmtKind::NodeGroup, kw);
  const Token& nameTok = expect(Tok::Ident, "after 'node_group'");
  if (nameTok.kind == Tok::Ident) {
    s->name = nameTok.text;
  } else if (!check(Tok::LBracket)) {
    synchronize();
    s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
    return s;
  }
  expect(Tok::LBracket, "after node_group name");
  s->sizeExpr = parseExpr();
  expect(Tok::RBracket, "after node_group size");
  expect(Tok::Colon, "before neuron type");
  if (check(Tok::Kw_Excitatory) || check(Tok::Kw_Inhibitory)) {
    const Token& nt = advance();
    s->ntype = (nt.kind == Tok::Kw_Excitatory) ? NeuronType::Excitatory
                                               : NeuronType::Inhibitory;
  } else {
    parseError("expected 'Excitatory' or 'Inhibitory', found " + spelling(cur().kind),
               cur().range);
  }
  expect(Tok::LParen, "before node_group parameters");
  s->params = parseCallArgList(); // named or positional; sema validates names
  expect(Tok::RParen, "to close node_group parameters");
  const Token& semi = expect(Tok::Semi, "after node_group declaration");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

StmtPtr Parser::parseConnect(ConnectKind kind) {
  const Token& kw = advance(); // connect keyword
  auto s = makeStmt(StmtKind::Connect, kw);
  s->ckind = kind;
  expect(Tok::LParen, "after connect keyword");
  std::vector<CallArg> args = parseCallArgList();
  size_t positional = 0;
  for (auto& a : args) {
    if (a.named) {
      s->opts.push_back(std::move(a));
      continue;
    }
    if (positional < 2 && a.value && a.value->kind == ExprKind::Ident) {
      if (positional == 0)
        s->srcName = a.value->name;
      else
        s->dstName = a.value->name;
    } else {
      // Extra/invalid positional argument: keep it so sema can reject it.
      s->opts.push_back(std::move(a));
    }
    ++positional;
  }
  expect(Tok::RParen, "to close connect arguments");
  const Token& semi = expect(Tok::Semi, "after connect statement");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

StmtPtr Parser::parseStdp() {
  const Token& kw = advance(); // 'configure_stdp'
  auto s = makeStmt(StmtKind::Stdp, kw);
  expect(Tok::LParen, "after 'configure_stdp'");
  const Token& src = expect(Tok::Ident, "as STDP source group");
  if (src.kind == Tok::Ident) s->srcName = src.text;
  expect(Tok::Arrow, "between STDP source and target group");
  const Token& dst = expect(Tok::Ident, "as STDP target group");
  if (dst.kind == Tok::Ident) {
    s->dstName = dst.text;
  } else {
    synchronize();
    s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
    return s;
  }
  while (match(Tok::Comma)) {
    if (!check(Tok::Ident)) {
      parseError("expected option name, found " + spelling(cur().kind), cur().range);
      while (!check(Tok::RParen) && !check(Tok::Eof)) advance();
      break;
    }
    const Token& n = advance();
    expect(Tok::Assign, "after STDP option name");
    ExprPtr v = parseExpr();
    if (!v) break;
    CallArg a;
    a.named = true;
    a.name = n.text;
    a.range = SourceRange(n.range.begin, v->range.end);
    a.value = std::move(v);
    s->opts.push_back(std::move(a));
  }
  expect(Tok::RParen, "to close configure_stdp arguments");
  const Token& semi = expect(Tok::Semi, "after configure_stdp");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

// ---------------------------------------------------------------------------
// Simple statements
// ---------------------------------------------------------------------------

TypeRef Parser::parseTypeRef() {
  TypeRef t;
  const Token& first = cur();
  const size_t start = pos_;
  if (match(Tok::Kw_int)) {
    t.k = TypeRef::K::Int;
  } else if (match(Tok::Kw_float)) {
    t.k = TypeRef::K::Float;
  } else if (match(Tok::Kw_bool)) {
    t.k = TypeRef::K::Bool;
  } else if (match(Tok::Kw_string)) {
    t.k = TypeRef::K::String;
  } else if (match(Tok::Kw_tensor)) {
    t.k = TypeRef::K::Tensor;
    expect(Tok::Lt, "after 'tensor'");
    if (check(Tok::Kw_float)) {
      advance(); // element type (f32 only in v1.0)
    } else if (check(Tok::Kw_int) || check(Tok::Kw_bool) || check(Tok::Kw_string) ||
               check(Tok::Kw_tensor)) {
      // Wrong element type: consume it anyway so the dimensions still parse.
      parseError("expected 'float' element type, found " + spelling(cur().kind),
                 cur().range);
      advance();
    } else {
      parseError("expected 'float' element type, found " + spelling(cur().kind),
                 cur().range);
    }
    if (!check(Tok::Gt)) {
      if (!match(Tok::Comma))
        parseError("expected ',' before tensor dimensions, found " + spelling(cur().kind),
                   cur().range);
      while (!check(Tok::Gt) && !check(Tok::Eof)) {
        const Token& d = expect(Tok::IntLit, "as tensor dimension");
        if (d.kind == Tok::IntLit) t.tensorDims.push_back(d.intVal);
        if (!match(Tok::Comma)) break;
      }
    }
    expect(Tok::Gt, "to close tensor type");
  } else {
    parseError("expected type, found " + spelling(cur().kind), cur().range);
  }
  t.range = SourceRange(first.range.begin,
                        pos_ > start ? prevEnd(toks_, pos_) : first.range.end);
  return t;
}

StmtPtr Parser::parseLet() {
  const Token& kw = advance(); // 'let'
  auto s = makeStmt(StmtKind::Let, kw);
  const Token& nameTok = expect(Tok::Ident, "after 'let'");
  if (nameTok.kind == Tok::Ident) {
    s->name = nameTok.text;
  } else if (!check(Tok::Colon) && !check(Tok::Assign)) {
    synchronize();
    s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
    return s;
  }
  if (match(Tok::Colon)) s->typeAnno = parseTypeRef();
  expect(Tok::Assign, "after 'let' name");
  s->init = parseExpr();
  const Token& semi = expect(Tok::Semi, "after 'let' initializer");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

StmtPtr Parser::parsePrint() {
  const Token& kw = advance(); // 'print'
  auto s = makeStmt(StmtKind::Print, kw);
  expect(Tok::LParen, "after 'print'");
  if (!check(Tok::RParen) && !check(Tok::Eof)) {
    for (;;) {
      ExprPtr e = parseExpr();
      if (!e) break;
      s->printArgs.push_back(std::move(e));
      if (!match(Tok::Comma)) break;
    }
  }
  expect(Tok::RParen, "to close print arguments");
  const Token& semi = expect(Tok::Semi, "after 'print'");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

StmtPtr Parser::parseSaveLoad(bool save) {
  const Token& kw = advance(); // save_checkpoint | load_checkpoint
  auto s = makeStmt(save ? StmtKind::SaveCheckpoint : StmtKind::LoadCheckpoint, kw);
  expect(Tok::LParen, "after checkpoint keyword");
  s->init = parseExpr(); // path expression (usually a string literal)
  expect(Tok::RParen, "to close checkpoint path");
  const Token& semi = expect(Tok::Semi, "after checkpoint statement");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

StmtPtr Parser::parseExportRaster() {
  const Token& kw = advance(); // 'export_raster'
  auto s = makeStmt(StmtKind::ExportRaster, kw);
  expect(Tok::LParen, "after 'export_raster'");
  const Token& groupTok = expect(Tok::Ident, "as group name to export");
  if (groupTok.kind == Tok::Ident) {
    s->srcName = groupTok.text;
  } else {
    synchronize();
    s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
    return s;
  }
  expect(Tok::Comma, "between group name and file path");
  s->init = parseExpr(); // path expression
  expect(Tok::RParen, "to close export_raster arguments");
  const Token& semi = expect(Tok::Semi, "after export_raster");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

// `ident ...` — assignment (`ident '=' expr ';'`) or expression statement.
StmtPtr Parser::parseSimpleOrExprStmt() {
  const Token& first = cur(); // Ident
  if (peek(1).kind == Tok::Assign) {
    const Token& nameTok = advance();
    advance(); // '='
    auto s = makeStmt(StmtKind::Assign, first);
    s->name = nameTok.text;
    s->init = parseExpr();
    const Token& semi = expect(Tok::Semi, "after assignment");
    s->range = endRange(toks_, pos_, first, semi);
    if (semi.kind != Tok::Semi) synchronize();
    return s;
  }
  auto s = makeStmt(StmtKind::ExprStmt, first);
  s->expr = parseExpr();
  const Token& semi = expect(Tok::Semi, "after expression");
  s->range = endRange(toks_, pos_, first, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

// ---------------------------------------------------------------------------
// Control flow
// ---------------------------------------------------------------------------

std::vector<StmtPtr> Parser::parseBlockBody() {
  std::vector<StmtPtr> out;
  if (!check(Tok::LBrace)) {
    parseError("expected '{' to start block, found " + spelling(cur().kind),
               cur().range);
    synchronize();
    return out;
  }
  advance(); // '{'
  while (!check(Tok::RBrace) && !check(Tok::Eof)) {
    const size_t before = pos_;
    StmtPtr s = parseBlockStmt();
    if (s) out.push_back(std::move(s));
    // Progress guarantee (also skips tokens rejected by the dispatcher).
    if (pos_ == before && !check(Tok::RBrace) && !check(Tok::Eof)) advance();
  }
  expect(Tok::RBrace, "to close block");
  return out;
}

StmtPtr Parser::parseIf() {
  const Token& kw = advance(); // 'if'
  auto s = makeStmt(StmtKind::If, kw);
  expect(Tok::LParen, "after 'if'");
  s->cond = parseExpr();
  expect(Tok::RParen, "after 'if' condition");
  s->body = parseBlockBody();
  if (match(Tok::Kw_else)) {
    if (check(Tok::Kw_if)) {
      s->elseBody.push_back(parseIf());
    } else if (check(Tok::LBrace)) {
      s->elseBody = parseBlockBody();
    } else {
      parseError("expected 'if' or '{' after 'else', found " + spelling(cur().kind),
                 cur().range);
    }
  }
  s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
  return s;
}

StmtPtr Parser::parseFor() {
  const Token& kw = advance(); // 'for'
  auto s = makeStmt(StmtKind::For, kw);
  expect(Tok::LParen, "after 'for'");
  const Token& varTok = expect(Tok::Ident, "as loop variable");
  if (varTok.kind == Tok::Ident) {
    s->loopVar = varTok.text;
  } else if (!check(Tok::Kw_in)) {
    synchronize();
    s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
    return s;
  }
  expect(Tok::Kw_in, "after loop variable");
  s->lo = parseExpr();
  expect(Tok::DotDot, "between loop bounds");
  s->hi = parseExpr();
  expect(Tok::RParen, "after 'for' header");
  s->body = parseBlockBody();
  s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
  return s;
}

StmtPtr Parser::parseWhile() {
  const Token& kw = advance(); // 'while'
  auto s = makeStmt(StmtKind::While, kw);
  expect(Tok::LParen, "after 'while'");
  s->cond = parseExpr();
  expect(Tok::RParen, "after 'while' condition");
  s->body = parseBlockBody();
  s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
  return s;
}

StmtPtr Parser::parseRun() {
  const Token& kw = advance(); // 'run'
  auto s = makeStmt(StmtKind::Run, kw);
  expect(Tok::LParen, "after 'run'");
  if (!check(Tok::RParen) && !check(Tok::Eof)) {
    for (;;) {
      if (check(Tok::Ident)) {
        const Token& n = advance();
        expect(Tok::Assign, "after run parameter name");
        ExprPtr v = parseExpr();
        if (!v) break;
        if (n.text == "duration") {
          s->hasDuration = true;
          s->durationExpr = std::move(v);
        } else if (n.text == "dt") {
          s->hasDt = true;
          s->dtExpr = std::move(v);
        } else if (n.text == "device") {
          if (v->kind == ExprKind::DeviceLit) {
            s->hasDevice = true;
            s->device = (v->name == "GPU") ? DeviceKind::GPU : DeviceKind::CPU;
          } else {
            parseError(std::string("expected 'CPU' or 'GPU' after 'device=', found ") +
                           exprKindName(v->kind),
                       v->range);
          }
        } else {
          parseError("expected 'duration', 'dt' or 'device' as run parameter, found '" +
                         n.text + "'",
                     n.range);
        }
      } else {
        parseError("expected run parameter name, found " + spelling(cur().kind),
                   cur().range);
        while (!check(Tok::Comma) && !check(Tok::RParen) && !check(Tok::Eof)) advance();
      }
      if (!match(Tok::Comma)) break;
    }
  }
  expect(Tok::RParen, "to close run parameters");
  s->body = parseBlockBody();
  // Contract (ast.hpp): Run::body holds only AtEmit statements.
  s->body.erase(
      std::remove_if(s->body.begin(), s->body.end(),
                     [this](const StmtPtr& st) {
                       if (!st) return true;
                       if (st->kind != StmtKind::AtEmit) {
                         parseError("only 'at ... emit' statements are allowed inside "
                                    "a 'run' block",
                                    st->range);
                         return true;
                       }
                       return false;
                     }),
      s->body.end());
  s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
  return s;
}

StmtPtr Parser::parseAtEmit() {
  const Token& kw = advance(); // 'at'
  auto s = makeStmt(StmtKind::AtEmit, kw);
  s->timeExpr = parseExpr();
  expect(Tok::Kw_emit, "after 'at' time expression");
  const Token& groupTok = expect(Tok::Ident, "as target group of 'emit'");
  if (groupTok.kind == Tok::Ident) {
    s->name = groupTok.text;
  } else {
    synchronize();
    s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
    return s;
  }
  s->sliceRange = groupTok.range; // whole target (grows to ']' when sliced)
  if (check(Tok::LBracket)) {
    advance();
    s->sliceLo = parseExpr();
    if (s->sliceLo) {
      if (match(Tok::DotDot)) {
        s->sliceHi = parseExpr();
        s->sliceHasRange = (s->sliceHi != nullptr);
      } else {
        s->sliceHasIndex = true; // Group[i]: index kept in sliceLo
      }
    }
    if (check(Tok::RBracket)) {
      const Token& rb = advance();
      s->sliceRange = SourceRange(groupTok.range.begin, rb.range.end);
    } else {
      const Token& rb = expect(Tok::RBracket, "to close slice");
      if (rb.kind == Tok::RBracket)
        s->sliceRange = SourceRange(groupTok.range.begin, rb.range.end);
    }
  }
  expect(Tok::LParen, "before emit parameters");
  const Token& paramTok = expect(Tok::Ident, "as emit parameter name ('current')");
  if (paramTok.kind != Tok::Ident) {
    synchronize();
    s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
    return s;
  }
  if (paramTok.text != "current")
    parseError("expected 'current', found '" + paramTok.text + "'", paramTok.range);
  expect(Tok::Assign, "after 'current'");
  s->currentExpr = parseExpr();
  expect(Tok::RParen, "to close emit parameters");
  const Token& semi = expect(Tok::Semi, "after 'at ... emit' statement");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

StmtPtr Parser::parseOnSpike() {
  const Token& kw = advance(); // 'on_spike'
  auto s = makeStmt(StmtKind::OnSpike, kw);
  expect(Tok::LParen, "after 'on_spike'");
  const Token& groupTok = expect(Tok::Ident, "as source group of spike handler");
  if (groupTok.kind == Tok::Ident) s->name = groupTok.text;
  expect(Tok::RParen, "after 'on_spike' group");
  s->body = parseBlockBody();
  s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
  return s;
}

// ---------------------------------------------------------------------------
// v2.0 — asynchronous continuous-time environment
// ---------------------------------------------------------------------------

// signal <Name> [':' 'float'] '=' expr ';'
StmtPtr Parser::parseSignal() {
  const Token& kw = advance(); // 'signal'
  auto s = makeStmt(StmtKind::SignalDecl, kw);
  const Token& nameTok = expect(Tok::Ident, "after 'signal'");
  if (nameTok.kind == Tok::Ident) {
    s->name = nameTok.text;
  } else if (!check(Tok::Colon) && !check(Tok::Assign)) {
    synchronize();
    s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
    return s;
  }
  if (match(Tok::Colon)) {
    // Type annotation: 'float' (v2.0 signals are scalar modulators).
    if (match(Tok::Kw_float)) {
      // ok
    } else {
      parseError("expected 'float' as signal type, found " + spelling(cur().kind),
                 cur().range);
      if (check(Tok::Kw_int) || check(Tok::Kw_bool) || check(Tok::Kw_string) ||
          check(Tok::Kw_tensor))
        advance();
    }
  }
  expect(Tok::Assign, "after signal name");
  s->init = parseExpr();
  const Token& semi = expect(Tok::Semi, "after signal initializer");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

// oscillator <Name>(frequency=<f>[Hz], amplitude=<f>, target=<Group>[, phase=<f>]);
StmtPtr Parser::parseOscillator() {
  const Token& kw = advance(); // 'oscillator'
  auto s = makeStmt(StmtKind::OscillatorDecl, kw);
  const Token& nameTok = expect(Tok::Ident, "after 'oscillator'");
  if (nameTok.kind == Tok::Ident) {
    s->name = nameTok.text;
  } else if (!check(Tok::LParen)) {
    synchronize();
    s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
    return s;
  }
  expect(Tok::LParen, "after oscillator name");
  s->opts = parseCallArgList(); // named or positional; sema validates names
  expect(Tok::RParen, "to close oscillator parameters");
  const Token& semi = expect(Tok::Semi, "after oscillator declaration");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

// external stream <Name> '[' expr ']' ';'
StmtPtr Parser::parseExternalStream() {
  const Token& kw = advance(); // 'external'
  if (!check(Tok::Kw_stream)) {
    parseError("expected 'stream' after 'external', found " + spelling(cur().kind),
               cur().range);
    synchronize();
    auto s = makeStmt(StmtKind::ExternalStreamDecl, kw);
    s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
    return s;
  }
  advance(); // 'stream'
  auto s = makeStmt(StmtKind::ExternalStreamDecl, kw);
  const Token& nameTok = expect(Tok::Ident, "after 'external stream'");
  if (nameTok.kind == Tok::Ident) {
    s->name = nameTok.text;
  } else if (!check(Tok::LBracket)) {
    synchronize();
    s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
    return s;
  }
  expect(Tok::LBracket, "after stream name");
  s->sizeExpr = parseExpr();
  expect(Tok::RBracket, "after stream size");
  const Token& semi = expect(Tok::Semi, "after external stream declaration");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

// Shared shape for set_plasticity / prune_weights:
//   kw '(' IDENT '->' IDENT {',' IDENT '=' expr} ')' ';'
StmtPtr Parser::parseConnArrowStmt(StmtKind kind) {
  const Token& kw = advance(); // 'set_plasticity' | 'prune_weights'
  auto s = makeStmt(kind, kw);
  expect(Tok::LParen, "after keyword");
  const Token& src = expect(Tok::Ident, "as source group");
  if (src.kind == Tok::Ident) s->srcName = src.text;
  expect(Tok::Arrow, "between source and target group");
  const Token& dst = expect(Tok::Ident, "as target group");
  if (dst.kind == Tok::Ident) {
    s->dstName = dst.text;
  } else {
    synchronize();
    s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
    return s;
  }
  while (match(Tok::Comma)) {
    if (!check(Tok::Ident)) {
      parseError("expected option name, found " + spelling(cur().kind), cur().range);
      while (!check(Tok::RParen) && !check(Tok::Eof)) advance();
      break;
    }
    const Token& n = advance();
    expect(Tok::Assign, "after option name");
    ExprPtr v = parseExpr();
    if (!v) break;
    CallArg a;
    a.named = true;
    a.name = n.text;
    a.range = SourceRange(n.range.begin, v->range.end);
    a.value = std::move(v);
    s->opts.push_back(std::move(a));
  }
  expect(Tok::RParen, "to close arguments");
  const Token& semi = expect(Tok::Semi, "after statement");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

// bind_input_stream '(' IDENT ',' IDENT {',' IDENT '=' expr} ')' ';'
// positional: stream, group; named: encoding (=Poisson), max_freq
StmtPtr Parser::parseBindInputStream() {
  const Token& kw = advance(); // 'bind_input_stream'
  auto s = makeStmt(StmtKind::BindInputStream, kw);
  expect(Tok::LParen, "after 'bind_input_stream'");
  size_t positional = 0;
  for (;;) {
    if (check(Tok::RParen) || check(Tok::Eof)) break;
    CallArg a;
    bool ok = true;
    if (check(Tok::Ident) && peek(1).kind == Tok::Assign) {
      const Token& n = advance();
      advance(); // '='
      a.named = true;
      a.name = n.text;
      a.value = parseExpr();
      if (!a.value) ok = false;
    } else {
      a.value = parseExpr();
      if (!a.value) ok = false;
    }
    if (ok) {
      if (a.named) {
        s->opts.push_back(std::move(a));
      } else if (positional == 0 && a.value->kind == ExprKind::Ident) {
        s->srcName = a.value->name; // stream
        ++positional;
      } else if (positional == 1 && a.value->kind == ExprKind::Ident) {
        s->dstName = a.value->name; // group
        ++positional;
      } else {
        s->opts.push_back(std::move(a)); // extra: sema rejects
        ++positional;
      }
    }
    // Recovery: skip to the next ',' / ')'.
    while (!check(Tok::Comma) && !check(Tok::RParen) && !check(Tok::Eof)) advance();
    if (!match(Tok::Comma)) break;
  }
  expect(Tok::RParen, "to close bind_input_stream arguments");
  const Token& semi = expect(Tok::Semi, "after bind_input_stream");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

// `set IDENT '=' expr ';'` — assignment with the optional 'set' prefix
// (v1.0 reserved the keyword; v2.0 uses it for signal updates). Equivalent
// to the bare `IDENT '=' expr ';'` form.
StmtPtr Parser::parseSetAssign() {
  const Token& kw = advance(); // 'set'
  const Token& nameTok = expect(Tok::Ident, "after 'set'");
  if (nameTok.kind != Tok::Ident) {
    synchronize();
    auto s = makeStmt(StmtKind::Assign, kw);
    s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
    return s;
  }
  expect(Tok::Assign, "after assignment target");
  auto s = makeStmt(StmtKind::Assign, kw);
  s->name = nameTok.text;
  s->init = parseExpr();
  const Token& semi = expect(Tok::Semi, "after assignment");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

// stop_continuous ['(' ')'] ';' — stops the continuous event loop (callable
// from the main thread and from spike handlers running inside it).
StmtPtr Parser::parseStopContinuous() {
  const Token& kw = advance(); // 'stop_continuous'
  auto s = makeStmt(StmtKind::StopContinuous, kw);
  if (check(Tok::LParen)) {
    advance();
    expect(Tok::RParen, "to close stop_continuous arguments");
  }
  const Token& semi = expect(Tok::Semi, "after 'stop_continuous'");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

// wait_continuous ['(' ')'] ';' — blocks the main thread until the continuous
// event loop stops (used for bounded simulation-time episodes).
StmtPtr Parser::parseWaitContinuous() {
  const Token& kw = advance(); // 'wait_continuous'
  auto s = makeStmt(StmtKind::WaitContinuous, kw);
  if (check(Tok::LParen)) {
    advance();
    expect(Tok::RParen, "to close wait_continuous arguments");
  }
  const Token& semi = expect(Tok::Semi, "after 'wait_continuous'");
  s->range = endRange(toks_, pos_, kw, semi);
  if (semi.kind != Tok::Semi) synchronize();
  return s;
}

// run_continuous '(' [dt = expr] [, device = CPU|GPU] ')' block
StmtPtr Parser::parseRunContinuous() {
  const Token& kw = advance(); // 'run_continuous'
  auto s = makeStmt(StmtKind::RunContinuous, kw);
  expect(Tok::LParen, "after 'run_continuous'");
  if (!check(Tok::RParen) && !check(Tok::Eof)) {
    for (;;) {
      if (check(Tok::Ident)) {
        const Token& n = advance();
        expect(Tok::Assign, "after run_continuous parameter name");
        ExprPtr v = parseExpr();
        if (!v) break;
        if (n.text == "dt") {
          s->hasDt = true;
          s->dtExpr = std::move(v);
        } else if (n.text == "device") {
          if (v->kind == ExprKind::DeviceLit) {
            s->hasDevice = true;
            s->device = (v->name == "GPU") ? DeviceKind::GPU : DeviceKind::CPU;
          } else {
            parseError(std::string("expected 'CPU' or 'GPU' after 'device=', found ") +
                           exprKindName(v->kind),
                       v->range);
          }
        } else {
          parseError("expected 'dt' or 'device' as run_continuous parameter, found '" +
                         n.text + "'",
                     n.range);
        }
      } else {
        parseError("expected run_continuous parameter name, found " + spelling(cur().kind),
                   cur().range);
        while (!check(Tok::Comma) && !check(Tok::RParen) && !check(Tok::Eof)) advance();
      }
      if (!match(Tok::Comma)) break;
    }
  }
  expect(Tok::RParen, "to close run_continuous parameters");
  s->body = parseBlockBody();
  // Unlike run(), the body keeps ALL statements — sema enforces what is legal
  // (everything except run/run_continuous/at_emit).
  s->range = SourceRange(kw.range.begin, prevEnd(toks_, pos_));
  return s;
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

int Parser::binPrec(Tok op) {
  switch (op) {
  case Tok::OrOr: return 1;
  case Tok::AndAnd: return 2;
  case Tok::EqEq:
  case Tok::NotEq: return 3;
  case Tok::Lt:
  case Tok::Gt:
  case Tok::Le:
  case Tok::Ge: return 4;
  case Tok::Plus:
  case Tok::Minus: return 5;
  case Tok::Star:
  case Tok::Slash:
  case Tok::Percent: return 6;
  default: return 0; // not a binary operator
  }
}

ExprPtr Parser::parseExpr() { return parseBinary(1); }

ExprPtr Parser::parseBinary(int minPrec) {
  ExprPtr lhs = parseUnary();
  if (!lhs) return nullptr;
  for (;;) {
    const int p = binPrec(cur().kind);
    if (p == 0 || p < minPrec) break;
    const Tok op = cur().kind;
    advance();
    ExprPtr rhs = parseBinary(p + 1); // left-assoc
    if (!rhs) return nullptr;         // error already reported
    auto e = makeExpr(ExprKind::Binary, SourceRange(lhs->range.begin, rhs->range.end));
    e->op = op;
    e->lhs = std::move(lhs);
    e->rhs = std::move(rhs);
    lhs = std::move(e);
  }
  return lhs;
}

ExprPtr Parser::parseUnary() {
  if (check(Tok::Not) || check(Tok::Minus)) {
    const Token& opTok = advance();
    ExprPtr operand = parseUnary();
    if (!operand) return nullptr;
    auto e = makeExpr(ExprKind::Unary, SourceRange(opTok.range.begin, operand->range.end));
    e->op = opTok.kind;
    e->lhs = std::move(operand);
    return e;
  }
  return parsePrimary();
}

ExprPtr Parser::parsePrimary() {
  const Token& t = cur();
  switch (t.kind) {
  case Tok::IntLit: {
    advance();
    auto e = makeExpr(ExprKind::IntLit, t.range);
    e->intVal = t.intVal;
    return e;
  }
  case Tok::FloatLit: {
    advance();
    auto e = makeExpr(ExprKind::FloatLit, t.range);
    e->floatVal = t.floatVal;
    return e;
  }
  case Tok::DurationLit: {
    advance();
    auto e = makeExpr(ExprKind::DurationLit, t.range);
    e->floatVal = t.floatVal; // milliseconds
    return e;
  }
  case Tok::FreqLit: {
    advance();
    auto e = makeExpr(ExprKind::FloatLit, t.range);
    e->floatVal = t.floatVal; // hertz — a plain Float value
    return e;
  }
  case Tok::StringLit: {
    advance();
    auto e = makeExpr(ExprKind::StringLit, t.range);
    e->strVal = t.text;
    return e;
  }
  case Tok::Kw_true:
  case Tok::Kw_false: {
    advance();
    auto e = makeExpr(ExprKind::BoolLit, t.range);
    e->boolVal = (t.kind == Tok::Kw_true);
    return e;
  }
  case Tok::Kw_CPU:
  case Tok::Kw_GPU: {
    advance();
    auto e = makeExpr(ExprKind::DeviceLit, t.range);
    e->name = (t.kind == Tok::Kw_CPU) ? "CPU" : "GPU";
    return e;
  }
  case Tok::Ident: {
    advance();
    if (check(Tok::LParen)) {
      auto id = makeExpr(ExprKind::Ident, t.range);
      id->name = t.text;
      return parseCallArgs(std::move(id));
    }
    auto e = makeExpr(ExprKind::Ident, t.range);
    e->name = t.text;
    return e;
  }
  case Tok::LParen: {
    advance();
    ExprPtr inner = parseExpr();
    if (!inner) return nullptr;
    if (check(Tok::RParen)) {
      const Token& rp = advance();
      inner->range = SourceRange(t.range.begin, rp.range.end);
    } else {
      expect(Tok::RParen, "to close parenthesized expression");
    }
    return inner;
  }
  default:
    parseError("expected expression, found " + spelling(t.kind), t.range);
    return nullptr;
  }
}

ExprPtr Parser::parseCallArgs(ExprPtr calleeExpr) {
  // cur() == '('. calleeExpr is the already-parsed Ident expression.
  const std::string name = calleeExpr ? calleeExpr->name : std::string();
  const SourceLoc begin = calleeExpr ? calleeExpr->range.begin : cur().range.begin;
  advance(); // '('
  auto call = std::make_unique<Expr>();
  call->kind = ExprKind::Call;
  call->name = name;
  call->args = parseCallArgList();
  if (check(Tok::RParen)) {
    const Token& rp = advance();
    call->range = SourceRange(begin, rp.range.end);
  } else {
    expect(Tok::RParen, "to close call arguments");
    call->range = SourceRange(begin, prevEnd(toks_, pos_));
  }
  return call;
}

std::vector<CallArg> Parser::parseCallArgList() {
  // '(' already consumed by the caller.
  std::vector<CallArg> out;
  while (!check(Tok::RParen) && !check(Tok::Eof)) {
    CallArg a;
    bool ok = true;
    if (check(Tok::Ident) && peek(1).kind == Tok::Assign) {
      const Token& n = advance();
      advance(); // '='
      a.named = true;
      a.name = n.text;
      a.value = parseExpr();
      if (a.value)
        a.range = SourceRange(n.range.begin, a.value->range.end);
      else
        ok = false;
    } else {
      a.value = parseExpr();
      if (a.value)
        a.range = a.value->range;
      else
        ok = false;
    }
    if (!ok) {
      // Recovery: skip to the next ',' / ')' and continue with the next arg.
      while (!check(Tok::Comma) && !check(Tok::RParen) && !check(Tok::Eof)) advance();
      if (check(Tok::Comma)) {
        advance();
        continue;
      }
      break;
    }
    out.push_back(std::move(a));
    if (!match(Tok::Comma)) break;
  }
  return out;
}

} // namespace ndl
