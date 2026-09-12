// NDL v1.0 — Hand-written lexer implementation (spec: docs/INTERNALS.md §2).
//
// This translation unit also hosts the token helpers declared in tokens.hpp
// (tokName / tokSpelling / tokIsKeyword): the build has no src/tokens.cpp.
//
// Summary of the lexical grammar:
//   int literal    : [0-9]+ | "0x" [0-9a-fA-F]+                      -> intVal
//   float literal  : [0-9]+ '.' [0-9]+ ( [eE] [+-]? [0-9]+ )?        -> floatVal
//                    (a digit is required BEFORE the dot; exponents too)
//   duration       : <int|float> "ms"  (ms NOT followed by [A-Za-z0-9_]) -> floatVal (ms)
//   string literal : "..." with escapes \" \\ \n \t \r \0
//   comments       : // line  and  /* block (non-nested) */
//   '..' rule      : while scanning a number, '.' followed by '.' ends the
//                    number (int); '..' is lexed as DotDot on the next call.
//                    A lone '.' is an E0001 error.
// Every token's range is [start, position after last char); errors are E0001,
// reported to the engine, 1 character skipped, lexing continues; the stream
// always ends with exactly one Tok::Eof. line/col are 1-based, col in bytes.
#include "ndl/lexer.hpp"

#include <cstdlib>
#include <limits>
#include <string>
#include <unordered_map>

namespace ndl {

// ---------------------------------------------------------------------------
// Token helpers (declared in tokens.hpp; kept here — no src/tokens.cpp exists)
// ---------------------------------------------------------------------------
const char* tokName(Tok t) {
  switch (t) {
  case Tok::Eof: return "Eof";
  case Tok::Ident: return "Ident";
  case Tok::IntLit: return "IntLit";
  case Tok::FloatLit: return "FloatLit";
  case Tok::DurationLit: return "DurationLit";
  case Tok::FreqLit: return "FreqLit";
  case Tok::StringLit: return "StringLit";
  case Tok::Kw_node_group: return "Kw_node_group";
  case Tok::Kw_dense_connect: return "Kw_dense_connect";
  case Tok::Kw_sparse_connect: return "Kw_sparse_connect";
  case Tok::Kw_one_to_one_connect: return "Kw_one_to_one_connect";
  case Tok::Kw_configure_stdp: return "Kw_configure_stdp";
  case Tok::Kw_run: return "Kw_run";
  case Tok::Kw_at: return "Kw_at";
  case Tok::Kw_emit: return "Kw_emit";
  case Tok::Kw_on_spike: return "Kw_on_spike";
  case Tok::Kw_if: return "Kw_if";
  case Tok::Kw_else: return "Kw_else";
  case Tok::Kw_for: return "Kw_for";
  case Tok::Kw_in: return "Kw_in";
  case Tok::Kw_while: return "Kw_while";
  case Tok::Kw_let: return "Kw_let";
  case Tok::Kw_set: return "Kw_set";
  case Tok::Kw_print: return "Kw_print";
  case Tok::Kw_save_checkpoint: return "Kw_save_checkpoint";
  case Tok::Kw_load_checkpoint: return "Kw_load_checkpoint";
  case Tok::Kw_export_raster: return "Kw_export_raster";
  case Tok::Kw_import: return "Kw_import";
  case Tok::Kw_use: return "Kw_use";
  case Tok::Kw_true: return "Kw_true";
  case Tok::Kw_false: return "Kw_false";
  case Tok::Kw_int: return "Kw_int";
  case Tok::Kw_float: return "Kw_float";
  case Tok::Kw_bool: return "Kw_bool";
  case Tok::Kw_string: return "Kw_string";
  case Tok::Kw_tensor: return "Kw_tensor";
  case Tok::Kw_Excitatory: return "Kw_Excitatory";
  case Tok::Kw_Inhibitory: return "Kw_Inhibitory";
  case Tok::Kw_CPU: return "Kw_CPU";
  case Tok::Kw_GPU: return "Kw_GPU";
  case Tok::Kw_ms: return "Kw_ms";
  case Tok::Kw_signal: return "Kw_signal";
  case Tok::Kw_oscillator: return "Kw_oscillator";
  case Tok::Kw_external: return "Kw_external";
  case Tok::Kw_stream: return "Kw_stream";
  case Tok::Kw_run_continuous: return "Kw_run_continuous";
  case Tok::Kw_stop_continuous: return "Kw_stop_continuous";
  case Tok::Kw_wait_continuous: return "Kw_wait_continuous";
  case Tok::Kw_set_plasticity: return "Kw_set_plasticity";
  case Tok::Kw_prune_weights: return "Kw_prune_weights";
  case Tok::Kw_bind_input_stream: return "Kw_bind_input_stream";
  case Tok::Plus: return "Plus";
  case Tok::Minus: return "Minus";
  case Tok::Star: return "Star";
  case Tok::Slash: return "Slash";
  case Tok::Percent: return "Percent";
  case Tok::EqEq: return "EqEq";
  case Tok::NotEq: return "NotEq";
  case Tok::Lt: return "Lt";
  case Tok::Gt: return "Gt";
  case Tok::Le: return "Le";
  case Tok::Ge: return "Ge";
  case Tok::AndAnd: return "AndAnd";
  case Tok::OrOr: return "OrOr";
  case Tok::Not: return "Not";
  case Tok::Assign: return "Assign";
  case Tok::Arrow: return "Arrow";
  case Tok::DotDot: return "DotDot";
  case Tok::Comma: return "Comma";
  case Tok::Semi: return "Semi";
  case Tok::Colon: return "Colon";
  case Tok::LParen: return "LParen";
  case Tok::RParen: return "RParen";
  case Tok::LBracket: return "LBracket";
  case Tok::RBracket: return "RBracket";
  case Tok::LBrace: return "LBrace";
  case Tok::RBrace: return "RBrace";
  }
  return "?";
}

const char* tokSpelling(Tok t) {
  switch (t) {
  case Tok::Eof: return "end of file";
  case Tok::Ident: return "identifier";
  case Tok::IntLit: return "integer literal";
  case Tok::FloatLit: return "float literal";
  case Tok::DurationLit: return "duration literal";
  case Tok::FreqLit: return "frequency literal";
  case Tok::StringLit: return "string literal";
  case Tok::Kw_node_group: return "'node_group'";
  case Tok::Kw_dense_connect: return "'dense_connect'";
  case Tok::Kw_sparse_connect: return "'sparse_connect'";
  case Tok::Kw_one_to_one_connect: return "'one_to_one_connect'";
  case Tok::Kw_configure_stdp: return "'configure_stdp'";
  case Tok::Kw_run: return "'run'";
  case Tok::Kw_at: return "'at'";
  case Tok::Kw_emit: return "'emit'";
  case Tok::Kw_on_spike: return "'on_spike'";
  case Tok::Kw_if: return "'if'";
  case Tok::Kw_else: return "'else'";
  case Tok::Kw_for: return "'for'";
  case Tok::Kw_in: return "'in'";
  case Tok::Kw_while: return "'while'";
  case Tok::Kw_let: return "'let'";
  case Tok::Kw_set: return "'set'";
  case Tok::Kw_print: return "'print'";
  case Tok::Kw_save_checkpoint: return "'save_checkpoint'";
  case Tok::Kw_load_checkpoint: return "'load_checkpoint'";
  case Tok::Kw_export_raster: return "'export_raster'";
  case Tok::Kw_import: return "'import'";
  case Tok::Kw_use: return "'use'";
  case Tok::Kw_true: return "'true'";
  case Tok::Kw_false: return "'false'";
  case Tok::Kw_int: return "'int'";
  case Tok::Kw_float: return "'float'";
  case Tok::Kw_bool: return "'bool'";
  case Tok::Kw_string: return "'string'";
  case Tok::Kw_tensor: return "'tensor'";
  case Tok::Kw_Excitatory: return "'Excitatory'";
  case Tok::Kw_Inhibitory: return "'Inhibitory'";
  case Tok::Kw_CPU: return "'CPU'";
  case Tok::Kw_GPU: return "'GPU'";
  case Tok::Kw_ms: return "'ms'";
  case Tok::Kw_signal: return "'signal'";
  case Tok::Kw_oscillator: return "'oscillator'";
  case Tok::Kw_external: return "'external'";
  case Tok::Kw_stream: return "'stream'";
  case Tok::Kw_run_continuous: return "'run_continuous'";
  case Tok::Kw_stop_continuous: return "'stop_continuous'";
  case Tok::Kw_wait_continuous: return "'wait_continuous'";
  case Tok::Kw_set_plasticity: return "'set_plasticity'";
  case Tok::Kw_prune_weights: return "'prune_weights'";
  case Tok::Kw_bind_input_stream: return "'bind_input_stream'";
  case Tok::Plus: return "'+'";
  case Tok::Minus: return "'-'";
  case Tok::Star: return "'*'";
  case Tok::Slash: return "'/'";
  case Tok::Percent: return "'%'";
  case Tok::EqEq: return "'=='";
  case Tok::NotEq: return "'!='";
  case Tok::Lt: return "'<'";
  case Tok::Gt: return "'>'";
  case Tok::Le: return "'<='";
  case Tok::Ge: return "'>='";
  case Tok::AndAnd: return "'&&'";
  case Tok::OrOr: return "'||'";
  case Tok::Not: return "'!'";
  case Tok::Assign: return "'='";
  case Tok::Arrow: return "'->'";
  case Tok::DotDot: return "'..'";
  case Tok::Comma: return "','";
  case Tok::Semi: return "';'";
  case Tok::Colon: return "':'";
  case Tok::LParen: return "'('";
  case Tok::RParen: return "')'";
  case Tok::LBracket: return "'['";
  case Tok::RBracket: return "']'";
  case Tok::LBrace: return "'{'";
  case Tok::RBrace: return "'}'";
  }
  return "?";
}

bool tokIsKeyword(Tok t) {
  switch (t) {
  case Tok::Kw_node_group:
  case Tok::Kw_dense_connect:
  case Tok::Kw_sparse_connect:
  case Tok::Kw_one_to_one_connect:
  case Tok::Kw_configure_stdp:
  case Tok::Kw_run:
  case Tok::Kw_at:
  case Tok::Kw_emit:
  case Tok::Kw_on_spike:
  case Tok::Kw_if:
  case Tok::Kw_else:
  case Tok::Kw_for:
  case Tok::Kw_in:
  case Tok::Kw_while:
  case Tok::Kw_let:
  case Tok::Kw_set:
  case Tok::Kw_print:
  case Tok::Kw_save_checkpoint:
  case Tok::Kw_load_checkpoint:
  case Tok::Kw_export_raster:
  case Tok::Kw_import:
  case Tok::Kw_use:
  case Tok::Kw_true:
  case Tok::Kw_false:
  case Tok::Kw_int:
  case Tok::Kw_float:
  case Tok::Kw_bool:
  case Tok::Kw_string:
  case Tok::Kw_tensor:
  case Tok::Kw_Excitatory:
  case Tok::Kw_Inhibitory:
  case Tok::Kw_CPU:
  case Tok::Kw_GPU:
  case Tok::Kw_ms:
  case Tok::Kw_signal:
  case Tok::Kw_oscillator:
  case Tok::Kw_external:
  case Tok::Kw_stream:
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

// ---------------------------------------------------------------------------
// Character classification (locale-independent, no <cctype>)
// ---------------------------------------------------------------------------
namespace {

bool isDigitChar(char c) { return c >= '0' && c <= '9'; }

bool isHexDigitChar(char c) {
  return isDigitChar(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

bool isIdentStart(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

bool isIdentChar(char c) { return isIdentStart(c) || isDigitChar(c); }

int hexDigitVal(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return c - 'A' + 10;
}

const std::unordered_map<std::string, Tok>& keywordMap() {
  static const std::unordered_map<std::string, Tok> m = {
      {"node_group", Tok::Kw_node_group},
      {"dense_connect", Tok::Kw_dense_connect},
      {"sparse_connect", Tok::Kw_sparse_connect},
      {"one_to_one_connect", Tok::Kw_one_to_one_connect},
      {"configure_stdp", Tok::Kw_configure_stdp},
      {"run", Tok::Kw_run},
      {"at", Tok::Kw_at},
      {"emit", Tok::Kw_emit},
      {"on_spike", Tok::Kw_on_spike},
      {"if", Tok::Kw_if},
      {"else", Tok::Kw_else},
      {"for", Tok::Kw_for},
      {"in", Tok::Kw_in},
      {"while", Tok::Kw_while},
      {"let", Tok::Kw_let},
      {"set", Tok::Kw_set},
      {"print", Tok::Kw_print},
      {"save_checkpoint", Tok::Kw_save_checkpoint},
      {"load_checkpoint", Tok::Kw_load_checkpoint},
      {"export_raster", Tok::Kw_export_raster},
      {"import", Tok::Kw_import},
      {"use", Tok::Kw_use},
      {"true", Tok::Kw_true},
      {"false", Tok::Kw_false},
      {"int", Tok::Kw_int},
      {"float", Tok::Kw_float},
      {"bool", Tok::Kw_bool},
      {"string", Tok::Kw_string},
      {"tensor", Tok::Kw_tensor},
      {"Excitatory", Tok::Kw_Excitatory},
      {"Inhibitory", Tok::Kw_Inhibitory},
      {"CPU", Tok::Kw_CPU},
      {"GPU", Tok::Kw_GPU},
      {"ms", Tok::Kw_ms},
      // v2.0 — asynchronous continuous-time environment
      {"signal", Tok::Kw_signal},
      {"oscillator", Tok::Kw_oscillator},
      {"external", Tok::Kw_external},
      {"stream", Tok::Kw_stream},
      {"run_continuous", Tok::Kw_run_continuous},
      {"stop_continuous", Tok::Kw_stop_continuous},
      {"wait_continuous", Tok::Kw_wait_continuous},
      {"set_plasticity", Tok::Kw_set_plasticity},
      {"prune_weights", Tok::Kw_prune_weights},
      {"bind_input_stream", Tok::Kw_bind_input_stream},
  };
  return m;
}

} // namespace

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------
Lexer::Lexer(std::string fileName, std::string source, DiagnosticEngine& diag)
    : file_(std::move(fileName)), src_(std::move(source)), diag_(diag) {}

bool Lexer::atEnd() const { return pos_ >= src_.size(); }

char Lexer::cur() const { return pos_ < src_.size() ? src_[pos_] : '\0'; }

char Lexer::peek(size_t n) const { return pos_ + n < src_.size() ? src_[pos_ + n] : '\0'; }

char Lexer::advance() {
  if (atEnd()) return '\0';
  const char c = src_[pos_++];
  if (c == '\n') {
    ++line_;
    col_ = 1;
  } else {
    ++col_; // '\r' counts as one column ("space" before '\n' in CRLF)
  }
  return c;
}

SourceLoc Lexer::locHere() const {
  return SourceLoc(file_, line_, col_, static_cast<uint32_t>(pos_));
}

Token Lexer::makeToken(Tok kind, SourceRange range, std::string text) {
  Token t;
  t.kind = kind;
  t.text = std::move(text);
  t.range = std::move(range);
  return t;
}

void Lexer::lexError(const std::string& msg, SourceRange range) {
  diag_.error("E0001", std::move(range), msg);
}

void Lexer::skipWhitespaceAndComments() {
  while (!atEnd()) {
    const char c = cur();
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      advance();
      continue;
    }
    if (c == '/' && peek(1) == '/') { // line comment: up to (not incl.) '\n'
      while (!atEnd() && cur() != '\n') advance();
      continue;
    }
    if (c == '/' && peek(1) == '*') { // block comment: non-nested
      const SourceLoc start = locHere();
      advance(); // '/'
      advance(); // '*'
      bool closed = false;
      while (!atEnd()) {
        if (cur() == '*' && peek(1) == '/') {
          advance();
          advance();
          closed = true;
          break;
        }
        advance();
      }
      if (!closed) lexError("unterminated block comment", SourceRange(start, locHere()));
      continue;
    }
    break;
  }
}

Token Lexer::lexNumber() {
  const SourceLoc start = locHere();
  std::string raw; // raw spelling, stored in Token.text
  bool isFloat = false;
  uint64_t intVal = 0;
  const uint64_t kMax = std::numeric_limits<uint64_t>::max();

  if (cur() == '0' && (peek(1) == 'x' || peek(1) == 'X')) {
    if (!isHexDigitChar(peek(2))) {
      // Malformed hex ("0x" with no digits): report E0001, skip one char ('0'),
      // emit IntLit 0; the trailing text is re-lexed from scratch.
      advance();
      lexError("invalid hex literal: no digits after '0x'", SourceRange(start, locHere()));
      Token t = makeToken(Tok::IntLit, SourceRange(start, locHere()), "0");
      t.intVal = 0;
      return t;
    }
    raw += advance(); // '0'
    raw += advance(); // 'x'
    while (isHexDigitChar(cur())) raw += advance();
    for (size_t i = 2; i < raw.size(); ++i) { // hex has no '.'/'e' continuation
      const uint64_t d = static_cast<uint64_t>(hexDigitVal(raw[i]));
      if (intVal > (kMax - d) / 16) intVal = kMax; // saturate on overflow
      else intVal = intVal * 16 + d;
    }
  } else {
    while (isDigitChar(cur())) raw += advance();

    // Fraction: '.' starts a float only when a digit follows. If the next char
    // is another '.', the number ends here (int) and '..' is lexed next; any
    // other char after '.' leaves it to the main loop (lone '.' -> E0001).
    if (cur() == '.' && isDigitChar(peek(1))) {
      isFloat = true;
      raw += advance(); // '.'
      while (isDigitChar(cur())) raw += advance();
    }

    // Exponent: 'e'/'E' [+|-] digits; otherwise the letter belongs to an ident.
    if (cur() == 'e' || cur() == 'E') {
      const char n1 = peek(1);
      if (isDigitChar(n1) || ((n1 == '+' || n1 == '-') && isDigitChar(peek(2)))) {
        isFloat = true;
        raw += advance(); // 'e'
        if (cur() == '+' || cur() == '-') raw += advance();
        while (isDigitChar(cur())) raw += advance();
      }
    }

    if (!isFloat) {
      for (const char ch : raw) {
        const uint64_t d = static_cast<uint64_t>(ch - '0');
        if (intVal > (kMax - d) / 10) intVal = kMax; // saturate on overflow
        else intVal = intVal * 10 + d;
      }
    }
  }

  // Duration suffix: "ms" right after the number, NOT followed by [A-Za-z0-9_].
  if (cur() == 'm' && peek(1) == 's' && !isIdentChar(peek(2))) {
    raw += advance(); // 'm'
    raw += advance(); // 's'
    Token t = makeToken(Tok::DurationLit, SourceRange(start, locHere()), raw);
    t.floatVal = isFloat ? std::strtod(raw.c_str(), nullptr) : static_cast<double>(intVal);
    return t;
  }

  // v2.0 frequency suffix: "Hz" right after the number, NOT followed by [A-Za-z0-9_].
  if (cur() == 'H' && peek(1) == 'z' && !isIdentChar(peek(2))) {
    raw += advance(); // 'H'
    raw += advance(); // 'z'
    Token t = makeToken(Tok::FreqLit, SourceRange(start, locHere()), raw);
    t.floatVal = isFloat ? std::strtod(raw.c_str(), nullptr) : static_cast<double>(intVal);
    return t;
  }

  const SourceRange range(start, locHere());
  if (isFloat) {
    Token t = makeToken(Tok::FloatLit, range, raw);
    t.floatVal = std::strtod(raw.c_str(), nullptr);
    return t;
  }
  Token t = makeToken(Tok::IntLit, range, raw);
  t.intVal = intVal;
  return t;
}

Token Lexer::lexString() {
  const SourceLoc start = locHere();
  advance(); // opening '"'
  std::string value;

  auto finishUnterminated = [&](const char* msg) {
    lexError(msg, SourceRange(start, locHere()));
    // Recovery: yield the partial literal so the parser gets a usable token.
    return makeToken(Tok::StringLit, SourceRange(start, locHere()), value);
  };

  while (true) {
    if (atEnd()) return finishUnterminated("unterminated string literal");
    const char ch = cur();
    if (ch == '"') {
      advance();
      break;
    }
    if (ch == '\n') {
      // Strings do not span lines; stop before the newline (CRLF trim).
      if (!value.empty() && value.back() == '\r') value.pop_back();
      return finishUnterminated("unterminated string literal (newline in string)");
    }
    if (ch == '\\') {
      const SourceLoc escStart = locHere();
      advance(); // '\'
      if (atEnd()) return finishUnterminated("unterminated string literal");
      const char esc = advance();
      switch (esc) {
      case '"': value.push_back('"'); break;
      case '\\': value.push_back('\\'); break;
      case 'n': value.push_back('\n'); break;
      case 't': value.push_back('\t'); break;
      case 'r': value.push_back('\r'); break;
      case '0': value.push_back('\0'); break;
      default:
        lexError(std::string("unknown escape sequence '\\") + esc + "'",
                 SourceRange(escStart, locHere()));
        value.push_back(esc); // recovery: keep the character, drop the backslash
        break;
      }
    } else {
      value.push_back(advance());
    }
  }
  return makeToken(Tok::StringLit, SourceRange(start, locHere()), value);
}

Token Lexer::lexIdentOrKeyword() {
  const SourceLoc start = locHere();
  std::string name;
  while (isIdentChar(cur())) name.push_back(advance());
  const SourceRange range(start, locHere());
  const auto& kw = keywordMap();
  const auto it = kw.find(name);
  return makeToken(it != kw.end() ? it->second : Tok::Ident, range, name);
}

std::vector<Token> Lexer::tokenize() {
  std::vector<Token> out;
  while (true) {
    skipWhitespaceAndComments();
    if (atEnd()) {
      const SourceLoc here = locHere();
      out.push_back(makeToken(Tok::Eof, SourceRange(here, here)));
      break;
    }

    const char c = cur();
    if (isIdentStart(c)) {
      out.push_back(lexIdentOrKeyword());
      continue;
    }
    if (isDigitChar(c)) {
      out.push_back(lexNumber());
      continue;
    }
    if (c == '"') {
      out.push_back(lexString());
      continue;
    }

    // Operators & punctuation. On unknown characters: E0001, skip 1 char.
    const SourceLoc start = locHere();
    switch (c) {
    case '+':
      advance();
      out.push_back(makeToken(Tok::Plus, SourceRange(start, locHere())));
      break;
    case '-':
      if (peek(1) == '>') {
        advance();
        advance();
        out.push_back(makeToken(Tok::Arrow, SourceRange(start, locHere())));
      } else {
        advance();
        out.push_back(makeToken(Tok::Minus, SourceRange(start, locHere())));
      }
      break;
    case '*':
      advance();
      out.push_back(makeToken(Tok::Star, SourceRange(start, locHere())));
      break;
    case '/':
      advance();
      out.push_back(makeToken(Tok::Slash, SourceRange(start, locHere())));
      break;
    case '%':
      advance();
      out.push_back(makeToken(Tok::Percent, SourceRange(start, locHere())));
      break;
    case '=':
      if (peek(1) == '=') {
        advance();
        advance();
        out.push_back(makeToken(Tok::EqEq, SourceRange(start, locHere())));
      } else {
        advance();
        out.push_back(makeToken(Tok::Assign, SourceRange(start, locHere())));
      }
      break;
    case '!':
      if (peek(1) == '=') {
        advance();
        advance();
        out.push_back(makeToken(Tok::NotEq, SourceRange(start, locHere())));
      } else {
        advance();
        out.push_back(makeToken(Tok::Not, SourceRange(start, locHere())));
      }
      break;
    case '<':
      if (peek(1) == '=') {
        advance();
        advance();
        out.push_back(makeToken(Tok::Le, SourceRange(start, locHere())));
      } else {
        advance();
        out.push_back(makeToken(Tok::Lt, SourceRange(start, locHere())));
      }
      break;
    case '>':
      if (peek(1) == '=') {
        advance();
        advance();
        out.push_back(makeToken(Tok::Ge, SourceRange(start, locHere())));
      } else {
        advance();
        out.push_back(makeToken(Tok::Gt, SourceRange(start, locHere())));
      }
      break;
    case '&':
      if (peek(1) == '&') {
        advance();
        advance();
        out.push_back(makeToken(Tok::AndAnd, SourceRange(start, locHere())));
      } else {
        advance(); // skip 1 char, continue
        lexError("unexpected character '&' (did you mean '&&'?)",
                 SourceRange(start, locHere()));
      }
      break;
    case '|':
      if (peek(1) == '|') {
        advance();
        advance();
        out.push_back(makeToken(Tok::OrOr, SourceRange(start, locHere())));
      } else {
        advance(); // skip 1 char, continue
        lexError("unexpected character '|' (did you mean '||'?)",
                 SourceRange(start, locHere()));
      }
      break;
    case '.':
      // A lone '.' is never valid: numbers need a digit before the dot, so a
      // leading '.' can only start the '..' range operator.
      if (peek(1) == '.') {
        advance();
        advance();
        out.push_back(makeToken(Tok::DotDot, SourceRange(start, locHere())));
      } else {
        advance(); // skip 1 char, continue
        lexError("unexpected character '.'", SourceRange(start, locHere()));
      }
      break;
    case ',':
      advance();
      out.push_back(makeToken(Tok::Comma, SourceRange(start, locHere())));
      break;
    case ';':
      advance();
      out.push_back(makeToken(Tok::Semi, SourceRange(start, locHere())));
      break;
    case ':':
      advance();
      out.push_back(makeToken(Tok::Colon, SourceRange(start, locHere())));
      break;
    case '(':
      advance();
      out.push_back(makeToken(Tok::LParen, SourceRange(start, locHere())));
      break;
    case ')':
      advance();
      out.push_back(makeToken(Tok::RParen, SourceRange(start, locHere())));
      break;
    case '[':
      advance();
      out.push_back(makeToken(Tok::LBracket, SourceRange(start, locHere())));
      break;
    case ']':
      advance();
      out.push_back(makeToken(Tok::RBracket, SourceRange(start, locHere())));
      break;
    case '{':
      advance();
      out.push_back(makeToken(Tok::LBrace, SourceRange(start, locHere())));
      break;
    case '}':
      advance();
      out.push_back(makeToken(Tok::RBrace, SourceRange(start, locHere())));
      break;
    default:
      advance(); // skip 1 char, continue
      lexError(std::string("unexpected character '") + c + "'",
               SourceRange(start, locHere()));
      break;
    }
  }
  return out;
}

} // namespace ndl
