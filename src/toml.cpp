// NDL v1.0 — TOML subset parser (implementation of include/ndl/toml.hpp)
//
// Supported subset (sufficient for ndl.toml):
//   - [section] headers, including dotted [a.b.c] forms
//   - key = value pairs
//   - values: "strings" (escapes \n \t \r \" \\), int64 (incl. 0x hex),
//     floats, true/false, [ arrays of values ] (multi-line allowed)
//   - inline tables { k = v, ... } — stored as sub-tables of the enclosing
//     table (Table::tables), since Value has no table kind. This makes
//     `name = { path = "..." }` in [dependencies] readable as a nested table.
//   - comments (#), blank lines, inline whitespace, UTF-8 BOM
//
// Not supported (by contract): multiline/basic-lite strings, dotted keys,
// tables inside arrays, dates/times.
//
// Errors are reported as "line N: message" through *errOut (first error wins).
// Value::line / Table::line carry the 1-based line where the item was defined.

#include "ndl/toml.hpp"

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace ndl::toml {
namespace {

bool isBareKeyChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '-';
}

class Parser {
public:
  explicit Parser(const std::string& src, std::string* errOut)
      : src_(src), errOut_(errOut) {
    // Skip a UTF-8 byte-order mark if present.
    if (src_.size() >= 3 && static_cast<unsigned char>(src_[0]) == 0xEF &&
        static_cast<unsigned char>(src_[1]) == 0xBB &&
        static_cast<unsigned char>(src_[2]) == 0xBF)
      pos_ = 3;
  }

  Table run() {
    Table root;
    Table* current = &root;
    while (!failed_) {
      skipAll();
      if (atEnd()) break;
      if (cur() == '[') {
        int hdrLine = line_;
        advance();
        std::vector<std::string> parts;
        if (!parseHeaderName(parts)) break;
        Table* t = &root;
        for (const std::string& part : parts) t = &t->tables[part];
        if (t->line == 0) t->line = hdrLine;
        current = t;
        expectEndOfLine();
        continue;
      }
      parseKeyValue(*current);
    }
    return root;
  }

private:
  // ---------------------------------------------------------------- stream
  bool atEnd() const { return pos_ >= src_.size(); }
  char cur() const { return pos_ < src_.size() ? src_[pos_] : '\0'; }
  void advance() {
    if (pos_ < src_.size()) {
      if (src_[pos_] == '\n') ++line_;
      ++pos_;
    }
  }

  void fail(const std::string& msg) {
    if (!failed_) {
      failed_ = true;
      if (errOut_) *errOut_ = "line " + std::to_string(line_) + ": " + msg;
    }
  }

  // Spaces/tabs/CR only — never crosses a newline.
  void skipInline() {
    while (!atEnd() && (cur() == ' ' || cur() == '\t' || cur() == '\r')) advance();
  }

  // Inline whitespace, newlines (line counter updated) and # comments.
  void skipAll() {
    for (;;) {
      while (!atEnd() && (cur() == ' ' || cur() == '\t' || cur() == '\r' || cur() == '\n'))
        advance();
      if (cur() == '#') {
        while (!atEnd() && cur() != '\n') advance();
        continue;
      }
      break;
    }
  }

  // ---------------------------------------------------------- lexical atoms
  std::string parseKey() {
    std::string key;
    while (!atEnd() && isBareKeyChar(cur())) {
      key += cur();
      advance();
    }
    return key;
  }

  // Matches a bare word (true/false) not followed by a bare-key character.
  bool matchWord(const char* word) {
    size_t n = std::strlen(word);
    if (src_.compare(pos_, n, word) != 0) return false;
    char after = pos_ + n < src_.size() ? src_[pos_ + n] : '\0';
    if (isBareKeyChar(after)) return false;
    for (size_t k = 0; k < n; ++k) advance();
    return true;
  }

  std::string parseString() {
    advance(); // opening quote
    std::string out;
    for (;;) {
      if (atEnd() || cur() == '\n') {
        fail("unterminated string");
        return out;
      }
      char c = cur();
      if (c == '"') {
        advance();
        return out;
      }
      if (c == '\\') {
        advance();
        char e = cur();
        switch (e) {
        case 'n': out += '\n'; advance(); break;
        case 't': out += '\t'; advance(); break;
        case 'r': out += '\r'; advance(); break;
        case '"': out += '"'; advance(); break;
        case '\\': out += '\\'; advance(); break;
        case '\0':
          fail("unterminated string escape");
          return out;
        default:
          fail(std::string("unknown escape sequence '\\") + e + "' in string");
          return out;
        }
        continue;
      }
      out += c;
      advance();
    }
  }

  Value parseNumber() {
    Value v;
    v.line = line_;
    size_t start = pos_;
    while (!atEnd()) {
      char c = cur();
      if (std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '+' || c == '-' || c == '.')
        advance();
      else
        break;
    }
    std::string tok = src_.substr(start, pos_ - start);
    if (tok.empty()) {
      fail("expected a value");
      return v;
    }
    bool hex = tok.size() > 2 && tok[0] == '0' && (tok[1] == 'x' || tok[1] == 'X');
    bool isFloat = !hex && (tok.find('.') != std::string::npos ||
                            tok.find('e') != std::string::npos ||
                            tok.find('E') != std::string::npos);
    errno = 0;
    char* end = nullptr;
    if (isFloat) {
      double d = std::strtod(tok.c_str(), &end);
      if (end != tok.c_str() + tok.size()) {
        fail("invalid float \"" + tok + "\"");
        return v;
      }
      v.k = Value::K::Float;
      v.d = d;
    } else {
      long long i = std::strtoll(tok.c_str(), &end, hex ? 16 : 10);
      if (end != tok.c_str() + tok.size() || errno == ERANGE) {
        fail("invalid integer \"" + tok + "\"");
        return v;
      }
      v.k = Value::K::Int;
      v.i = i;
    }
    return v;
  }

  Value parsePlainValue() {
    Value v;
    skipInline();
    v.line = line_;
    if (atEnd() || cur() == '\n') {
      fail("expected a value");
      return v;
    }
    char c = cur();
    if (c == '"') {
      v.k = Value::K::Str;
      v.s = parseString();
      return v;
    }
    if (c == 't') {
      if (matchWord("true")) {
        v.k = Value::K::Bool;
        v.b = true;
      } else {
        fail("invalid value: expected \"true\"");
      }
      return v;
    }
    if (c == 'f') {
      if (matchWord("false")) {
        v.k = Value::K::Bool;
        v.b = false;
      } else {
        fail("invalid value: expected \"false\"");
      }
      return v;
    }
    if (c == '[') return parseArray();
    if (c == '{') {
      fail("inline tables inside arrays are not supported");
      return v;
    }
    if (std::isdigit(static_cast<unsigned char>(c)) != 0 || c == '+' || c == '-')
      return parseNumber();
    fail(std::string("unexpected character '") + c + "'");
    return v;
  }

  Value parseArray() {
    Value v;
    v.k = Value::K::Array;
    v.line = line_;
    advance(); // '['
    for (;;) {
      skipAll();
      if (failed_) return v;
      if (atEnd()) {
        fail("unterminated array");
        return v;
      }
      if (cur() == ']') {
        advance();
        return v;
      }
      Value e = parsePlainValue();
      if (failed_) return v;
      v.arr.push_back(std::move(e));
      skipAll();
      if (failed_) return v;
      if (cur() == ',') {
        advance();
        continue;
      }
      if (cur() == ']') {
        advance();
        return v;
      }
      fail("expected ',' or ']' in array");
      return v;
    }
  }

  // Body of an inline table (after '{'). Nested inline tables become
  // sub-tables of `t`; scalar/array values land in t.values.
  void parseInlineTableBody(Table& t) {
    for (;;) {
      skipAll();
      if (failed_) return;
      if (atEnd()) {
        fail("unterminated inline table");
        return;
      }
      if (cur() == '}') {
        advance();
        return;
      }
      int stmtLine = line_;
      std::string key = parseKey();
      if (key.empty()) {
        fail("expected a key inside inline table");
        return;
      }
      skipInline();
      if (cur() != '=') {
        fail("expected '=' after key \"" + key + "\" in inline table");
        return;
      }
      advance();
      skipInline();
      if (cur() == '{') {
        advance();
        Table sub;
        sub.line = stmtLine;
        parseInlineTableBody(sub);
        if (failed_) return;
        t.tables[key] = std::move(sub);
      } else {
        Value v = parsePlainValue();
        if (failed_) return;
        t.values[key] = std::move(v);
      }
      skipAll();
      if (failed_) return;
      if (cur() == ',') {
        advance();
        continue;
      }
      if (cur() == '}') {
        advance();
        return;
      }
      fail("expected ',' or '}' in inline table");
      return;
    }
  }

  bool parseHeaderName(std::vector<std::string>& parts) {
    for (;;) {
      skipInline();
      std::string k = parseKey();
      if (k.empty()) {
        fail("expected a table name after '['");
        return false;
      }
      parts.push_back(std::move(k));
      skipInline();
      if (cur() == '.') {
        advance();
        continue;
      }
      if (cur() == ']') {
        advance();
        return true;
      }
      fail("expected '.' or ']' in table header");
      return false;
    }
  }

  void expectEndOfLine() {
    skipInline();
    if (atEnd()) return;
    if (cur() == '\n') {
      advance();
      return;
    }
    if (cur() == '#') {
      while (!atEnd() && cur() != '\n') advance();
      if (!atEnd()) advance();
      return;
    }
    fail("unexpected characters after value (expected end of line)");
  }

  void parseKeyValue(Table& section) {
    int stmtLine = line_;
    std::string key = parseKey();
    if (key.empty()) {
      fail(std::string("expected a key or table header, got '") +
           (atEnd() ? ' ' : cur()) + "'");
      return;
    }
    skipInline();
    if (cur() != '=') {
      fail("expected '=' after key \"" + key + "\"");
      return;
    }
    advance();
    skipInline();
    if (cur() == '{') {
      advance();
      Table t;
      t.line = stmtLine;
      parseInlineTableBody(t);
      if (failed_) return;
      section.tables[key] = std::move(t);
    } else {
      Value v = parsePlainValue();
      if (failed_) return;
      section.values[key] = std::move(v);
    }
    expectEndOfLine();
  }

  const std::string& src_;
  std::string* errOut_;
  size_t pos_ = 0;
  int line_ = 1;
  bool failed_ = false;
};

} // namespace

Table parse(const std::string& source, std::string* errOut) {
  Parser p(source, errOut);
  return p.run();
}

} // namespace ndl::toml
