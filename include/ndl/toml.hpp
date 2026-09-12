// NDL v1.0 — Minimal TOML parser (subset) for ndl.toml
// Contract header — frozen. Implementation: src/toml.cpp
//
// Supported subset (sufficient for ndl.toml):
//   [section] headers (one nesting level), key = value,
//   value := "string" | integer | float | true | false | [ array-of-values ],
//   comments (#), blank lines, inline whitespace. No multiline strings,
//   no tables-of-tables, no dotted keys.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ndl::toml {

struct Value {
  enum class K { Str, Int, Float, Bool, Array } k = K::Str;
  std::string s;
  int64_t i = 0;
  double d = 0.0;
  bool b = false;
  std::vector<Value> arr;
  int line = 0; // 1-based line where defined

  bool isStr() const { return k == K::Str; }
  bool isInt() const { return k == K::Int; }
  bool isFloat() const { return k == K::Float; }
  bool isBool() const { return k == K::Bool; }
  bool isArray() const { return k == K::Array; }
};

struct Table {
  std::map<std::string, Value> values;
  std::map<std::string, Table> tables;
  int line = 0;

  bool has(const std::string& key) const { return values.count(key) != 0; }
  bool hasTable(const std::string& key) const { return tables.count(key) != 0; }
};

// Parses TOML text. On failure returns a table with whatever parsed and sets
// *errOut (if non-null) to "line N: message".
Table parse(const std::string& source, std::string* errOut);

} // namespace ndl::toml
