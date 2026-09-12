// NDL v1.0 — Diagnostics engine (rustc-style rendering)
// Contract header — frozen. Implementation: src/diag.cpp
#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace ndl {

inline constexpr const char* kNdlVersion = "1.0.0";
inline constexpr const char* kNdlcVersion = "1.0.0";

// ---------------------------------------------------------------------------
// Source locations
// ---------------------------------------------------------------------------
struct SourceLoc {
  std::string file;
  uint32_t line = 1;   // 1-based
  uint32_t col = 1;    // 1-based (bytes)
  uint32_t offset = 0; // byte offset from start

  SourceLoc() = default;
  SourceLoc(std::string f, uint32_t l, uint32_t c, uint32_t o)
      : file(std::move(f)), line(l), col(c), offset(o) {}

  std::string toString() const; // "file:line:col"
};

struct SourceRange {
  SourceLoc begin;
  SourceLoc end; // exclusive
  SourceRange() = default;
  SourceRange(SourceLoc b, SourceLoc e) : begin(std::move(b)), end(std::move(e)) {}
  SourceRange(SourceLoc b) : begin(b), end(b) {}
};

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------
enum class Severity { Note, Warning, Error, Fatal };

struct DiagLabel {
  std::string message;
  SourceRange range;
};

struct Diagnostic {
  Severity severity = Severity::Error;
  std::string code;      // e.g. "E0201"
  std::string message;
  SourceRange range;
  std::vector<DiagLabel> labels;   // secondary underlined spans (rendered below main)
  std::vector<std::string> notes;  // "note: ..." and "help: ..." lines
};

class DiagnosticEngine {
public:
  explicit DiagnosticEngine(bool color = supportsColor());

  void report(Diagnostic d);

  // Convenience builders.
  void error(std::string code, SourceRange range, std::string message,
             std::vector<DiagLabel> labels = {}, std::vector<std::string> notes = {});
  void warning(std::string code, SourceRange range, std::string message,
               std::vector<DiagLabel> labels = {}, std::vector<std::string> notes = {});
  void note(SourceRange range, std::string message);

  bool hasErrors() const { return errorCount_ > 0; }
  size_t errorCount() const { return errorCount_; }
  size_t warningCount() const { return warningCount_; }
  const std::vector<Diagnostic>& all() const { return diags_; }

  void setUseColor(bool c) { color_ = c; }

  // Renders every diagnostic in collection order to stderr-like stream.
  void renderAll(std::ostream& os) const;
  // Renders one diagnostic (rustc-style: header, snippet, caret, notes).
  std::string renderOne(const Diagnostic& d) const;

  static bool supportsColor(); // isatty(stderr)

private:
  std::vector<Diagnostic> diags_;
  size_t errorCount_ = 0;
  size_t warningCount_ = 0;
  bool color_ = false;
};

// Edit distance helpers (for "did you mean" suggestions).
size_t levenshtein(std::string_view a, std::string_view b);
// Returns "" when no candidate is close enough (threshold: max(2, len/3)).
std::string bestMatch(std::string_view word, const std::vector<std::string>& candidates);

} // namespace ndl
