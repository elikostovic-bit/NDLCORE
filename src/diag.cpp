// NDL v1.0 — Diagnostic engine implementation (rustc-style rendering).
// Implements: DiagnosticEngine, SourceLoc::toString(), levenshtein(), bestMatch().
// Rendered format (no color):
//
//   error[E0201]: unknown identifier `HiddenLayr`
//     --> main.ndl:14:15
//      |
//   14 | dense_connect(HiddenLayr, Interneurons, density=0.2, weight=0.8);
//      |               ^^^^^^^^^^ unknown identifier
//      |
//   help: did you mean `HiddenLayer`?
#include "ndl/diag.hpp"

#include "ndl/port.hpp"  // stderrIsTty (POSIX/Win32)

#include <algorithm>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <ostream>
#include <sstream>
#include <utility>
#include <vector>

namespace ndl {

// ---------------------------------------------------------------------------
// SourceLoc
// ---------------------------------------------------------------------------
std::string SourceLoc::toString() const {
  std::ostringstream os;
  os << file << ':' << line << ':' << col;
  return os.str();
}

// ---------------------------------------------------------------------------
// Edit distance helpers (for "did you mean" suggestions)
// ---------------------------------------------------------------------------
size_t levenshtein(std::string_view a, std::string_view b) {
  const size_t n = a.size();
  const size_t m = b.size();
  if (n == 0) return m;
  if (m == 0) return n;

  // Standard DP over two rolling rows: O(n*m) time, O(m) space.
  std::vector<size_t> prev(m + 1), row(m + 1);
  for (size_t j = 0; j <= m; ++j) prev[j] = j;
  for (size_t i = 1; i <= n; ++i) {
    row[0] = i;
    for (size_t j = 1; j <= m; ++j) {
      const size_t substitution = prev[j - 1] + (a[i - 1] == b[j - 1] ? 0u : 1u);
      const size_t deletion = prev[j] + 1;
      const size_t insertion = row[j - 1] + 1;
      row[j] = std::min({substitution, deletion, insertion});
    }
    prev.swap(row);
  }
  return prev[m];
}

std::string bestMatch(std::string_view word, const std::vector<std::string>& candidates) {
  const size_t threshold = std::max<size_t>(2, word.size() / 3);
  const size_t kInf = std::numeric_limits<size_t>::max();
  size_t bestDist = kInf;
  const std::string* best = nullptr;
  for (const std::string& cand : candidates) {
    const size_t d = levenshtein(word, cand);
    if (d < bestDist) {
      bestDist = d;
      best = &cand;
    }
  }
  if (best != nullptr && bestDist <= threshold) return *best;
  return "";
}

// ---------------------------------------------------------------------------
// DiagnosticEngine — collection
// ---------------------------------------------------------------------------
DiagnosticEngine::DiagnosticEngine(bool color) : color_(color) {}

void DiagnosticEngine::report(Diagnostic d) {
  if (d.severity == Severity::Error || d.severity == Severity::Fatal) {
    ++errorCount_;
  } else if (d.severity == Severity::Warning) {
    ++warningCount_;
  }
  diags_.push_back(std::move(d));
}

void DiagnosticEngine::error(std::string code, SourceRange range, std::string message,
                             std::vector<DiagLabel> labels, std::vector<std::string> notes) {
  Diagnostic d;
  d.severity = Severity::Error;
  d.code = std::move(code);
  d.range = std::move(range);
  d.message = std::move(message);
  d.labels = std::move(labels);
  d.notes = std::move(notes);
  report(std::move(d));
}

void DiagnosticEngine::warning(std::string code, SourceRange range, std::string message,
                               std::vector<DiagLabel> labels, std::vector<std::string> notes) {
  Diagnostic d;
  d.severity = Severity::Warning;
  d.code = std::move(code);
  d.range = std::move(range);
  d.message = std::move(message);
  d.labels = std::move(labels);
  d.notes = std::move(notes);
  report(std::move(d));
}

void DiagnosticEngine::note(SourceRange range, std::string message) {
  Diagnostic d;
  d.severity = Severity::Note;
  d.range = std::move(range);
  d.message = std::move(message);
  report(std::move(d));
}

bool DiagnosticEngine::supportsColor() { return ndlport::stderrIsTty(); }

// ---------------------------------------------------------------------------
// Rendering internals
// ---------------------------------------------------------------------------
namespace {

constexpr const char* kReset = "\x1b[0m";
constexpr const char* kBoldRed = "\x1b[1;31m";
constexpr const char* kBoldYellow = "\x1b[1;33m";
constexpr const char* kBoldBlue = "\x1b[1;34m";
constexpr const char* kBold = "\x1b[1m";

std::string paint(const std::string& s, const char* code, bool on) {
  if (!on || s.empty()) return s;
  return std::string(code) + s + kReset;
}

// Width (digit count) of a 1-based line number.
uint32_t decimalWidth(uint32_t v) {
  uint32_t w = 1;
  while (v >= 10) {
    v /= 10;
    ++w;
  }
  return w;
}

// Source file cached as lines (0-based). '\r' immediately before '\n' is
// trimmed so CRLF sources align with the lexer's byte columns.
struct CachedSource {
  bool readable = false;
  std::vector<std::string> lines;
};

const CachedSource& cachedSource(const std::string& path) {
  static std::map<std::string, CachedSource> cache;
  const auto it = cache.find(path);
  if (it != cache.end()) return it->second;

  CachedSource cs;
  std::ifstream f(path, std::ios::binary);
  if (f) {
    cs.readable = true;
    const std::string content((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
    std::string line;
    for (const char c : content) {
      if (c == '\n') {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        cs.lines.push_back(line);
        line.clear();
      } else {
        line.push_back(c);
      }
    }
    if (!line.empty()) {
      if (line.back() == '\r') line.pop_back();
      cs.lines.push_back(line);
    }
  }
  return cache.emplace(path, std::move(cs)).first->second;
}

// Short form of a message for the primary caret row: the part before the
// first '`' segment ("unknown identifier `HiddenLayr`" -> "unknown identifier").
// An explicit label whose range equals the diagnostic range always wins.
std::string messageHead(const std::string& msg) {
  const size_t tick = msg.find('`');
  if (tick == std::string::npos) return msg;
  std::string head = msg.substr(0, tick);
  while (!head.empty() && (head.back() == ' ' || head.back() == '\t')) head.pop_back();
  return head;
}

bool sameLoc(const SourceLoc& a, const SourceLoc& b) {
  return a.file == b.file && a.line == b.line && a.col == b.col;
}

const char* severityColor(Severity s) {
  switch (s) {
  case Severity::Error:
  case Severity::Fatal:
    return kBoldRed;
  case Severity::Warning:
    return kBoldYellow;
  case Severity::Note:
    return kBoldBlue;
  }
  return kBoldRed;
}

// Prints one underline row:  "<gutter>|<pad>^^^^ label"
// Carets start under b.col and run for (e.col - b.col), at least 1; a range
// whose end sits on another line is underlined to the end of the line.
void emitUnderline(std::ostream& os, uint32_t numWidth, const SourceLoc& b, const SourceLoc& e,
                   const std::string& lineText, const std::string& label, bool color,
                   const char* code) {
  os << std::string(numWidth + 1, ' ') << '|';
  size_t startIdx = b.col > 0 ? static_cast<size_t>(b.col) - 1 : 0;
  if (startIdx > lineText.size()) startIdx = lineText.size();
  size_t len;
  if (e.line != b.line) {
    len = lineText.size() - startIdx; // to end of line
  } else {
    len = e.col > b.col ? static_cast<size_t>(e.col - b.col) : 1;
  }
  if (len < 1) len = 1;
  os << std::string(startIdx + 1, ' '); // +1 for the space after '|'
  os << paint(std::string(len, '^'), code, color);
  if (!label.empty()) os << ' ' << label;
  os << '\n';
}

} // namespace

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
std::string DiagnosticEngine::renderOne(const Diagnostic& d) const {
  const bool c = color_;
  std::ostringstream os;

  // Header: "error[CODE]: msg" / "warning[CODE]: msg" / "note: msg".
  const bool hasCode = !d.code.empty() && d.severity != Severity::Note;
  const char* headColor = kBold;
  const char* headWord = "error";
  switch (d.severity) {
  case Severity::Error:
  case Severity::Fatal:
    headWord = "error";
    headColor = kBoldRed;
    break;
  case Severity::Warning:
    headWord = "warning";
    headColor = kBoldYellow;
    break;
  case Severity::Note:
    headWord = "note";
    headColor = kBold;
    break;
  }
  std::string head = headWord;
  if (hasCode) head += "[" + d.code + "]";
  head += ":";
  os << paint(head, headColor, c) << ' ' << d.message << '\n';

  const SourceLoc& b = d.range.begin;
  const SourceLoc& e = d.range.end;
  const bool haveFile = !b.file.empty();

  // Primary caret label: an explicit label spanning exactly d.range, else the
  // short (pre-backtick) form of the message itself.
  const DiagLabel* primaryLabel = nullptr;
  for (const DiagLabel& l : d.labels) {
    if (sameLoc(l.range.begin, b) && sameLoc(l.range.end, e)) {
      primaryLabel = &l;
      break;
    }
  }
  const std::string mainLabelText =
      primaryLabel != nullptr ? primaryLabel->message : messageHead(d.message);

  const CachedSource* src = haveFile ? &cachedSource(b.file) : nullptr;
  const std::string* lineText = nullptr;
  if (src != nullptr && src->readable && b.line >= 1 &&
      static_cast<size_t>(b.line - 1) < src->lines.size()) {
    lineText = &src->lines[b.line - 1];
  }

  if (haveFile) {
    const uint32_t nw = decimalWidth(b.line);
    os << std::string(nw, ' ') << "--> " << b.toString() << '\n';

    // Partition labels: same file+line -> extra underline rows; others -> one
    // "--> file:line:col" line each.
    std::vector<const DiagLabel*> sameLine;
    std::vector<const DiagLabel*> other;
    for (const DiagLabel& l : d.labels) {
      if (&l == primaryLabel) continue; // rendered inline on the main caret row
      const bool same = lineText != nullptr && !l.range.begin.file.empty() &&
                        l.range.begin.file == b.file && l.range.begin.line == b.line;
      (same ? sameLine : other).push_back(&l);
    }

    if (lineText != nullptr) {
      // Blank gutter, source line, main underline, then same-line labels.
      os << std::string(nw + 1, ' ') << "|\n";
      os << b.line << " | " << *lineText << '\n';
      emitUnderline(os, nw, b, e, *lineText, mainLabelText, c, severityColor(d.severity));
      for (const DiagLabel* l : sameLine) {
        emitUnderline(os, nw, l->range.begin, l->range.end, *lineText, l->message, c,
                      severityColor(d.severity));
      }
    }

    const bool hasTail = !other.empty() || !d.notes.empty();
    if (lineText != nullptr && hasTail) os << std::string(nw + 1, ' ') << "|\n";

    for (const DiagLabel* l : other) {
      if (l->range.begin.file.empty()) continue;
      os << std::string(nw, ' ') << "--> " << l->range.begin.toString();
      if (!l->message.empty()) os << ' ' << l->message;
      os << '\n';
    }
  }

  // Notes ("note: ...") and helps ("help: ...", rendered as-is in blue).
  for (const std::string& n : d.notes) {
    if (n.rfind("help:", 0) == 0) {
      os << paint(n, kBoldBlue, c) << '\n';
    } else if (n.rfind("note:", 0) == 0) {
      os << paint("note:", kBold, c) << n.substr(5) << '\n';
    } else {
      os << paint("note:", kBold, c) << ' ' << n << '\n';
    }
  }

  return os.str();
}

void DiagnosticEngine::renderAll(std::ostream& os) const {
  // renderOne() always ends with a newline, so a single '\n' between blocks
  // yields rustc's blank-line separation; the final line stays terminated.
  for (size_t i = 0; i < diags_.size(); ++i) {
    if (i != 0) os << '\n';
    os << renderOne(diags_[i]);
  }
}

} // namespace ndl
