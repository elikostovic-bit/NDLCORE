// NDL v1.0 — manifest (ndl.toml) loading & import resolution
// (implementation of include/ndl/manifest.hpp)
//
// loadManifest: parses ndl.toml with the ndl::toml subset parser, validates
// the structure ([project], [build], [simulation], [dependencies]) and fills
// the Manifest. Structural errors are reported as E0502 through the
// DiagnosticEngine and `loaded` stays false.
//
// ImportResolver: DFS over `import "P";` statements (collected with the real
// Lexer), post-order output (dependencies first, entry last). Cycles → E0401,
// unreadable/unresolvable files → E0402. Manifest deps (<root>/<dep>/pkg.ndl)
// are resolved before the entry file.

#include "ndl/manifest.hpp"
#include "ndl/lexer.hpp"
#include "ndl/port.hpp"  // selfExePath (POSIX/Win32)

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <utility>
#include <vector>

namespace ndl {
namespace {

// ------------------------------------------------------------- fs helpers
bool fileExists(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool readFile(const std::string& path, std::string& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}

std::string dirName(const std::string& p) {
  size_t slash = p.find_last_of("/\\");
  if (slash == std::string::npos) return ".";
  if (slash == 0) return "/";
  return p.substr(0, slash);
}

std::string baseName(const std::string& p) {
  size_t slash = p.find_last_of("/\\");
  if (slash == std::string::npos) return p;
  return p.substr(slash + 1);
}

std::string joinPath(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (b.empty()) return a;
  if (b.front() == '/' || b.front() == '\\') return b; // absolute component wins
  if (a.back() == '/') return a + b;
  return a + "/" + b;
}

// Lexical path normalization without realpath(3): collapses "//", "/./" and
// "/../" as far as the components allow; keeps a leading '/'; a relative
// result that would escape its base keeps leading ".." components.
// Backslashes are normalized to '/' first (Windows inputs).
std::string normalizePath(const std::string& raw) {
  std::string p;
  p.reserve(raw.size());
  for (char c : raw) p += (c == '\\') ? '/' : c;
  if (p.empty()) return p;
  // POSIX absolute, Windows drive-letter path ("C:/...") or UNC ("//...").
  bool isAbs = p.front() == '/' || (p.size() >= 2 && p[1] == ':');
  std::vector<std::string> parts;
  std::string comp;
  auto flush = [&]() {
    if (comp.empty() || comp == ".") {
      comp.clear();
      return;
    }
    if (comp == "..") {
      const bool atDriveRoot = parts.size() == 1 && parts[0].size() == 2 &&
                               parts[0][1] == ':';  // "C:" — above it is nothing
      if (!parts.empty() && parts.back() != ".." && !atDriveRoot) {
        parts.pop_back();
      } else if (!isAbs) {
        parts.push_back("..");
      }
      // absolute paths cannot go above '/': drop the component
      comp.clear();
      return;
    }
    parts.push_back(std::move(comp));
    comp.clear();
  };
  for (char c : p) {
    if (c == '/') flush();
    else comp += c;
  }
  flush();
  if (parts.empty()) return isAbs ? "/" : ".";
  // Windows drive ("C:") is itself the root: no leading '/' before it.
  const bool driveRoot = parts[0].size() == 2 && parts[0][1] == ':';
  std::string out = (isAbs && !driveRoot) ? "/" : "";
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i != 0) out += "/";
    out += parts[i];
  }
  return out;
}

// ----------------------------------------------------------------- diag
void reportManifestError(DiagnosticEngine& diag, const std::string& file, int line,
                         const std::string& msg) {
  std::ostringstream ss;
  ss << file;
  if (line > 0) ss << ":" << line;
  ss << ": " << msg;
  diag.error("E0502", SourceRange(), ss.str());
}

const toml::Value* valueOf(const toml::Table& t, const char* key) {
  auto it = t.values.find(key);
  return it == t.values.end() ? nullptr : &it->second;
}

// --------------------------------------------------- import resolution DFS
struct ResolveState {
  DiagnosticEngine& diag;
  std::string root;
  std::set<std::string> visited;    // completed files (canonical paths)
  std::set<std::string> inProgress; // files on the current DFS stack
  std::vector<ResolvedSource> order;

  ResolveState(DiagnosticEngine& d, std::string r) : diag(d), root(std::move(r)) {}
};

// Path relative to the project root ('/'-separated); falls back to the
// canonical path itself when it is not under the root.
std::string logicalNameFor(const std::string& root, const std::string& canon) {
  if (root.empty() || canon.empty()) return canon;
  if (canon == root) return baseName(canon);
  std::string prefix = root;
  if (prefix.back() != '/') prefix += '/';
  if (canon.compare(0, prefix.size(), prefix) == 0) return canon.substr(prefix.size());
  return canon;
}

void resolveFile(ResolveState& st, const std::string& path) {
  if (path.empty()) return;
  std::string canon = normalizePath(path);
  if (canon.empty()) return;
  if (st.visited.count(canon) != 0) return;
  if (st.inProgress.count(canon) != 0) {
    st.diag.error("E0401", SourceRange(), "cyclic import: " + canon);
    return;
  }

  std::string content;
  if (!readFile(canon, content)) {
    st.diag.error("E0402", SourceRange(), "cannot read import: " + canon);
    return;
  }

  st.inProgress.insert(canon);

  // Collect `import "P";` and `use Name;` statements with a real lexer pass.
  // (`use` binds a manifest dependency whose files are pre-resolved from
  // [dependencies], so it needs no file resolution here.)
  Lexer lexer(canon, content, st.diag);
  std::vector<Token> toks = lexer.tokenize();
  for (size_t i = 0; i + 1 < toks.size(); ++i) {
    if (toks[i].kind == Tok::Kw_import && toks[i + 1].kind == Tok::StringLit) {
      std::string target = ImportResolver::findImport(canon, st.root, toks[i + 1].text);
      if (target.empty()) {
        st.diag.error("E0402", SourceRange(),
                      "cannot resolve import \"" + toks[i + 1].text +
                          "\" (from " + canon + ")");
        continue;
      }
      resolveFile(st, target); // DFS: imports complete before the importer
    }
  }

  st.inProgress.erase(canon);
  st.visited.insert(canon);

  ResolvedSource rs;
  rs.path = canon;
  rs.logicalName = logicalNameFor(st.root, canon);
  rs.source = std::move(content);
  st.order.push_back(std::move(rs));
}

} // namespace

// ------------------------------------------------------------------ API

Manifest loadManifest(const std::string& tomlPath, DiagnosticEngine& diag) {
  Manifest m;
  m.path = tomlPath;
  m.rootDir = dirName(normalizePath(tomlPath));

  std::string src;
  if (!readFile(tomlPath, src)) {
    reportManifestError(diag, tomlPath, 0, "cannot read manifest file");
    return m; // loaded == false
  }

  std::string err;
  toml::Table root = toml::parse(src, &err);
  if (!err.empty()) {
    reportManifestError(diag, tomlPath, 0, "TOML parse error: " + err);
    return m; // loaded == false
  }

  bool ok = true;
  auto bad = [&](int line, const std::string& msg) {
    ok = false;
    reportManifestError(diag, tomlPath, line, msg);
  };

  // [project]
  if (root.hasTable("project")) {
    const toml::Table& p = root.tables.at("project");
    if (const toml::Value* v = valueOf(p, "name")) {
      if (v->isStr()) m.projectName = v->s;
      else bad(v->line, "[project] name must be a string");
    }
    if (const toml::Value* v = valueOf(p, "version")) {
      if (v->isStr()) m.version = v->s;
      else bad(v->line, "[project] version must be a string");
    }
    if (const toml::Value* v = valueOf(p, "ndl_version")) {
      if (v->isStr()) m.ndlVersion = v->s;
      else bad(v->line, "[project] ndl_version must be a string");
    }
    if (const toml::Value* v = valueOf(p, "main")) {
      if (v->isStr()) m.main = v->s;
      else bad(v->line, "[project] main must be a string");
    }
    if (const toml::Value* v = valueOf(p, "authors")) {
      if (!v->isArray()) {
        bad(v->line, "[project] authors must be an array of strings");
      } else {
        for (const toml::Value& e : v->arr) {
          if (!e.isStr()) {
            bad(e.line, "[project] authors must contain only strings");
            break;
          }
          m.authors.push_back(e.s);
        }
      }
    }
  }

  // [build]
  if (root.hasTable("build")) {
    const toml::Table& b = root.tables.at("build");
    if (const toml::Value* v = valueOf(b, "target")) {
      if (!v->isStr()) {
        bad(v->line, "[build] target must be a string (auto | cpu | gpu)");
      } else if (v->s == "auto" || v->s == "cpu" || v->s == "gpu") {
        m.target = v->s;
      } else {
        bad(v->line, "[build] target must be one of: auto, cpu, gpu (got \"" + v->s + "\")");
      }
    }
    if (const toml::Value* v = valueOf(b, "opt_level")) {
      if (!v->isInt()) {
        bad(v->line, "[build] opt_level must be an integer (0..3)");
      } else {
        m.optLevel = static_cast<int>(std::clamp(v->i, int64_t{0}, int64_t{3}));
      }
    }
    if (const toml::Value* v = valueOf(b, "fastmath")) {
      if (!v->isBool()) bad(v->line, "[build] fastmath must be true or false");
      else m.fastmath = v->b;
    }
  }

  // [simulation]
  if (root.hasTable("simulation")) {
    const toml::Table& s = root.tables.at("simulation");
    if (const toml::Value* v = valueOf(s, "seed")) {
      if (!v->isInt()) bad(v->line, "[simulation] seed must be an integer");
      else if (v->i < 0) bad(v->line, "[simulation] seed must be non-negative");
      else m.seed = static_cast<uint64_t>(v->i);
    }
  }

  // [dependencies]
  if (root.hasTable("dependencies")) {
    const toml::Table& d = root.tables.at("dependencies");
    // Canonical form: name = { path = "..." } (also [dependencies.name] headers).
    for (const auto& entry : d.tables) {
      const std::string& name = entry.first;
      const toml::Table& dt = entry.second;
      const toml::Value* pv = valueOf(dt, "path");
      if (pv == nullptr || !pv->isStr()) {
        bad(dt.line, "[dependencies] \"" + name + "\" must define a string \"path\"");
        continue;
      }
      Dependency dep;
      dep.name = name;
      dep.path = pv->s;
      m.deps.push_back(std::move(dep));
    }
    // Lenient form: name = "path" (a plain string means the package directory).
    for (const auto& entry : d.values) {
      const std::string& name = entry.first;
      const toml::Value& v = entry.second;
      if (d.tables.count(name) != 0) continue; // table form wins
      if (v.isStr()) {
        Dependency dep;
        dep.name = name;
        dep.path = v.s;
        m.deps.push_back(std::move(dep));
      } else {
        bad(v.line, "[dependencies] \"" + name +
                        "\" must be a table { path = \"...\" } or a string path");
      }
    }
  }

  m.loaded = ok;
  return m;
}

std::string findProjectRoot(const std::string& startPath) {
  std::string dir;
  struct stat st;
  if (!startPath.empty() && ::stat(startPath.c_str(), &st) == 0 && S_ISREG(st.st_mode))
    dir = dirName(normalizePath(startPath)); // file → start at its directory
  else
    dir = normalizePath(startPath);
  if (dir.empty()) dir = ".";
  for (;;) {
    if (fileExists(joinPath(dir, "ndl.toml"))) return dir;
    std::string parent = dirName(dir);
    if (parent == dir || parent.empty()) return ""; // reached '/' (or ".")
    dir = parent;
  }
}

ImportResolver::ImportResolver(DiagnosticEngine& diag, std::string projectRoot,
                               const Manifest& manifest)
    : diag_(diag), root_(normalizePath(projectRoot)), manifest_(manifest) {}

std::vector<ResolvedSource> ImportResolver::resolve(const std::string& entryFile) {
  ResolveState st(diag_, root_);

  // 1. Manifest dependencies first: <root>/<dep.path>/pkg.ndl, post-order each
  //    (their own imports are resolved recursively the same way).
  for (const Dependency& dep : manifest_.deps) {
    std::string pkg = normalizePath(joinPath(joinPath(root_, dep.path), "pkg.ndl"));
    resolveFile(st, pkg);
  }

  // 2. Entry file last (its own imports complete before it: DFS post-order).
  if (!entryFile.empty()) resolveFile(st, entryFile);

  return st.order;
}

std::string ImportResolver::findImport(const std::string& importerFile,
                                       const std::string& projectRoot,
                                       const std::string& importPath) {
  if (importPath.empty()) return "";

  // 1. Relative to the importing file's directory.
  std::string c1 = normalizePath(joinPath(dirName(importerFile), importPath));
  if (fileExists(c1)) return c1;

  // 2. Relative to the project root.
  if (!projectRoot.empty()) {
    std::string c2 = normalizePath(joinPath(projectRoot, importPath));
    if (fileExists(c2)) return c2;
  }

  // 3. Standard library: $NDLC_STD or <exe_dir>/../std.
  std::string stdDir;
  if (const char* env = ::getenv("NDLC_STD")) {
    if (env[0] != '\0') stdDir = env;
  }
  if (stdDir.empty()) {
    std::string exe = ndlport::selfExePath();
    if (!exe.empty()) {
      std::string exeDir = dirName(exe);
      stdDir = joinPath(exeDir + "/..", "std");
    }
  }
  if (!stdDir.empty()) {
    std::string c3 = normalizePath(joinPath(stdDir, importPath));
    if (fileExists(c3)) return c3;
  }
  return "";
}

} // namespace ndl
