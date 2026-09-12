// NDL v1.0 — Project manifest (ndl.toml), package manager & import resolution
// Contract header — frozen. Implementation: src/manifest.cpp
#pragma once

#include "ndl/diag.hpp"
#include "ndl/toml.hpp"
#include <string>
#include <vector>

namespace ndl {

struct Dependency {
  std::string name;
  std::string path; // local directory containing pkg.ndl
};

struct Manifest {
  bool loaded = false;            // true when a real ndl.toml was read
  std::string path;               // path to ndl.toml
  std::string rootDir;            // directory containing ndl.toml

  std::string projectName = "ndl-app";
  std::string version = "0.1.0";
  std::string ndlVersion = "1.0";
  std::vector<std::string> authors;
  std::string main = "main.ndl";  // entry source, relative to rootDir

  std::string target = "auto";    // auto | cpu | gpu
  int optLevel = 2;               // 0..3
  bool fastmath = true;
  uint64_t seed = 42;

  std::vector<Dependency> deps;   // [dependencies] name = { path = "..." }
};

// Loads manifest from tomlPath; on structural errors reports E05xx via engine
// and returns a default manifest with loaded=false.
Manifest loadManifest(const std::string& tomlPath, DiagnosticEngine& diag);

// Returns "" if no ndl.toml found walking up from `start` (inclusive).
std::string findProjectRoot(const std::string& startPath);

// ---------------------------------------------------------------------------
// Import resolution
// ---------------------------------------------------------------------------
struct ResolvedSource {
  std::string path;             // absolute or root-relative normalized path
  std::string logicalName;      // path relative to project root, '/'-separated
  std::string source;           // file contents
};

class ImportResolver {
public:
  ImportResolver(DiagnosticEngine& diag, std::string projectRoot, const Manifest& manifest);

  // Resolves the entry file and its transitive imports (DFS post-order:
  // dependencies first, entry last). Cycles → E0401; missing files → E0402.
  // Reads file contents. Main entry is returned LAST.
  std::vector<ResolvedSource> resolve(const std::string& entryFile);

  // Search order for an import path P relative to importer file F:
  //   1. dir(F)/P
  //   2. projectRoot/P
  //   3. NDLC std dir: $NDLC_STD or <exe_dir>/../std/P
  static std::string findImport(const std::string& importerFile,
                                const std::string& projectRoot,
                                const std::string& importPath);

private:
  DiagnosticEngine& diag_;
  std::string root_;
  const Manifest& manifest_;
};

} // namespace ndl
