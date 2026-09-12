// NDL v1.0 — external toolchain driver (implementation of include/ndl/linker.hpp)
//
// The ndlc compiler itself has NO LLVM library dependency: it emits LLVM IR
// text and PTX text and drives external tools found in $PATH:
//   native build:  clang++ app.ll -O<level> libndl_rt.a -lpthread -ldl -lm -o app
//              or  llc app.ll -filetype=obj -O=<level> -o app.o
//                  g++ app.o libndl_rt.a -lpthread -ldl -lm -o app
//   gpu tools:     ptxas -arch=sm_70 (optional, static cubin)
//
// All subprocesses run via /bin/sh (std::system) with output redirected into
// a temp file that is captured into the build log. No LLVM libraries linked.

#include "ndl/linker.hpp"
#include "ndl/port.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#if defined(_WIN32)
  #include <io.h>       // access
#else
  #include <sys/wait.h> // WIFEXITED/WEXITSTATUS
  #include <unistd.h>   // access, X_OK
#endif
#include <vector>

namespace ndl {
namespace {

bool fileExists(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

[[maybe_unused]] bool endsWithSuffix(const std::string& s, const char* suf) {
  const size_t n = std::strlen(suf);
  return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

bool isExecutableFile(const std::string& p) {
  struct stat st;
  if (::stat(p.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return false;
#if defined(_WIN32)
  // Windows has no exec bit: a regular file with an executable suffix is
  // launchable. Callers try both bare and ".exe"-suffixed names.
  const std::string lows = [&] {
    std::string s;
    for (char c : p) s += static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    return s;
  }();
  return endsWithSuffix(lows, ".exe") || endsWithSuffix(lows, ".bat") ||
         endsWithSuffix(lows, ".cmd");
#else
  return ::access(p.c_str(), X_OK) == 0;
#endif
}

// Strip surrounding whitespace and double quotes. Hand-edited Windows PATH
// entries often look like "C:\\Program Files\\LLVM\\bin" (literal quotes),
// which would otherwise break stat() on every lookup.
std::string trimEntry(const std::string& in) {
  size_t b = 0, e = in.size();
  while (b < e && (in[b] == ' ' || in[b] == '\t' || in[b] == '"')) ++b;
  while (e > b && (in[e - 1] == ' ' || in[e - 1] == '\t' || in[e - 1] == '"')) --e;
  return in.substr(b, e - b);
}

std::vector<std::string> splitPathEnv(const char* env) {
  std::vector<std::string> dirs;
  if (env == nullptr) return dirs;
#if defined(_WIN32)
  const char sep = ';';  // Windows PATH contains drive letters — ':' would split "C:"
#else
  const char sep = ':';
#endif
  std::string s(env);
  size_t start = 0;
  while (start <= s.size()) {
    size_t colon = s.find(sep, start);
    if (colon == std::string::npos) {
      dirs.push_back(trimEntry(s.substr(start)));
      break;
    }
    dirs.push_back(trimEntry(s.substr(start, colon - start)));
    start = colon + 1;
  }
  return dirs;
}

// Directories probed after $PATH when looking for LLVM tools. Covers the
// llvm.org Windows installer default location even when "Add LLVM to PATH"
// was not ticked (or the shell was opened before installation), plus an
// explicit override via NDLC_LLVM_PREFIX / LLVM_PREFIX.
std::vector<std::string> extraLlvmDirs() {
  std::vector<std::string> dirs;
  auto addEnv = [&dirs](const char* env) {
    const char* v = ::getenv(env);
    if (v == nullptr || *v == '\0') return;
    std::string d(v);
    while (!d.empty() && (d.back() == '/' || d.back() == '\\')) d.pop_back();
    if (!d.empty()) dirs.push_back(d + "/bin");
  };
  addEnv("NDLC_LLVM_PREFIX");  // user override, highest priority
  addEnv("LLVM_PREFIX");
#if defined(_WIN32)
  const char* roots[] = {"ProgramFiles", "ProgramFiles(x86)", "LocalAppData"};
  for (const char* envName : roots) {
    const char* v = ::getenv(envName);
    if (v == nullptr || *v == '\0') continue;
    dirs.push_back(std::string(v) + "/LLVM/bin");
  }
#endif
  return dirs;
}

std::string findInPath(const std::string& name,
                       const std::vector<std::string>& extraDirs = {}) {
  const bool hasDirSep = name.find('/') != std::string::npos
#if defined(_WIN32)
                         || name.find('\\') != std::string::npos
#endif
      ;
  if (hasDirSep) return isExecutableFile(name) ? name : std::string();

  // Candidate directories: $PATH first, then well-known install locations.
  std::vector<std::string> all;
  const char* pathEnv = ::getenv("PATH");
  if (pathEnv != nullptr) {
    std::vector<std::string> fromPath = splitPathEnv(pathEnv);
    all.insert(all.end(), fromPath.begin(), fromPath.end());
  }
  all.insert(all.end(), extraDirs.begin(), extraDirs.end());

#if defined(_WIN32)
  const std::string bare = name;
  const std::string exe = name + ".exe";
  const std::string* trials[] = {&bare, &exe, nullptr};
  for (const std::string* leaf : trials) {
    if (leaf == nullptr) break;
    for (const std::string& dir : all) {
      if (dir.empty()) continue;
      std::string cand = dir + "/" + *leaf;
      if (isExecutableFile(cand)) return cand;
    }
  }
  return "";
#else
  for (const std::string& dir : all) {
    if (dir.empty()) continue;
    std::string cand = dir + "/" + name;
    if (isExecutableFile(cand)) return cand;
  }
  return "";
#endif
}

// First hit among several names — used for distro-versioned LLVM binaries
// (Debian/Ubuntu ship clang++-18, llc-18, ... without unversioned aliases).
[[maybe_unused]] std::string findFirstOf(const std::vector<std::string>& names,
                                         const std::vector<std::string>& extraDirs) {
  for (const std::string& n : names) {
    std::string p = findInPath(n, extraDirs);
    if (!p.empty()) return p;
  }
  return "";
}

// Quote a path for the platform shell (sh single quotes / cmd.exe double quotes).
std::string shellQuote(const std::string& s) { return ndlport::shellQuote(s); }

// Runs `cmd` through the platform shell with stdout+stderr redirected into a
// temp file; appends the captured output to `log`. Returns the tool exit code
// (or -1).
int runCapture(const std::string& cmd, std::string& log, bool verbose) {
  std::string tmpl = ndlport::makeTempFilePath("ndlc-tool");
  if (tmpl.empty()) {
    log += "ndlc: cannot create temporary file for tool output\n";
    return -1;
  }

  std::string full = cmd + " > " + shellQuote(tmpl) + " 2>&1";
  if (verbose) log += "+ " + cmd + "\n";
  int status = std::system(ndlport::wrapForCmd(full).c_str());
#if defined(_WIN32)
  // std::system() on Windows returns the command's exit code directly.
  int code = (status == -1) ? -1 : status;
#else
  int code = -1;
  if (status != -1 && WIFEXITED(status)) code = WEXITSTATUS(status);
#endif

  std::ifstream f(tmpl, std::ios::binary);
  if (f) {
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string out = ss.str();
    if (!out.empty()) {
      log += out;
      if (out.back() != '\n') log += '\n';
    }
  }
  ::unlink(tmpl.c_str());
  return code;
}

// First line of `cmd`'s stdout (best effort).
std::string firstLineOf(const std::string& cmd) {
#if defined(_WIN32)
  FILE* fp = ::_popen(ndlport::wrapForCmd(cmd).c_str(), "r");
#else
  FILE* fp = ::popen(cmd.c_str(), "r");
#endif
  if (fp == nullptr) return "";
  std::string line;
  int c;
  while ((c = fgetc(fp)) != EOF && c != '\n') line += static_cast<char>(c);
#if defined(_WIN32)
  ::_pclose(fp);
#else
  ::pclose(fp);
#endif
  return line;
}

// "Ubuntu clang version 17.0.1 (+llvm ...)" → "17.0.1"
std::string extractVersionToken(const std::string& line) {
  for (size_t i = 0; i < line.size(); ++i) {
    if (std::isdigit(static_cast<unsigned char>(line[i])) != 0) {
      size_t j = i;
      while (j < line.size() && (std::isdigit(static_cast<unsigned char>(line[j])) != 0 ||
                                 line[j] == '.'))
        ++j;
      while (j > i && line[j - 1] == '.') --j;
      return line.substr(i, j - i);
    }
  }
  return "";
}

std::string probeVersion(const std::string& path) {
  if (path.empty()) return "";
  std::string line = firstLineOf(shellQuote(path) + " --version " + ndlport::devNullRedir());
  std::string v = extractVersionToken(line);
  return v.empty() ? line : v;
}

int clampLevel(int level) { return level < 0 ? 0 : (level > 3 ? 3 : level); }

// Directory part of a path ("", ".", or up to the last '/" or '\\').
std::string dirOf(const std::string& p) {
  size_t slash = p.find_last_of("/\\");
  if (slash == std::string::npos) return ".";
  return p.substr(0, slash);
}

} // namespace

// ------------------------------------------------------------------ API

std::string Toolchain::summary() const {
  auto item = [](bool has, const std::string& path) -> std::string {
    if (!has) return "-"; // tool not found (ASCII: safe on cp866/cp1251 consoles)
    std::string v = probeVersion(path);
    return v.empty() ? std::string("?") : v;
  };
  return "clang " + item(hasClang, clangPath) + "; llc " + item(hasLlc, llcPath) +
         "; opt " + item(hasOpt, optPath) + "; g++ " + item(hasGpp, gppPath) +
         "; nvcc " + item(hasNvcc, nvccPath) + "; ptxas " + item(hasPtxas, ptxasPath);
}

Toolchain detectToolchain() {
  Toolchain t;
  const std::vector<std::string> llvmDirs = extraLlvmDirs();

#if defined(_WIN32)
  t.clangPath = findInPath("clang++", llvmDirs);
  t.llcPath = findInPath("llc", llvmDirs);
  t.optPath = findInPath("opt", llvmDirs);
#else
  // Debian/Ubuntu name LLVM tools with a version suffix.
  std::vector<std::string> clangNames = {"clang++"};
  std::vector<std::string> llcNames = {"llc"};
  std::vector<std::string> optNames = {"opt"};
  for (int v = 19; v >= 14; --v) {
    const std::string suf = "-" + std::to_string(v);
    clangNames.push_back("clang++" + suf);
    llcNames.push_back("llc" + suf);
    optNames.push_back("opt" + suf);
  }
  t.clangPath = findFirstOf(clangNames, llvmDirs);
  t.llcPath = findFirstOf(llcNames, llvmDirs);
  t.optPath = findFirstOf(optNames, llvmDirs);
#endif
  // Whatever dir clang came from is the best candidate for its siblings
  // (e.g. VS-bundled clang keeps lld-link/llc in the same directory).
  if (!t.clangPath.empty()) {
    const std::string clangDir = dirOf(t.clangPath);
    if (t.llcPath.empty()) t.llcPath = findInPath("llc", {clangDir});
    if (t.optPath.empty()) t.optPath = findInPath("opt", {clangDir});
  }
  t.hasClang = !t.clangPath.empty();
  t.hasLlc = !t.llcPath.empty();
  t.hasOpt = !t.optPath.empty();
  t.gppPath = findInPath("g++");
  t.hasGpp = !t.gppPath.empty();
  t.nvccPath = findInPath("nvcc");
  t.hasNvcc = !t.nvccPath.empty();
  t.ptxasPath = findInPath("ptxas");
  t.hasPtxas = !t.ptxasPath.empty();

  // LLVM version: from clang++ (preferred) or llc, best effort.
  std::string vline;
  if (t.hasClang)
    vline = firstLineOf(shellQuote(t.clangPath) + " --version " + ndlport::devNullRedir());
  else if (t.hasLlc)
    vline = firstLineOf(shellQuote(t.llcPath) + " --version " + ndlport::devNullRedir());
  std::string v = extractVersionToken(vline);
  t.llvmVersion = v.empty() ? "unknown" : v;
  return t;
}

NativeBuildResult buildNative(const std::string& llPath, const std::string& outExeIn,
                              const std::string& runtimeLib,
                              const std::string& runtimeSrcDir, int optLevel,
                              const Toolchain& tools, bool verbose) {
  NativeBuildResult r;
  std::string log;
  int level = clampLevel(optLevel);

#if defined(_WIN32)
  // MinGW g++/clang auto-append ".exe" when the output name has no suffix —
  // keep our existence checks in sync.
  std::string outExe = outExeIn;
  {
    std::string lows;
    for (char c : outExe) lows += static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    if (!endsWithSuffix(lows, ".exe")) outExe += ".exe";
  }
#else
  const std::string& outExe = outExeIn;
#endif
#if defined(_WIN32)
  // No libdl on Windows; winpthreads provides the std::thread symbols.
  const char* sysLibs = "-lwinpthread";
#else
  const char* sysLibs = "-lpthread -ldl -lm";
#endif

  // ---- Route 1: clang++ + runtime compiled FROM SOURCE with the same clang
  // ---- (always ABI-consistent; works with llvm.org clang, VS-bundled clang
  // ----  and llvm-mingw alike — no MSVC-vs-MinGW object mixing possible).
  if (tools.hasClang && !runtimeSrcDir.empty()) {
    const std::string objDir = dirOf(outExe) + "/ndl_rt_objs";
    ndlport::makeDir(objDir);
    std::vector<std::string> sources = ndlport::listFiles(runtimeSrcDir, ".cpp");
    if (sources.empty()) {
      log += "ndlc: no runtime sources (*.cpp) found in " + runtimeSrcDir +
             " -- falling back to the prebuilt archive\n";
    } else {
      bool rtOk = true;
      std::string objFlags;
      for (const std::string& src : sources) {
        const std::string base = src.substr(0, src.size() - 4);  // strip ".cpp"
        const std::string obj = objDir + "/" + base + ".o";
        const std::string cmd = shellQuote(tools.clangPath) + " -std=c++20 -O" +
                                std::to_string(level) + " -I" +
                                shellQuote(runtimeSrcDir) + " -c " +
                                shellQuote(runtimeSrcDir + "/" + src) + " -o " +
                                shellQuote(obj);
        const int rc = runCapture(cmd, log, verbose);
        if (rc != 0 || !fileExists(obj)) {
          log += "ndlc: runtime source compile failed: " + src +
                 " (exit " + std::to_string(rc) + ")\n";
          rtOk = false;
          break;
        }
        objFlags += " " + shellQuote(obj);
      }
      if (rtOk) {
#if defined(_WIN32)
        // MSVC CRT autolinks its threading runtime; llvm-mingw drivers add
        // winpthread themselves — nothing extra needed in either case.
        const char* rtSysLibs = "";
#else
        const char* rtSysLibs = "-lpthread -ldl -lm";
#endif
        const std::string cmd =
            shellQuote(tools.clangPath) + " -Wno-override-module -O" +
            std::to_string(level) + " " + shellQuote(llPath) + objFlags + " " +
            rtSysLibs + " -o " + shellQuote(outExe);
        const int rc = runCapture(cmd, log, verbose);
        if (rc == 0 && fileExists(outExe)) {
          r.ok = true;
          r.artifact = outExe;
        } else {
          log += "ndlc: clang++ link failed (exit " + std::to_string(rc) + ")\n";
        }
      }
    }
  }

  // ---- Route 2: llc → object, then g++ links the prebuilt archive.
  // ---- Also reached when clang++ exists but its routes failed.
  if (!r.ok && tools.hasLlc) {
    std::string obj = outExe + ".o";
    // PIC is a POSIX/PIE concern; Windows COFF objects are relocatable by
    // default and MinGW g++ links them as-is.
#if defined(_WIN32)
    const char* picFlag = "";
#else
    const char* picFlag = " -relocation-model=pic";
#endif
    std::string c1 = shellQuote(tools.llcPath) + " " + shellQuote(llPath) +
                     " -filetype=obj" + picFlag + " -O=" + std::to_string(level) +
                     " -o " + shellQuote(obj);
    int rc1 = runCapture(c1, log, verbose);
    if (rc1 != 0 || !fileExists(obj)) {
      log += "ndlc: llc compilation failed (exit " + std::to_string(rc1) + ")\n";
    } else if (!tools.hasGpp) {
      log += "ndlc: no g++ found -- cannot link the object file\n";
    } else {
      std::string c2 = shellQuote(tools.gppPath) + " " + shellQuote(obj) + " " +
                       shellQuote(runtimeLib) + " " + sysLibs + " -o " +
                       shellQuote(outExe);
      int rc2 = runCapture(c2, log, verbose);
      if (rc2 == 0 && fileExists(outExe)) {
        r.ok = true;
        r.artifact = outExe;
      } else {
        log += "ndlc: native link failed (exit " + std::to_string(rc2) + ")\n";
      }
    }
  }
  // ---- Route 3: clang++ + prebuilt archive directly (last resort — the
  // ---- archive ABI must match the clang target, which is not guaranteed;
  // ---- kept for setups where routes 1–2 are unavailable).
  if (!r.ok && tools.hasClang) {
    std::string cmd = shellQuote(tools.clangPath) + " -Wno-override-module -O" +
                      std::to_string(level) + " " + shellQuote(llPath) +
                      " " + shellQuote(runtimeLib) +
                      " " + sysLibs + " -o " + shellQuote(outExe);
    int rc = runCapture(cmd, log, verbose);
    if (rc == 0 && fileExists(outExe)) {
      r.ok = true;
      r.artifact = outExe;
    } else {
      log += "ndlc: clang++ native build failed (exit " + std::to_string(rc) + ")\n";
    }
  }
  if (!r.ok && !tools.hasClang && !tools.hasLlc) {
    log += "ndlc: no clang++ or llc found -- cannot build a native binary\n";
  }

  r.log = std::move(log);
  return r;
}

bool assemblePtx(const std::string& ptxPath, const std::string& outCubin,
                 const Toolchain& tools, std::string& log) {
  if (!tools.hasPtxas) {
    log += "ndlc: ptxas not found -- skipping cubin assembly "
           "(libndl_rt JITs PTX through the CUDA driver at runtime)\n";
    return false;
  }
  std::string cmd = shellQuote(tools.ptxasPath) + " -arch=sm_70 -o " +
                    shellQuote(outCubin) + " " + shellQuote(ptxPath);
  int rc = runCapture(cmd, log, /*verbose=*/true);
  if (rc == 0 && fileExists(outCubin)) return true;
  log += "ndlc: ptxas failed (exit " + std::to_string(rc) + ")\n";
  return false;
}

} // namespace ndl
