// NDL v1.0 — ndlc compiler driver (CLI entry point)
//
// Commands: init | check | build | run | clean | help | --version
// Pipeline: manifest → import resolution → lex → parse → sema → { LLVM IR, PTX } → VM/native
#include "ndl/ast.hpp"
#include "ndl/diag.hpp"
#include "ndl/irgen_llvm.hpp"
#include "ndl/irgen_ptx.hpp"
#include "ndl/lexer.hpp"
#include "ndl/linker.hpp"
#include "ndl/manifest.hpp"
#include "ndl/parser.hpp"
#include "ndl/port.hpp"
#include "ndl/sema.hpp"
#include "ndl/toml.hpp"
#include "ndl/version.hpp"
#include "ndl/vm.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

using namespace ndl;

// ---------------------------------------------------------------------------
// Tiny filesystem helpers (POSIX)
// ---------------------------------------------------------------------------
namespace fsx {

bool fileExists(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}
bool dirExists(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
bool readFile(const std::string& path, std::string& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}
bool writeFile(const std::string& path, const std::string& data) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f << data;
  return f.good();
}
std::string dirName(const std::string& p) {
  auto slash = p.find_last_of("/\\");
  if (slash == std::string::npos) return ".";
  if (slash == 0) return "/";
  return p.substr(0, slash);
}
std::string baseName(const std::string& p) {
  auto slash = p.find_last_of("/\\");
  return slash == std::string::npos ? p : p.substr(slash + 1);
}
std::string joinPath(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (!b.empty() && (b[0] == '/' || b[0] == '\\')) return b;
  return a.back() == '/' ? a + b : a + "/" + b;
}
std::string absolute(const std::string& p) {
#if defined(_WIN32)
  // "C:/...", "C:\\...", UNC "//server/..." and rooted "/x" are absolute.
  if (p.size() >= 2 && (p[1] == ':' || p[0] == '/' || p[0] == '\\')) return p;
#else
  if (!p.empty() && p[0] == '/') return p;
#endif
  char buf[4096];
  if (::getcwd(buf, sizeof(buf))) return joinPath(buf, p);
  return p;
}
bool endsWith(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}
void mkdirP(const std::string& path) {
  std::string cur;
  for (size_t i = 0; i < path.size(); ++i) {
    cur += path[i];
    if (path[i] == '/' || path[i] == '\\' || i + 1 == path.size()) {
      if (!cur.empty() && cur != "/" && cur.back() == ':' ) continue;  // "C:" — always exists
      if (!cur.empty() && cur != "/" && !dirExists(cur)) ndlport::makeDir(cur);
    }
  }
}
#ifdef _WIN32
bool removeDirRecursive(const std::string& path) {
  // cmd.exe: rmdir /s /q — no external tools required.
  return ::system(("rmdir /s /q " + ndlport::shellQuote(path)).c_str()) == 0;
}
#else
bool removeDirRecursive(const std::string& path) { return ::system(("rm -rf " + ndlport::shellQuote(path)).c_str()) == 0; }
#endif

} // namespace fsx

// ---------------------------------------------------------------------------
// Locate libndl_rt.a for the native build. Priority: NDLC_RUNTIME_LIB env
// (validated), then exe-relative layouts, then cwd-relative guesses.
// ---------------------------------------------------------------------------
static std::string resolveRuntimeLib() {
  if (const char* env = ::getenv("NDLC_RUNTIME_LIB")) {
    if (env[0] != '\0' && fsx::fileExists(env)) return env;
  }
  std::vector<std::string> cands;
  std::string exe = ndlport::selfExePath();
  if (!exe.empty()) {
    const std::string exeDir = fsx::dirName(exe);
    cands.push_back(fsx::joinPath(exeDir + "/..", "lib/libndl_rt.a"));  // dist/bin/../lib
    cands.push_back(fsx::joinPath(exeDir, "libndl_rt.a"));              // flat layout
  }
  cands.push_back("dist/lib/libndl_rt.a");  // running from the project root
  cands.push_back("libndl_rt.a");           // last resort (clang reports it clearly)
  for (const std::string& c : cands) {
    if (!c.empty() && fsx::fileExists(c)) return c;
  }
  return "libndl_rt.a";
}

// ---------------------------------------------------------------------------
// Locate the runtime SOURCE directory. When found, the native build compiles
// the runtime with the same clang (ABI-guaranteed) instead of relying on the
// prebuilt archive. Priority: NDLC_RUNTIME_SRC env, then exe-relative, cwd.
// ---------------------------------------------------------------------------
static std::string resolveRuntimeSrcDir() {
  if (const char* env = ::getenv("NDLC_RUNTIME_SRC")) {
    if (env[0] != '\0' && fsx::dirExists(env)) return env;
  }
  std::vector<std::string> cands;
  std::string exe = ndlport::selfExePath();
  if (!exe.empty()) {
    const std::string exeDir = fsx::dirName(exe);
    // dist/bin/ndlc -> ../../runtime  (repo layout: ndl/dist/bin, ndl/runtime)
    cands.push_back(fsx::joinPath(exeDir + "/../..", "runtime"));
    // dist/ndlc or root-relative install
    cands.push_back(fsx::joinPath(exeDir + "/..", "runtime"));
    // ndlc sitting next to runtime/
    cands.push_back(fsx::joinPath(exeDir, "runtime"));
  }
  cands.push_back("runtime");  // running from the project root
  for (const std::string& c : cands) {
    if (fsx::dirExists(c)) return c;
  }
  return std::string();
}

// ---------------------------------------------------------------------------
// CLI options
// ---------------------------------------------------------------------------
struct Options {
  std::string command;
  std::string inputPath = ".";   // file or directory
  std::string projectOverride;   // --project
  bool emitLlvm = false, emitPtx = false, emitTokens = false, emitAst = false;
  int optLevel = -1;             // -1 = from manifest
  std::string triple;            // empty = default
  uint64_t seed = 0;
  bool seedSet = false;
  bool deviceSet = false;
  DeviceKind device = DeviceKind::CPU;
  bool forceNative = false, forceVm = false;
  bool noColor = false;
  bool verbose = false;
};

static bool gColor = false;

static std::string colorize(const std::string& code, const std::string& text) {
  return gColor ? "\033[" + code + "m" + text + "\033[0m" : text;
}
static void statusLine(const std::string& head, const std::string& headColor,
                       const std::string& rest) {
  std::cerr << colorize(headColor, head) << " " << rest << "\n";
}

// ---------------------------------------------------------------------------
// Project resolution
// ---------------------------------------------------------------------------
struct ProjectCtx {
  Manifest manifest;
  std::string entry; // absolute-ish entry .ndl path
};

static ProjectCtx resolveProject(const Options& opt, DiagnosticEngine& diag) {
  ProjectCtx ctx;
  std::string root;
  std::string input = opt.inputPath;

  if (!opt.projectOverride.empty()) root = opt.projectOverride;

  if (fsx::endsWith(input, ".ndl")) {
    if (!fsx::fileExists(input)) {
      diag.error("E0501", SourceRange(), "input file not found: " + input);
      return ctx;
    }
    if (root.empty()) root = findProjectRoot(input);
    if (root.empty()) root = fsx::dirName(fsx::absolute(input));
    ctx.entry = fsx::absolute(input);
  } else {
    std::string dir = input;
    if (!fsx::dirExists(dir)) {
      diag.error("E0501", SourceRange(), "input directory not found: " + input);
      return ctx;
    }
    if (root.empty()) {
      std::string probe = fsx::joinPath(dir, "ndl.toml");
      if (fsx::fileExists(probe)) root = fsx::absolute(dir);
      else root = fsx::absolute(dir); // defaults below
    }
    ctx.entry.clear();
  }

  std::string manifestPath = fsx::joinPath(root, "ndl.toml");
  if (fsx::fileExists(manifestPath)) {
    ctx.manifest = loadManifest(manifestPath, diag);
  } else {
    ctx.manifest.loaded = false;
    ctx.manifest.rootDir = root;
  }
  ctx.manifest.rootDir = root;

  if (ctx.entry.empty()) ctx.entry = fsx::joinPath(root, ctx.manifest.main);
  if (!fsx::fileExists(ctx.entry)) {
    diag.error("E0501", SourceRange(),
               "entry source not found: " + ctx.entry + " (set `main` in ndl.toml)");
  }
  return ctx;
}

// ---------------------------------------------------------------------------
// Compilation pipeline
// ---------------------------------------------------------------------------
struct Compiled {
  std::unique_ptr<Program> program;
  SemaResult sema;
  std::string llvmIR;
  std::string ptx;
  double msLexParse = 0, msSema = 0, msPTX = 0, msLLVM = 0;
};

static void dumpTokens(const std::vector<Token>& toks, const std::string& file) {
  std::cout << "tokens of " << file << " (" << toks.size() << "):\n";
  for (const auto& t : toks) {
    std::cout << "  " << tokName(t.kind) << " @" << t.range.begin.line << ":"
              << t.range.begin.col;
    if (!t.text.empty()) std::cout << " `" << t.text << "`";
    else if (t.kind == Tok::IntLit) std::cout << " " << t.intVal;
    else if (t.kind == Tok::FloatLit || t.kind == Tok::DurationLit)
      std::cout << " " << t.floatVal;
    std::cout << "\n";
  }
}

static void dumpExpr(const Expr* e, int indent) {
  std::string pad(indent, ' ');
  switch (e->kind) {
  case ExprKind::IntLit: std::cout << pad << "Int " << e->intVal << "\n"; break;
  case ExprKind::FloatLit: std::cout << pad << "Float " << e->floatVal << "\n"; break;
  case ExprKind::DurationLit: std::cout << pad << "Duration " << e->floatVal << "ms\n"; break;
  case ExprKind::StringLit: std::cout << pad << "Str \"" << e->strVal << "\"\n"; break;
  case ExprKind::BoolLit: std::cout << pad << "Bool " << (e->boolVal ? "true" : "false") << "\n"; break;
  case ExprKind::DeviceLit: std::cout << pad << "Device " << e->name << "\n"; break;
  case ExprKind::Ident: std::cout << pad << "Ident " << e->name << "\n"; break;
  case ExprKind::Unary:
    std::cout << pad << "Unary " << tokName(e->op) << "\n";
    dumpExpr(e->lhs.get(), indent + 2);
    break;
  case ExprKind::Binary:
    std::cout << pad << "Binary " << tokName(e->op) << "\n";
    dumpExpr(e->lhs.get(), indent + 2);
    dumpExpr(e->rhs.get(), indent + 2);
    break;
  case ExprKind::Call:
    std::cout << pad << "Call " << e->name << " (" << e->args.size() << " args)\n";
    for (auto& a : e->args) {
      std::cout << pad << "  arg" << (a.named ? " " + a.name + "=" : " =") << "\n";
      dumpExpr(a.value.get(), indent + 4);
    }
    break;
  }
}

static const char* stmtName(StmtKind k) {
  switch (k) {
  case StmtKind::Import: return "Import"; case StmtKind::Use: return "Use";
  case StmtKind::NodeGroup: return "NodeGroup"; case StmtKind::Connect: return "Connect";
  case StmtKind::Stdp: return "Stdp"; case StmtKind::Let: return "Let";
  case StmtKind::Assign: return "Assign"; case StmtKind::Print: return "Print";
  case StmtKind::SaveCheckpoint: return "SaveCheckpoint";
  case StmtKind::LoadCheckpoint: return "LoadCheckpoint";
  case StmtKind::ExportRaster: return "ExportRaster"; case StmtKind::If: return "If";
  case StmtKind::For: return "For"; case StmtKind::While: return "While";
  case StmtKind::Run: return "Run"; case StmtKind::AtEmit: return "AtEmit";
  case StmtKind::OnSpike: return "OnSpike"; case StmtKind::ExprStmt: return "ExprStmt";
  case StmtKind::SignalDecl: return "SignalDecl";
  case StmtKind::OscillatorDecl: return "OscillatorDecl";
  case StmtKind::ExternalStreamDecl: return "ExternalStreamDecl";
  case StmtKind::BindInputStream: return "BindInputStream";
  case StmtKind::RunContinuous: return "RunContinuous";
  case StmtKind::StopContinuous: return "StopContinuous";
  case StmtKind::WaitContinuous: return "WaitContinuous";
  case StmtKind::SetPlasticity: return "SetPlasticity";
  case StmtKind::PruneWeights: return "PruneWeights";
  }
  return "?";
}

static void dumpStmt(const Stmt* s, int indent) {
  std::string pad(indent, ' ');
  std::cout << pad << stmtName(s->kind);
  if (!s->name.empty()) std::cout << " name=" << s->name;
  if (!s->srcName.empty()) std::cout << " src=" << s->srcName;
  if (!s->dstName.empty()) std::cout << " dst=" << s->dstName;
  if (!s->path.empty()) std::cout << " path=" << s->path;
  std::cout << "\n";
  if (s->sizeExpr) dumpExpr(s->sizeExpr.get(), indent + 2);
  if (s->cond) dumpExpr(s->cond.get(), indent + 2);
  if (s->lo) dumpExpr(s->lo.get(), indent + 2);
  if (s->hi) dumpExpr(s->hi.get(), indent + 2);
  if (s->init) dumpExpr(s->init.get(), indent + 2);
  if (s->durationExpr) dumpExpr(s->durationExpr.get(), indent + 2);
  if (s->dtExpr) dumpExpr(s->dtExpr.get(), indent + 2);
  if (s->timeExpr) dumpExpr(s->timeExpr.get(), indent + 2);
  if (s->currentExpr) dumpExpr(s->currentExpr.get(), indent + 2);
  if (s->expr) dumpExpr(s->expr.get(), indent + 2);
  for (auto& a : s->params) dumpExpr(a.value.get(), indent + 2);
  for (auto& a : s->opts) dumpExpr(a.value.get(), indent + 2);
  for (auto& e : s->printArgs) dumpExpr(e.get(), indent + 2);
  for (auto& c : s->body) dumpStmt(c.get(), indent + 2);
  for (auto& c : s->elseBody) dumpStmt(c.get(), indent + 2);
}

static bool compileProject(const ProjectCtx& ctx, const Options& opt,
                    DiagnosticEngine& diag, Compiled& out) {
  // 1. Import resolution
  ImportResolver resolver(diag, ctx.manifest.rootDir, ctx.manifest);
  std::vector<ResolvedSource> sources = resolver.resolve(ctx.entry);
  if (diag.hasErrors()) return false;

  // 2. Lex + parse every module (deps first, main last)
  out.program = std::make_unique<Program>();
  out.program->mainModulePath = ctx.entry;
  auto lp0 = std::chrono::steady_clock::now();
  for (auto& src : sources) {
    Lexer lexer(src.path, src.source, diag);
    auto toks = lexer.tokenize();
    if (opt.emitTokens) dumpTokens(toks, src.path);
    Parser parser(src.path, std::move(toks), diag);
    auto mod = parser.parse();
    if (mod) {
      mod->path = src.path;
      mod->name = src.logicalName;
      out.program->modules.push_back(std::move(mod));
    }
  }
  if (diag.hasErrors()) return false;

  // 2b. Validate `use` packages against manifest dependencies
  for (auto& mod : out.program->modules) {
    for (auto& item : mod->items) {
      if (item->kind != StmtKind::Use) continue;
      bool found = false;
      for (auto& d : ctx.manifest.deps)
        if (d.name == item->name) { found = true; break; }
      if (!found) {
        diag.error("E0403", item->range,
                   "package `" + item->name + "` is not declared in [dependencies] of ndl.toml");
      }
    }
  }
  if (diag.hasErrors()) return false;

  auto lp1 = std::chrono::steady_clock::now();
  out.msLexParse = std::chrono::duration<double, std::milli>(lp1 - lp0).count();

  // 3. Semantic analysis
  auto s0 = std::chrono::steady_clock::now();
  Sema sema(*out.program, diag);
  out.sema = sema.run();
  auto s1 = std::chrono::steady_clock::now();
  out.msSema = std::chrono::duration<double, std::milli>(s1 - s0).count();
  if (diag.hasErrors()) return false;

  // 4. PTX generation (always: embedded into native binaries, used by VM for GPU runs)
  auto p0 = std::chrono::steady_clock::now();
  PTXCodeGen ptxGen(*out.program, out.sema, diag);
  out.ptx = ptxGen.generate();
  auto p1 = std::chrono::steady_clock::now();
  out.msPTX = std::chrono::duration<double, std::milli>(p1 - p0).count();

  // 5. LLVM IR generation (needed for build/--emit-llvm; not for pure VM run)
  if (opt.command != "run" || opt.emitLlvm || opt.forceNative) {
    auto l0 = std::chrono::steady_clock::now();
    LLVMEmitOptions lopts;
    lopts.triple = opt.triple.empty() ? lopts.triple : opt.triple;
    lopts.optLevel = opt.optLevel >= 0 ? opt.optLevel : ctx.manifest.optLevel;
    lopts.fastmath = ctx.manifest.fastmath;
    lopts.ptxSource = out.ptx;
    lopts.moduleName = fsx::baseName(ctx.entry);
    LLVMCodeGen llvmGen(*out.program, out.sema, lopts, diag);
    out.llvmIR = llvmGen.generate();
    auto l1 = std::chrono::steady_clock::now();
    out.msLLVM = std::chrono::duration<double, std::milli>(l1 - l0).count();
    if (diag.hasErrors()) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------
static int cmdCheck(const Options& opt) {
  DiagnosticEngine diag(!opt.noColor);
  ProjectCtx ctx = resolveProject(opt, diag);
  if (diag.hasErrors()) { diag.renderAll(std::cerr); return 1; }

  statusLine("   Checking", "32", ctx.manifest.projectName + " v" + ctx.manifest.version +
                                     " (" + fsx::baseName(ctx.entry) + ")");
  Compiled compiled;
  bool ok = compileProject(ctx, opt, diag, compiled);
  diag.renderAll(std::cerr);
  if (!ok) return 1;

  size_t groups = compiled.sema.groups.size();
  size_t globals = compiled.sema.globals.size();
  std::ostringstream ss;
  ss << "in " << (compiled.msLexParse + compiled.msSema) / 1000.0 << "s -- "
     << groups << " node groups, " << globals << " top-level variables, "
     << compiled.program->modules.size() << " modules: OK";
  statusLine("    Finished", "32", ss.str());
  if (diag.warningCount() > 0)
    statusLine("    Warnings", "33", std::to_string(diag.warningCount()));
  return 0;
}

static int cmdBuild(const Options& opt) {
  DiagnosticEngine diag(!opt.noColor);
  ProjectCtx ctx = resolveProject(opt, diag);
  if (diag.hasErrors()) { diag.renderAll(std::cerr); return 1; }

  statusLine("   Compiling", "36", ctx.manifest.projectName + " v" + ctx.manifest.version +
                                      " (" + fsx::baseName(ctx.entry) + ")");
  Compiled compiled;
  if (!compileProject(ctx, opt, diag, compiled)) {
    diag.renderAll(std::cerr);
    return 1;
  }
  diag.renderAll(std::cerr);

  std::string buildDir = fsx::joinPath(ctx.manifest.rootDir, "build");
  fsx::mkdirP(buildDir);
  std::string name = ctx.manifest.projectName;

  std::string llPath = fsx::joinPath(buildDir, name + ".ll");
  std::string ptxPath = fsx::joinPath(buildDir, name + ".ptx");
  fsx::writeFile(llPath, compiled.llvmIR);
  fsx::writeFile(ptxPath, compiled.ptx);

  std::string exePath;
  Toolchain tools = detectToolchain();
  bool nativeDone = false;
  if (tools.hasClang || tools.hasLlc) {
    const std::string runtimeLib = resolveRuntimeLib();
    const std::string runtimeSrcDir = resolveRuntimeSrcDir();
    exePath = fsx::joinPath(buildDir, name);
    NativeBuildResult r = buildNative(llPath, exePath, runtimeLib, runtimeSrcDir,
                                      opt.optLevel >= 0 ? opt.optLevel : ctx.manifest.optLevel,
                                      tools, opt.verbose);
    if (r.ok) {
      nativeDone = true;
      statusLine("    Native", "32", r.artifact);
    } else {
      statusLine("    Native", "31", "failed: " + r.log);
    }
  } else {
    statusLine("    Native", "33",
               "skipped -- no clang/llc found (searched PATH and standard LLVM "
               "install dirs; set NDLC_LLVM_PREFIX=<dir> to override; "
               "LLVM IR + PTX artifacts written)");
  }

  std::ostringstream ss;
  ss << "LLVM IR: " << llPath << " (" << compiled.llvmIR.size() / 1024 << " KiB) | "
     << "PTX: " << ptxPath << " (" << compiled.ptx.size() / 1024 << " KiB)";
  statusLine("    Finished", "32", ss.str());
  (void)nativeDone;
  return 0;
}

static int cmdRun(const Options& opt) {
  DiagnosticEngine diag(!opt.noColor);
  ProjectCtx ctx = resolveProject(opt, diag);
  if (diag.hasErrors()) { diag.renderAll(std::cerr); return 1; }

  statusLine("    Running", "36", ctx.manifest.projectName + " v" + ctx.manifest.version +
                                     " (" + fsx::baseName(ctx.entry) + ")");
  Compiled compiled;
  if (!compileProject(ctx, opt, diag, compiled)) {
    diag.renderAll(std::cerr);
    return 1;
  }
  diag.renderAll(std::cerr);

  if (opt.emitLlvm) {
    std::string buildDir = fsx::joinPath(ctx.manifest.rootDir, "build");
    fsx::mkdirP(buildDir);
    std::string llPath = fsx::joinPath(buildDir, ctx.manifest.projectName + ".ll");
    std::string ptxPath = fsx::joinPath(buildDir, ctx.manifest.projectName + ".ptx");
    fsx::writeFile(llPath, compiled.llvmIR);
    fsx::writeFile(ptxPath, compiled.ptx);
    statusLine("    Emitted", "36", llPath + " , " + ptxPath);
  }

  // Native mode requested?
  Toolchain tools = detectToolchain();
  if (opt.forceNative && !tools.hasClang && !tools.hasLlc) {
    statusLine("    Native", "33", "no clang/llc available -- falling back to VM");
  }
  if (opt.forceNative && (tools.hasClang || tools.hasLlc)) {
    std::string buildDir = fsx::joinPath(ctx.manifest.rootDir, "build");
    fsx::mkdirP(buildDir);
    std::string llPath = fsx::joinPath(buildDir, ctx.manifest.projectName + ".ll");
    fsx::writeFile(llPath, compiled.llvmIR);
    std::string exePath = fsx::joinPath(buildDir, ctx.manifest.projectName);
    const std::string runtimeLib = resolveRuntimeLib();
    const std::string runtimeSrcDir = resolveRuntimeSrcDir();
    NativeBuildResult r = buildNative(llPath, exePath, runtimeLib, runtimeSrcDir,
                                      opt.optLevel >= 0 ? opt.optLevel : ctx.manifest.optLevel,
                                      tools, opt.verbose);
    if (!r.ok) {
      statusLine("    Native", "31", "failed: " + r.log);
      return 1;
    }
    statusLine("    Native", "32", "built " + r.artifact + " -- launching");
    // shellQuote handles spaces; wrapForCmd protects the quotes from cmd.exe's
    // legacy outer-quote stripping (no-op on POSIX).
    return std::system(ndlport::wrapForCmd(ndlport::shellQuote(r.artifact)).c_str());
  }

  // VM mode (default)
  VMOptions vopts;
  vopts.seed = opt.seedSet ? opt.seed : ctx.manifest.seed;
  vopts.ptxSource = compiled.ptx;
  vopts.verbose = opt.verbose;
  VM vm(*compiled.program, diag, vopts);
  int rc = vm.run();
  statusLine("    Finished", "32", "vm execution");
  return rc;
}

static int cmdInit(const Options& opt) {
  std::string dir = opt.inputPath == "." ? "." : opt.inputPath;
  fsx::mkdirP(dir);
  fsx::mkdirP(fsx::joinPath(dir, "stdlib"));
  fsx::mkdirP(fsx::joinPath(dir, "models"));

  std::string name = fsx::baseName(fsx::absolute(dir));
  if (name.empty() || name == ".") name = "ndl-app";

  const char* toml = R"([project]
name = "%NAME%"
version = "0.1.0"
ndl_version = "1.0"
main = "main.ndl"

[build]
target = "auto"
opt_level = 2
fastmath = true

[simulation]
seed = 42
)";
  std::string tomlSrc = toml;
  auto pos = tomlSrc.find("%NAME%");
  if (pos != std::string::npos) tomlSrc.replace(pos, 6, name);

  const char* mainSrc = R"(// %NAME% -- NDL v1.0 starter project
import "stdlib/math.ndl";

node_group Stimulus[100] : Excitatory(tau=20.0, threshold=-55.0, rest=-70.0, reset=-75.0);
node_group Output[50]    : Excitatory(tau=20.0, threshold=-55.0, rest=-70.0, reset=-75.0);

dense_connect(Stimulus, Output, weight_func=random_gaussian(0.4, 0.1), plastic=true);
configure_stdp(Stimulus -> Output, lr_pot=0.01, lr_dep=0.005, window_ms=20.0);

print("Hello from NDL! Simulating 500ms of spiking dynamics.");

run (duration = 500ms, dt = 1.0ms, device = CPU) {
    at 10ms  emit Stimulus[0..10](current = 30.0);
    at 100ms emit Stimulus[10..30](current = 35.0);
    at 300ms emit Stimulus[0..5](current = 28.0);
}

print("Simulation complete.");
export_raster(Output, "spikes.csv");
)";
  std::string mainStr = mainSrc;
  pos = mainStr.find("%NAME%");
  if (pos != std::string::npos) mainStr.replace(pos, 6, name);

  bool ok = true;
  ok &= fsx::writeFile(fsx::joinPath(dir, "ndl.toml"), tomlSrc);
  ok &= fsx::writeFile(fsx::joinPath(dir, "main.ndl"), mainStr);
  ok &= fsx::writeFile(fsx::joinPath(dir, "stdlib/math.ndl"),
                       "// NDL v1.0 stdlib math\nlet E = 2.718281828459045;\nlet PI = 3.141592653589793;\n");
  ok &= fsx::writeFile(fsx::joinPath(dir, ".gitignore"), "build/\nmodels/\n*.ndlbin\n*.csv\n");
  if (!ok) {
    std::cerr << "error: failed to write project files\n";
    return 1;
  }
  statusLine("     Created", "32", "NDL project `" + name + "` in " + fsx::absolute(dir));
  statusLine("        Next", "36", "ndlc run " + fsx::joinPath(dir, "main.ndl"));
  return 0;
}

static int cmdClean(const Options& opt) {
  DiagnosticEngine diag(!opt.noColor);
  ProjectCtx ctx = resolveProject(opt, diag);
  if (diag.hasErrors()) { diag.renderAll(std::cerr); return 1; }
  std::string buildDir = fsx::joinPath(ctx.manifest.rootDir, "build");
  if (fsx::dirExists(buildDir)) {
    fsx::removeDirRecursive(buildDir);
    statusLine("     Cleaned", "32", buildDir);
  } else {
    statusLine("     Cleaned", "32", "nothing to clean");
  }
  return 0;
}

static void printUsage() {
  std::cout <<
      "ndlc " << kNdlcVersion << " -- NDL (Neural Description Language) compiler & toolchain\n"
      "\n"
      "USAGE:\n"
      "    ndlc <COMMAND> [PATH] [OPTIONS]\n"
      "\n"
      "COMMANDS:\n"
      "    init [dir]      Scaffold a new NDL project\n"
      "    check [path]    Parse & type-check without executing\n"
      "    build [path]    Compile: emit LLVM IR + PTX; native binary when toolchain present\n"
      "    run [path]      Execute (VM by default; --native for compiled binary)\n"
      "    clean [path]    Remove build artifacts\n"
      "    help            Show this help\n"
      "\n"
      "OPTIONS:\n"
      "    --emit-llvm      Write .ll artifact during run/check\n"
      "    --emit-ptx       Write .ptx artifact during run/check\n"
      "    --emit-tokens    Dump lexer tokens (debug)\n"
      "    --emit-ast       Dump AST (debug)\n"
      "    --opt <0-3>      Override optimization level from ndl.toml\n"
      "    --triple <t>     Override LLVM target triple\n"
      "    --seed <n>       Override RNG seed\n"
      "    --device <cpu|gpu>  Override simulation device\n"
      "    --native         Force native build & run (requires clang or llc)\n"
      "    --vm             Force VM execution (default)\n"
      "    --project <dir>  Explicit project root\n"
      "    --no-color       Disable colored diagnostics\n"
      "    --verbose        Verbose output\n"
      "\n"
      "VERSION:\n"
      "    ndlc " << kNdlcVersion << " (NDL language v" << kNdlVersion << ")\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  Options opt;
  std::vector<std::string> args(argv + 1, argv + argc);

  if (args.empty()) {
    printUsage();
    return 64;
  }

  size_t idx = 0;
  auto isCommand = [](const std::string& s) {
    return s == "init" || s == "check" || s == "build" || s == "run" || s == "clean" ||
           s == "help" || s == "--help" || s == "-h" || s == "--version" || s == "-V";
  };

  if (isCommand(args[0])) {
    opt.command = args[0];
    if (opt.command == "--help" || opt.command == "-h") opt.command = "help";
    if (opt.command == "--version" || opt.command == "-V") opt.command = "version";
    idx = 1;
  } else {
    opt.command = "run"; // default command
  }

  std::vector<std::string> positional;
  for (; idx < args.size(); ++idx) {
    const std::string& a = args[idx];
    auto next = [&](std::string& dst) -> bool {
      if (idx + 1 >= args.size()) return false;
      dst = args[++idx];
      return true;
    };
    if (a == "--emit-llvm") opt.emitLlvm = true;
    else if (a == "--emit-ptx") opt.emitPtx = true;
    else if (a == "--emit-tokens") opt.emitTokens = true;
    else if (a == "--emit-ast") opt.emitAst = true;
    else if (a == "--opt") { std::string v; if (!next(v)) { std::cerr << "missing value for --opt\n"; return 64; } opt.optLevel = std::atoi(v.c_str()); }
    else if (a.rfind("--opt=", 0) == 0) opt.optLevel = std::atoi(a.c_str() + 6);
    else if (a == "--triple") { if (!next(opt.triple)) return 64; }
    else if (a.rfind("--triple=", 0) == 0) opt.triple = a.c_str() + 9;
    else if (a == "--seed") { std::string v; if (!next(v)) return 64; opt.seed = std::strtoull(v.c_str(), nullptr, 10); opt.seedSet = true; }
    else if (a.rfind("--seed=", 0) == 0) { opt.seed = std::strtoull(a.c_str() + 7, nullptr, 10); opt.seedSet = true; }
    else if (a == "--device") { std::string v; if (!next(v)) return 64; opt.device = (v == "gpu" || v == "GPU") ? DeviceKind::GPU : DeviceKind::CPU; opt.deviceSet = true; }
    else if (a == "--native") opt.forceNative = true;
    else if (a == "--vm") opt.forceVm = true;
    else if (a == "--project") { if (!next(opt.projectOverride)) return 64; }
    else if (a.rfind("--project=", 0) == 0) opt.projectOverride = a.c_str() + 10;
    else if (a == "--no-color") opt.noColor = true;
    else if (a == "--verbose") opt.verbose = true;
    else if (!a.empty() && a[0] == '-') {
      std::cerr << "unknown option: " << a << "\n";
      return 64;
    } else positional.push_back(a);
  }

  if (opt.command == "version") {
    std::cout << "ndlc " << kNdlcVersion << " (NDL v" << kNdlVersion
              << ", llvm-ir + cuda-ptx backends)\n";
    Toolchain t = detectToolchain();
    std::cout << "toolchain: " << t.summary() << "\n";
    return 0;
  }
  if (opt.command == "help") {
    printUsage();
    return 0;
  }

  if (!positional.empty()) opt.inputPath = positional[0];
  if (positional.size() > 1) {
    std::cerr << "error: unexpected extra arguments\n";
    return 64;
  }

  gColor = !opt.noColor && DiagnosticEngine::supportsColor();

  switch (opt.command[0]) {
  case 'i': return cmdInit(opt);
  case 'c': return opt.command == "clean" ? cmdClean(opt) : cmdCheck(opt);
  case 'b': return cmdBuild(opt);
  case 'r': return cmdRun(opt);
  default:
    printUsage();
    return 64;
  }
}
