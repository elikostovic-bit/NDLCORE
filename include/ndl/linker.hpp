// NDL v1.0 — External toolchain driver (LLVM tools, linker, CUDA tools)
// Contract header — frozen. Implementation: src/linker.cpp
//
// The ndlc compiler itself has NO LLVM dependency: it emits LLVM IR text and
// drives external tools when available:
//   native build:  clang++ out.ll -O2 libndl_rt.a -o app   (preferred)
//              or  llc out.ll -o out.o  →  cc out.o libndl_rt.a -o app
//   gpu tools:     nvcc/ptxas (optional; static PTX assembly)
#pragma once

#include <string>
#include <vector>

namespace ndl {

struct Toolchain {
  bool hasClang = false;
  bool hasLlc = false;
  bool hasOpt = false;
  bool hasGpp = false;   // g++ (linker fallback only, cannot compile .ll)
  bool hasNvcc = false;
  bool hasPtxas = false;

  std::string clangPath, llcPath, optPath, gppPath, nvccPath, ptxasPath;
  std::string llvmVersion; // from clang/llc, best effort

  // "clang 17.0.1; llc —; g++ 12; ptxas —" style one-liner
  std::string summary() const;
};

Toolchain detectToolchain();

struct NativeBuildResult {
  bool ok = false;
  std::string log;      // captured stdout+stderr of the tool invocations
  std::string artifact; // path to executable when ok
};

// Builds a native executable from an .ll file.
// Route selection (first that succeeds):
//   1. clang++ + runtime compiled FROM SOURCE with the same clang
//      (always ABI-consistent; needs runtimeSrcDir, which may be empty)
//   2. llc → object, then g++ links with the prebuilt libndl_rt.a
//   3. clang++ + prebuilt libndl_rt.a directly
// Extra link flags are added automatically (-lpthread -ldl -lm on POSIX,
// -lwinpthread for MinGW-g++ links; MSVC-target clang needs none).
NativeBuildResult buildNative(const std::string& llPath, const std::string& outExe,
                              const std::string& runtimeLib,
                              const std::string& runtimeSrcDir, int optLevel,
                              const Toolchain& tools, bool verbose);

// Optional static PTX assembly (only when ptxas exists). Never required —
// libndl_rt JITs PTX through the driver API at runtime.
bool assemblePtx(const std::string& ptxPath, const std::string& outCubin,
                 const Toolchain& tools, std::string& log);

} // namespace ndl
