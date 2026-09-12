// NDL v1.0 — LLVM IR (text) code generator — native CPU backend
// Contract header — frozen. Implementation: src/irgen_llvm.cpp
//
// Emits a complete, standalone LLVM IR module (.ll) per INTERNALS.md §7.
// Optimization & object emission are done by external LLVM tools (clang/llc/opt)
// driven by src/linker.cpp — the compiler itself has no LLVM dependency.
#pragma once

#include "ndl/ast.hpp"
#include "ndl/diag.hpp"
#include "ndl/sema.hpp"
#include <string>

namespace ndl {

struct LLVMEmitOptions {
  std::string triple = "x86_64-pc-linux-gnu"; // override with --triple
  int optLevel = 2;                            // 0..3 (passed to external tools; IR is level-agnostic)
  bool fastmath = true;                        // emit `fast` flags on float ops
  std::string ptxSource;                       // when non-empty: embedded as @ndl_ptx_source
  std::string moduleName = "main.ndl";         // ModuleID / source_filename
};

class LLVMCodeGen {
public:
  LLVMCodeGen(const Program& program, const SemaResult& sema,
              const LLVMEmitOptions& opts, DiagnosticEngine& diag);

  // Returns the full .ll text. On internal errors reports E03xx and returns
  // whatever was generated (driver aborts when engine hasErrors()).
  std::string generate();

private:
  const Program& prog_;
  const SemaResult& sema_;
  const LLVMEmitOptions& opts_;
  DiagnosticEngine& diag_;
};

} // namespace ndl
