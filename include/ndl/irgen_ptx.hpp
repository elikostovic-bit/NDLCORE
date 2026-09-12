// NDL v1.0 — CUDA PTX code generator — GPU backend
// Contract header — frozen. Implementation: src/irgen_ptx.cpp
//
// Emits a standalone PTX module (PTX 7.0, sm_70, f64) containing the fixed
// kernel set defined in INTERNALS.md §8. The kernels are loaded at runtime by
// libndl_rt via the CUDA driver API (cuModuleLoadData → JIT) when a run with
// device=GPU executes.
#pragma once

#include "ndl/ast.hpp"
#include "ndl/diag.hpp"
#include "ndl/sema.hpp"
#include <string>

namespace ndl {

class PTXCodeGen {
public:
  PTXCodeGen(const Program& program, const SemaResult& sema, DiagnosticEngine& diag);

  // Returns the full .ptx text.
  std::string generate();

private:
  const Program& prog_;
  const SemaResult& sema_;
  DiagnosticEngine& diag_;
};

} // namespace ndl
