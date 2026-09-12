// NDL v1.0 — Tree-walking VM (execution engine for `ndlc run`)
// Contract header — frozen. Implementation: src/vm.cpp
//
// Interprets the checked AST. All heavy operations (simulation, connections,
// STDP, IO, RNG, math) are delegated to libndl_rt — the same C ABI used by
// natively compiled binaries, guaranteeing identical semantics (§9).
#pragma once

#include "ndl/ast.hpp"
#include "ndl/diag.hpp"
#include <cstdint>
#include <string>

namespace ndl {

struct VMOptions {
  uint64_t seed = 42;            // passed to ndl_rt_set_seed before execution
  std::string ptxSource;         // passed to ndl_rt_gpu_load_ptx (may be empty)
  bool verbose = false;
};

class VM {
public:
  VM(const Program& program, DiagnosticEngine& diag, VMOptions opts);

  // Executes the merged program (modules in order). Returns process exit code:
  // 0 on success; does not return on runtime panic (libndl_rt exits(70)).
  int run();

private:
  const Program& prog_;
  DiagnosticEngine& diag_;
  VMOptions opts_;
};

} // namespace ndl
