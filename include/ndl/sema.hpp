// NDL v1.0 — Semantic analyzer (two-pass, symbol tables, type checker)
// Contract header — frozen. Implementation: src/sema.cpp
#pragma once

#include "ndl/ast.hpp"
#include "ndl/diag.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ndl {

struct TypeInfo {
  enum class K { Error, Int, Float, Bool, Str, Duration, Group, Tensor, Void } k = K::Error;
  std::vector<uint64_t> tensorDims; // for Tensor

  TypeInfo() = default;
  TypeInfo(K kk) : k(kk) {}
  static TypeInfo Int() { return TypeInfo(K::Int); }
  static TypeInfo Float() { return TypeInfo(K::Float); }
  static TypeInfo Bool() { return TypeInfo(K::Bool); }
  static TypeInfo Str() { return TypeInfo(K::Str); }
  static TypeInfo Duration() { return TypeInfo(K::Duration); }
  static TypeInfo Group() { return TypeInfo(K::Group); }
  static TypeInfo Tensor() { return TypeInfo(K::Tensor); }
  static TypeInfo Void() { return TypeInfo(K::Void); }
  static TypeInfo Error() { return TypeInfo(K::Error); }

  bool isError() const { return k == K::Error; }
  bool isNumeric() const { return k == K::Int || k == K::Float; }
  std::string toString() const;
};

struct GroupInfo {
  std::string name;
  uint64_t size = 0;
  NeuronType type = NeuronType::Excitatory;
  std::string module;      // defining module path
  SourceRange declRange;
};

struct GlobalVarInfo {
  std::string name;
  TypeInfo type;
  const Stmt* decl = nullptr; // the Let stmt
};

// v2.0 — declarations of the asynchronous continuous-time environment.
struct SignalInfo {
  std::string name;
  SourceRange declRange;
};
struct OscillatorInfo {
  std::string name;
  SourceRange declRange;
};
struct StreamInfo {
  std::string name;
  uint64_t size = 0;
  SourceRange declRange;
};

struct SemaResult {
  std::vector<GroupInfo> groups;                          // declaration order
  std::unordered_map<std::string, size_t> groupIndex;     // name -> index in groups
  std::vector<GlobalVarInfo> globals;                     // direct top-level lets, order

  // v2.0 — signals / oscillators / streams in declaration order.
  std::vector<SignalInfo> signals;
  std::unordered_map<std::string, size_t> signalIndex;    // name -> index in signals
  std::vector<OscillatorInfo> oscillators;
  std::vector<StreamInfo> streams;
  std::unordered_map<std::string, size_t> streamIndex;    // name -> index in streams

  // v2.0 — backend flags (PTX kernel selection, GPU-path eligibility).
  bool hasContinuous = false;      // program contains run_continuous
  bool hasOscillators = false;     // program declares oscillators
  bool hasStreams = false;         // program declares external streams
  bool hasPrune = false;           // program contains prune_weights
  bool hasModulatedStdp = false;   // some configure_stdp uses modulator=

  // For native codegen: types of every variable binding site.
  // Keyed by the Stmt pointer: Let (variable type), Assign (variable type),
  // For (loop var = Int).
  std::unordered_map<const Stmt*, TypeInfo> varTypes;

  // Type of EVERY expression node after promotion (codegen relies on it).
  std::unordered_map<const Expr*, TypeInfo> exprTypes;

  const GroupInfo* findGroup(const std::string& name) const;
};

class Sema {
public:
  Sema(Program& program, DiagnosticEngine& diag);
  // Runs both passes. Returns partial results even when errors were reported.
  SemaResult run();

private:
  Program& prog_;
  DiagnosticEngine& diag_;
  SemaResult res_;
};

} // namespace ndl
