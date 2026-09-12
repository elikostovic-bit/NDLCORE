// NDL v1.0 — Abstract Syntax Tree (tagged structs; field usage documented per kind)
// Contract header — frozen.
//
// FIELD USAGE TABLE (Stmt):
//  Import       : path (string literal), pathRange
//  Use          : name (package name)
//  NodeGroup    : name, sizeExpr, ntype, params (CallArg list: tau/threshold/rest/reset)
//  Connect      : ckind, srcName, dstName, opts (CallArg list)
//  Stdp         : srcName, dstName, opts (lr_pot/lr_dep/window_ms)
//  Let          : name, typeAnno, init
//  Assign       : name, init (value)
//  Print        : printArgs
//  SaveCheckpoint / LoadCheckpoint : path (expr, usually StringLit)
//  ExportRaster : srcName (group), path (expr)
//  If           : cond, body, elseBody
//  For          : loopVar, lo, hi, body
//  While        : cond, body
//  Run          : hasDuration, durationExpr, hasDt, dtExpr, hasDevice, device, body (only AtEmit)
//  RunContinuous: hasDt, dtExpr, hasDevice, device, body (any statements)      [v2.0]
//  StopContinuous: (no fields) — stops the continuous event loop               [v2.0]
//  SignalDecl   : name, init (numeric expr)                                   [v2.0]
//  OscillatorDecl: name, opts (frequency/amplitude/target/phase)              [v2.0]
//  ExternalStreamDecl: name, sizeExpr                                         [v2.0]
//  BindInputStream: srcName (stream), dstName (group), opts (encoding/max_freq) [v2.0]
//  SetPlasticity: srcName, dstName, opts (enabled)                            [v2.0]
//  PruneWeights : srcName, dstName, opts (threshold)                          [v2.0]
//  AtEmit       : timeExpr, groupName, sliceHasIndex, sliceHasRange, sliceLo, sliceHi,
//                 currentExpr, sliceRange
//  OnSpike      : handlerGroup, body
//  ExprStmt     : expr
//
// FIELD USAGE TABLE (Expr):
//  IntLit       : intVal
//  FloatLit     : floatVal
//  DurationLit  : floatVal (ms)
//  StringLit    : strVal
//  BoolLit      : boolVal
//  DeviceLit    : name ("CPU"/"GPU")
//  Ident        : name
//  Unary        : op (Tok::Not|Minus), lhs = operand
//  Binary       : op, lhs, rhs
//  Call         : name, args
#pragma once

#include "ndl/diag.hpp"
#include "ndl/tokens.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ndl {

enum class NeuronType { Excitatory, Inhibitory };
enum class DeviceKind { CPU, GPU };
enum class ConnectKind { Dense, Sparse, OneToOne };

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------
enum class ExprKind {
  IntLit, FloatLit, DurationLit, StringLit, BoolLit, DeviceLit,
  Ident, Unary, Binary, Call
};

struct Expr;
using ExprPtr = std::unique_ptr<Expr>;

struct CallArg {
  bool named = false;
  std::string name;      // when named
  ExprPtr value;
  SourceRange range;     // full arg range
};

struct Expr {
  ExprKind kind;
  SourceRange range;

  // Literals
  uint64_t intVal = 0;
  double floatVal = 0.0;
  bool boolVal = false;
  std::string strVal;

  // Ident / DeviceLit / Call: callee or symbol name
  std::string name;

  // Unary / Binary
  Tok op = Tok::Eof;
  ExprPtr lhs; // Binary lhs; Unary operand
  ExprPtr rhs; // Binary rhs

  // Call arguments (positional or named)
  std::vector<CallArg> args;
};

// ---------------------------------------------------------------------------
// Types (annotations)
// ---------------------------------------------------------------------------
struct TypeRef {
  enum class K { None, Int, Float, Bool, String, Tensor } k = K::None;
  std::vector<uint64_t> tensorDims; // for Tensor
  SourceRange range;
};

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------
enum class StmtKind {
  Import, Use, NodeGroup, Connect, Stdp,
  Let, Assign, Print, SaveCheckpoint, LoadCheckpoint, ExportRaster,
  If, For, While, Run, AtEmit, OnSpike, ExprStmt,
  // v2.0 — asynchronous continuous-time environment
  SignalDecl, OscillatorDecl, ExternalStreamDecl, BindInputStream,
  RunContinuous, StopContinuous, WaitContinuous, SetPlasticity, PruneWeights
};

struct Stmt;
using StmtPtr = std::unique_ptr<Stmt>;

struct Stmt {
  StmtKind kind;
  SourceRange range;

  // Import / Use
  std::string path;        // Import: path string as written
  SourceRange pathRange;   // Import: range of the string literal
  std::string name;        // Use: package name; NodeGroup/Let/Assign: name;
                           // ExportRaster/AtEmit: group name; OnSpike: group name;
                           // Connect/Stdp: srcName/dstName below.

  // NodeGroup
  ExprPtr sizeExpr;
  NeuronType ntype = NeuronType::Excitatory;
  std::vector<CallArg> params; // tau, threshold, rest, reset

  // Connect / Stdp
  ConnectKind ckind = ConnectKind::Dense;
  std::string srcName, dstName;
  std::vector<CallArg> opts;

  // Let / Assign
  TypeRef typeAnno;
  ExprPtr init;

  // Print
  std::vector<ExprPtr> printArgs;

  // If / While
  ExprPtr cond;
  std::vector<StmtPtr> body, elseBody;

  // For
  std::string loopVar;
  ExprPtr lo, hi;

  // Run
  bool hasDuration = false;
  ExprPtr durationExpr;
  bool hasDt = false;
  ExprPtr dtExpr;
  bool hasDevice = false;
  DeviceKind device = DeviceKind::CPU;

  // AtEmit slice
  bool sliceHasIndex = false; // Group[i]
  bool sliceHasRange = false; // Group[a..b]
  ExprPtr sliceLo, sliceHi;   // sliceLo used for single index too
  ExprPtr timeExpr, currentExpr;
  SourceRange sliceRange;

  // ExternalStreamDecl
  // sizeExpr reused from NodeGroup field above.

  // ExprStmt
  ExprPtr expr;
};

// ---------------------------------------------------------------------------
// Module / Program
// ---------------------------------------------------------------------------
struct Module {
  std::string path;      // file path as resolved
  std::string name;      // logical module name (path relative to root, '/'-separated)
  std::vector<StmtPtr> items;
};

struct Program {
  std::vector<std::unique_ptr<Module>> modules; // import order; main module LAST
  std::string mainModulePath;
};

} // namespace ndl
