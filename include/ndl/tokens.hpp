// NDL v1.0 — Token definitions
// Contract header — frozen.
#pragma once

#include "ndl/diag.hpp"
#include <cstdint>
#include <string>

namespace ndl {

enum class Tok {
  Eof,
  Ident,
  IntLit,      // intVal
  FloatLit,    // floatVal
  DurationLit, // floatVal (ms)
  FreqLit,     // floatVal (Hz)   -- v2.0
  StringLit,   // unescaped content stored in `text`

  // Keywords
  Kw_node_group, Kw_dense_connect, Kw_sparse_connect, Kw_one_to_one_connect,
  Kw_configure_stdp, Kw_run, Kw_at, Kw_emit, Kw_on_spike,
  Kw_if, Kw_else, Kw_for, Kw_in, Kw_while,
  Kw_let, Kw_set, Kw_print, Kw_save_checkpoint, Kw_load_checkpoint,
  Kw_export_raster, Kw_import, Kw_use,
  Kw_true, Kw_false,
  Kw_int, Kw_float, Kw_bool, Kw_string, Kw_tensor,
  Kw_Excitatory, Kw_Inhibitory, Kw_CPU, Kw_GPU, Kw_ms,
  // v2.0 keywords (async continuous-time environment)
  Kw_signal, Kw_oscillator, Kw_external, Kw_stream, Kw_run_continuous,
  Kw_stop_continuous, Kw_wait_continuous, Kw_set_plasticity, Kw_prune_weights, Kw_bind_input_stream,

  // Operators & punctuation
  Plus, Minus, Star, Slash, Percent,
  EqEq, NotEq, Lt, Gt, Le, Ge, AndAnd, OrOr, Not,
  Assign, Arrow, DotDot,
  Comma, Semi, Colon,
  LParen, RParen, LBracket, RBracket, LBrace, RBrace,
};

struct Token {
  Tok kind = Tok::Eof;
  std::string text;    // identifier name / string content (unescaped) / raw spelling
  SourceRange range;
  uint64_t intVal = 0;
  double floatVal = 0.0;
};

const char* tokName(Tok t);     // debug name, e.g. "Kw_run"
const char* tokSpelling(Tok t); // printable spelling, e.g. "'run'", "'->'", "identifier"
bool tokIsKeyword(Tok t);

} // namespace ndl
