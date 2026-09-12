// NDL v1.0 — Recursive descent parser
// Contract header — frozen. Implementation: src/parser.cpp
#pragma once

#include "ndl/ast.hpp"
#include "ndl/diag.hpp"
#include "ndl/tokens.hpp"
#include <memory>
#include <vector>

namespace ndl {

class Parser {
public:
  Parser(std::string file, std::vector<Token> tokens, DiagnosticEngine& diag);

  // Parses a full module. Never returns nullptr (on hard errors returns a
  // module with whatever items were recovered; errors reported via engine).
  std::unique_ptr<Module> parse();

private:
  std::string file_;
  std::vector<Token> toks_;
  DiagnosticEngine& diag_;
  size_t pos_ = 0;

  // token access
  const Token& cur() const;
  const Token& peek(size_t n = 1) const;
  const Token& advance();          // returns token consumed
  bool check(Tok k) const;
  bool match(Tok k);               // if match: advance, true
  const Token& expect(Tok k, const std::string& context); // error E0100 on mismatch
  bool atStmtStart() const;        // sync points for panic-mode recovery
  void synchronize();              // skip to next ';' / stmt start / '}'
  void parseError(const std::string& msg, SourceRange range); // reports E0100

  // grammar
  bool parseTopItem(Module& m);    // returns false on EOF
  StmtPtr parseImport();
  StmtPtr parseUse();
  StmtPtr parseNodeGroup();
  StmtPtr parseConnect(ConnectKind kind);
  StmtPtr parseStdp();
  StmtPtr parseLet();
  StmtPtr parsePrint();
  StmtPtr parseSaveLoad(bool save);
  StmtPtr parseExportRaster();
  StmtPtr parseIf();
  StmtPtr parseFor();
  StmtPtr parseWhile();
  StmtPtr parseRun();
  StmtPtr parseAtEmit();
  StmtPtr parseOnSpike();
  // v2.0 — asynchronous continuous-time environment
  StmtPtr parseSignal();
  StmtPtr parseOscillator();
  StmtPtr parseExternalStream();
  StmtPtr parseRunContinuous();
  StmtPtr parseStopContinuous();
  StmtPtr parseWaitContinuous();
  StmtPtr parseSetAssign(); // v2.0: `set IDENT = expr;` (optional prefix)
  StmtPtr parseConnArrowStmt(StmtKind kind); // set_plasticity | prune_weights
  StmtPtr parseBindInputStream();
  StmtPtr parseBlockStmt();        // '{' stmt* '}' as an anonymous block? NOT in grammar:
                                   // blocks only attach to if/for/while/run/on_spike.
  StmtPtr parseSimpleOrExprStmt();

  std::vector<StmtPtr> parseBlockBody(); // after '{', until matching '}'

  // expressions
  ExprPtr parseExpr();
  ExprPtr parseBinary(int minPrec);
  ExprPtr parseUnary();
  ExprPtr parsePrimary();
  ExprPtr parseCallArgs(ExprPtr calleeExpr); // '(' args ')' applied to Ident expr -> Call

  std::vector<CallArg> parseCallArgList(); // inside already-consumed '('

  static int binPrec(Tok op); // 0 = not binary

  TypeRef parseTypeRef();
};

} // namespace ndl
