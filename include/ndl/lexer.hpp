// NDL v1.0 — Hand-written lexer
// Contract header — frozen. Implementation: src/lexer.cpp
#pragma once

#include "ndl/diag.hpp"
#include "ndl/tokens.hpp"
#include <string>
#include <vector>

namespace ndl {

class Lexer {
public:
  Lexer(std::string fileName, std::string source, DiagnosticEngine& diag);

  // Tokenizes the whole source. Always ends with a single Eof token.
  // Lexical errors are reported to the engine (E0001) and skipped.
  std::vector<Token> tokenize();

private:
  std::string file_;
  std::string src_;
  DiagnosticEngine& diag_;
  size_t pos_ = 0;
  uint32_t line_ = 1, col_ = 1;

  bool atEnd() const;
  char cur() const;
  char peek(size_t n = 1) const;
  char advance();
  void skipWhitespaceAndComments();
  Token makeToken(Tok kind, SourceRange range, std::string text = "");
  SourceLoc locHere() const;

  Token lexNumber();
  Token lexString();
  Token lexIdentOrKeyword();
  void lexError(const std::string& msg, SourceRange range);
};

} // namespace ndl
