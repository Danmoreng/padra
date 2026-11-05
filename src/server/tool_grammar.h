#pragma once
#include "runtime/token_constraint.h"
#include "text/tokenizer.h"
#include "util/json.h"
#include <memory>
namespace mellum16::server {
// A fresh lazy constraint for one assistant response. The tokenizer must outlive
// it. strict JSON Schema remains unsupported; see docs/tool-grammar.md.
Result<std::shared_ptr<TokenConstraint>> MakeToolGrammar(const json::Value &request,
                                                         const text::Tokenizer &tokenizer);
} // namespace mellum16::server
