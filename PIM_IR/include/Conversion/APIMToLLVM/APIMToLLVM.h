//===- LinalgToLLVM.h - Utils to convert from the linalg dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_APIMToLLVM_APIMToLLVM_H_
#define MLIR_CONVERSION_APIMToLLVM_APIMToLLVM_H_
#include "Pass/Passes.h"
#include "Dialect/APIM/IR/APIMOps.hpp"
#include <memory>

namespace mlir {

class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;

/// Populate the given list with patterns that convert from APIM to LLVM.
void populateLoweringAPIMVectorOpToLLVMPatterns(RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringAPIMVectorImmOpToLLVMPatterns(RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringAPIMLoadOpToLLVMPatterns(RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringAPIMStoreOpToLLVMPatterns(RewritePatternSet &patterns, MLIRContext *context);
} // namespace mlir

#endif // MLIR_CONVERSION_APIMToLLVM_APIMToLLVM_H_
