//===- LinalgToLLVM.h - Utils to convert from the linalg dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_DPIMToLLVM_DPIMToLLVM_H_
#define MLIR_CONVERSION_DPIMToLLVM_DPIMToLLVM_H_
#include "Pass/Passes.h"
#include "Dialect/DPIM/IR/DPIMOps.hpp"
#include <memory>

namespace mlir {

class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;

/// Populate the given list with patterns that convert from DPIM to LLVM.
void populateLoweringDPIMVectorCompOpToLLVMPatterns(RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringDPIMVectorImmOpToLLVMPatterns(RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringDPIMSetOpToLLVMPatterns(RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringDPIMStoreOpToLLVMPatterns(RewritePatternSet &patterns, MLIRContext *context);
} // namespace mlir

#endif // MLIR_CONVERSION_DPIMToLLVM_DPIMToLLVM_H_
