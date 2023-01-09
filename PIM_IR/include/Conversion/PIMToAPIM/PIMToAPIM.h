//===- LinalgToLLVM.h - Utils to convert from the linalg dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_PIMToAPIM_PIMToAPIM_H_
#define MLIR_CONVERSION_PIMToAPIM_PIMToAPIM_H_
#include "Pass/Passes.h"
#include "Dialect/PIM/IR/PIMOps.hpp"
#include "Dialect/APIM/IR/APIMOps.hpp"
#include <memory>

namespace mlir {

class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;

/// Populate the given list with patterns that convert from PIM to APIM.
void populateLoweringPIMSIMDADDOpToAPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringPIMSIMDSUBOpToAPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringPIMSIMDMULOpToAPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringPIMSIMDDIVOpToAPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringPIMSIMDSCALADDOpToAPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringPIMSIMDSCALSUBOpToAPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringPIMSIMDSCALMULOpToAPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringPIMSIMDSCALDIVOpToAPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
} // namespace mlir

#endif // MLIR_CONVERSION_PIMToAPIM_PIMToAPIM_H_
