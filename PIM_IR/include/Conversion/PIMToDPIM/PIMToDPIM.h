//===- LinalgToLLVM.h - Utils to convert from the linalg dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_PIMToDPIM_PIMToDPIM_H_
#define MLIR_CONVERSION_PIMToDPIM_PIMToDPIM_H_
#include "Pass/Passes.h"
#include "Dialect/PIM/IR/PIMOps.hpp"
#include "Dialect/DPIM/IR/DPIMOps.hpp"
#include <memory>

namespace mlir {

class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;

/// Populate the given list with patterns that convert from PIM to DPIM.
void populateLoweringPIMSIMDADDOpToDPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringPIMSIMDSUBOpToDPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringPIMSIMDMULOpToDPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringPIMSIMDDIVOpToDPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringPIMSIMDSCALADDOpToDPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringPIMSIMDSCALSUBOpToDPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringPIMSIMDSCALMULOpToDPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringPIMSIMDSCALDIVOpToDPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringPIMMACOpToDPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
void populateLoweringPIMACCOpToDPIMPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
} // namespace mlir

#endif // MLIR_CONVERSION_PIMToDPIM_PIMToDPIM_H_
