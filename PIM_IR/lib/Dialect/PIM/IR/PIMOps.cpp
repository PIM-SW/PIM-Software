//===--------- PIMAPIOps.cpp - PIMAPI dialect ops ---------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <queue>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Support/TypeID.h"

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"

#include "Dialect/PIM/IR/PIMOps.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// PIM dialect.
//===----------------------------------------------------------------------===//

PIMOpsDialect::PIMOpsDialect(MLIRContext *context)
  : Dialect(getDialectNamespace(), context, TypeID::get<PIMOpsDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "Dialect/PIM/IR/PIMOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "Dialect/PIM/IR/PIMOps.cpp.inc"

//===----------------------------------------------------------------------===//
// SIMD_ADD_Op

void SIMD_ADD_Op::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// SIMD_SUB_Op

void SIMD_SUB_Op::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// SIMD_MUL_Op

void SIMD_MUL_Op::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// SIMD_DIV_Op

void SIMD_DIV_Op::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// SIMD_SCAL_ADD_Op

void SIMD_SCAL_ADD_Op::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(rhs.getType());
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// SIMD_SCAL_SUB_Op

void SIMD_SCAL_SUB_Op::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(rhs.getType());
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// SIMD_SCAL_MUL_Op

void SIMD_SCAL_MUL_Op::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(rhs.getType());
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// SIMD_SCAL_DIV_Op

void SIMD_SCAL_DIV_Op::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(rhs.getType());
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// MAC_Op

void MAC_Op::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(builder.getF32Type());
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// ACC_Op

void ACC_Op::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value X){
  state.addTypes(builder.getF32Type());
  state.addOperands(X);
}

//===----------------------------------------------------------------------===//
// PIMConvertOp

void PIMConvertOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(lhs);
}
