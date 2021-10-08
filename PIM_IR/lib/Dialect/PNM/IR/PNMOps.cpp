//===--------- PNMAPIOps.cpp - PIMAPI dialect ops ---------------*- C++ -*-===//
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

#include "Dialect/PNM/IR/PNMOps.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// PNM dialect.
//===----------------------------------------------------------------------===//

PNMOpsDialect::PNMOpsDialect(MLIRContext *context)
  : Dialect(getDialectNamespace(), context, TypeID::get<PNMOpsDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "Dialect/PNM/IR/PNMOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "Dialect/PNM/IR/PNMOps.cpp.inc"

//===----------------------------------------------------------------------===//
// VectorOp
/*
void VectorOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value X, mlir::Value Y, mlir::Value width) {
  state.addTypes(X.getType());
  state.addOperands({X, Y, width});
}
*/
void VectorOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value X, mlir::Value Y, mlir::IntegerAttr width, mlir::IntegerAttr type) {
  state.addTypes(X.getType());
  state.addOperands({X, Y});
	state.addAttribute("width", width);
	state.addAttribute("type", type);
}
//===----------------------------------------------------------------------===//
// VectorImmOp

void VectorImmOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value X, mlir::Value imm, mlir::IntegerAttr width, mlir::IntegerAttr type) {
  state.addTypes(X.getType());
  state.addOperands({X, imm});
	state.addAttribute("width", width);
	state.addAttribute("type", type);
}

//===----------------------------------------------------------------------===//
// SetOp

void SetOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value input) {
  state.addTypes(input.getType());//MemRefType(builder.getF32Type()));
  //state.addTypes(builder.getF32Type());
  state.addOperands({input});
}


