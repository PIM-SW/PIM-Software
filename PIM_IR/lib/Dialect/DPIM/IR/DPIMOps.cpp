//===------------- DPIMOps.cpp - DPIM dialect ops ---------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <queue>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

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


#include "Dialect/DPIM/IR/DPIMOps.hpp"

using namespace mlir;
using namespace dpim;

//===----------------------------------------------------------------------===//
// DPIM dialect.
//===----------------------------------------------------------------------===//

DPIMOpsDialect::DPIMOpsDialect(MLIRContext *context)
	: Dialect(getDialectNamespace(), context, TypeID::get<DPIMOpsDialect>()) {
		addOperations<
#define GET_OP_LIST
#include "Dialect/DPIM/IR/DPIMOps.cpp.inc"
			>();
	}

#define GET_OP_CLASSES
#include "Dialect/DPIM/IR/DPIMOps.cpp.inc"

//===----------------------------------------------------------------------===//
// VectorCompOp

void VectorCompOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
		mlir::Value X, 
		mlir::Value Y, 
		mlir::IntegerAttr width,
		mlir::IntegerAttr type
		) {
	state.addTypes(X.getType());//MemRefType(builder.getF32Type()));
	state.addOperands({X, Y});
	state.addAttribute("width", width);
	state.addAttribute("type", type);
}

//===----------------------------------------------------------------------===//
// VectorImmOp

void VectorImmOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
		mlir::Value X,
		mlir::Value imm, 
		mlir::IntegerAttr width,
		mlir::IntegerAttr type
		) {
	state.addTypes(X.getType());
	state.addOperands({X, imm});
	state.addAttribute("width", width);
	state.addAttribute("type", type);
}

//===----------------------------------------------------------------------===//
// VectorAccOp

void VectorAccOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
		mlir::Value X, 
		mlir::IntegerAttr width
		) {
	state.addTypes(X.getType());//MemRefType(builder.getF32Type()));
	state.addOperands({X});
	state.addAttribute("width", width);
}

//===----------------------------------------------------------------------===//
// SetOp

void SetOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
		mlir::Value input,
		mlir::IntegerAttr width
		) {
	state.addTypes(input.getType());
	state.addOperands({input});
	state.addAttribute("width", width);
}

void SetImmOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
		mlir::Value input,
		mlir::IntegerAttr width
		) {
	state.addTypes(MemRefType::get({}, builder.getF32Type()));//,width));
	state.addOperands({input});
	state.addAttribute("width", width);
}

//===----------------------------------------------------------------------===//
// CopyOp
/*
void CopyOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, 
		mlir::Value src,
		mlir::Value dst
		) {
	state.addOperands({src,dst});
}
*/

