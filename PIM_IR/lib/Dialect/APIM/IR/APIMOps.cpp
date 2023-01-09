//===------------- APIMOps.cpp - APIM dialect ops ---------------*- C++ -*-===//
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

#include "Dialect/APIM/IR/APIMOps.hpp"

using namespace mlir;
using namespace apim;

//===----------------------------------------------------------------------===//
// APIM dialect.
//===----------------------------------------------------------------------===//

APIMOpsDialect::APIMOpsDialect(MLIRContext *context)
	: Dialect(getDialectNamespace(), context, TypeID::get<APIMOpsDialect>()) {
		addOperations<
#define GET_OP_LIST
#include "Dialect/APIM/IR/APIMOps.cpp.inc"
			>();
	}

#define GET_OP_CLASSES
#include "Dialect/APIM/IR/APIMOps.cpp.inc"

//===----------------------------------------------------------------------===//
// VectorImmOp

void VectorImmOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
		mlir::Value X,
		mlir::IntegerAttr inc, 
		mlir::Value imm, 
		mlir::IntegerAttr width,
		mlir::IntegerAttr type
		) {
	state.addTypes(X.getType());
	state.addOperands({X, imm});
	state.addAttribute("inc", inc);
	state.addAttribute("width", width);
	state.addAttribute("type", type);
}

//===----------------------------------------------------------------------===//
// VectorOp

void VectorOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
		mlir::Value X, 
		mlir::IntegerAttr incX, 
		mlir::Value Y, 
		mlir::IntegerAttr incY, 
		mlir::IntegerAttr width,
		mlir::IntegerAttr type
		) {
	state.addTypes(X.getType());//MemRefType(builder.getF32Type()));
	state.addOperands({X, Y});
	state.addAttribute("incX", incX);
	state.addAttribute("incY", incY);
	state.addAttribute("width", width);
	state.addAttribute("type", type);
}

//===----------------------------------------------------------------------===//
// MVOp

void MVOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
		mlir::Value A, 
		mlir::IntegerAttr lda, 
		mlir::Value X, 
		mlir::IntegerAttr incX
		) {
	state.addTypes(X.getType());
	state.addOperands({A, X});
	state.addAttribute("lda", lda);
	state.addAttribute("incX", incX);
}

//===----------------------------------------------------------------------===//
// LoadOp
/*
void LoadOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
		mlir::Value input) {
//	state.addTypes(input.getType());//MemRefType(builder.getF32Type()));
	state.addOperands({input});
}
*/
//===----------------------------------------------------------------------===//
// SetImmOp

void SetImmOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value input) {
	state.addTypes(input.getType());//MemRefType(builder.getF32Type()));
	state.addOperands({input});
}


