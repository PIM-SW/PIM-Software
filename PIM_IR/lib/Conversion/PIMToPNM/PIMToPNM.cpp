//===- PIMToPNM.cpp - conversion from PIM to PNM dialect ----------===//
//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

#include "Pass/Passes.h"
#include "Conversion/PIMToPNM/PIMToPNM.h"

#include "Dialect/PIM/IR/PIMOps.hpp"
#include "Dialect/PNM/IR/PNMOps.hpp"

#include <iostream>

using namespace mlir;
//===----------------------------------------------------------------------===//
//SIMDADD
//===----------------------------------------------------------------------===//
struct PIMSIMDADDOpToPNM : public mlir::ConversionPattern {
	PIMSIMDADDOpToPNM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_ADD_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(0);
	 
		SIMD_ADD_OpAdaptor operandAdaptor(operands);
		auto vectorop = rewriter.create<VectorOp>(loc, 
				operandAdaptor.X(), 
				operandAdaptor.Y(),
				width,
				type
				);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDADDOpToPNMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDADDOpToPNM>(context);
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//SIMDSUB
//===----------------------------------------------------------------------===//
struct PIMSIMDSUBOpToPNM : public mlir::ConversionPattern {
	PIMSIMDSUBOpToPNM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_SUB_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(1);
	 
		SIMD_SUB_OpAdaptor operandAdaptor(operands);
		auto vectorop = rewriter.create<VectorOp>(loc, 
				operandAdaptor.X(), 
				operandAdaptor.Y(),
				width,
				type
				);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDSUBOpToPNMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDSUBOpToPNM>(context);
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//SIMDMUL
//===----------------------------------------------------------------------===//
struct PIMSIMDMULOpToPNM : public mlir::ConversionPattern {
	PIMSIMDMULOpToPNM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_MUL_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(2);
	 
		SIMD_MUL_OpAdaptor operandAdaptor(operands);
		auto vectorop = rewriter.create<VectorOp>(loc, 
				operandAdaptor.X(), 
				operandAdaptor.Y(),
				width,
				type
				);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDMULOpToPNMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDMULOpToPNM>(context);
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//SIMDDIV
//===----------------------------------------------------------------------===//
struct PIMSIMDDIVOpToPNM : public mlir::ConversionPattern {
	PIMSIMDDIVOpToPNM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_DIV_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(3);
	 
		SIMD_DIV_OpAdaptor operandAdaptor(operands);
		auto vectorop = rewriter.create<VectorOp>(loc, 
				operandAdaptor.X(), 
				operandAdaptor.Y(),
				width,
				type
				);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDDIVOpToPNMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDDIVOpToPNM>(context);
}
//===----------------------------------------------------------------------===//

namespace{
	struct ConvertPIMToPNMPass
		: public PassWrapper<ConvertPIMToPNMPass, OperationPass<ModuleOp>>{
			void getDependentDialects(mlir::DialectRegistry &registry) const override {
				registry.insert<PIMOpsDialect, PNMOpsDialect>();
			}  
			void runOnOperation() final;
		};
}

void ConvertPIMToPNMPass::runOnOperation() {
	ModuleOp module = getOperation();
	ConversionTarget target(getContext());

	target.addIllegalDialect<PIMOpsDialect>();
	target.addLegalDialect<PNMOpsDialect>();
	
	target.addLegalOp<PIMReturnOp>();
	
	RewritePatternSet patterns(&getContext());

	// ----------- Adding Patterns for Lowering Pass ----------- //
	populateLoweringPIMSIMDADDOpToPNMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDSUBOpToPNMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDMULOpToPNMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDDIVOpToPNMPatterns(patterns, &getContext());
	// --------------------------------------------------------- //
	if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
		signalPassFailure();
	}	
}
std::unique_ptr<mlir::Pass> mlir::createConvertPIMToPNMPass() {
	return std::make_unique<ConvertPIMToPNMPass>();
}
