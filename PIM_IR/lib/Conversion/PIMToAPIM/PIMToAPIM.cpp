//===- PIMToAPIM.cpp - conversion from PIM Dialect to APIM Dialect ----------===//
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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

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
#include "Conversion/PIMToAPIM/PIMToAPIM.h"

#include "Dialect/PIM/IR/PIMOps.hpp"
#include "Dialect/APIM/IR/APIMOps.hpp"

#include <iostream>

using namespace mlir;
using namespace apim;

//===----------------------------------------------------------------------===//
//SIMDADD
//===----------------------------------------------------------------------===//
struct PIMSIMDADDOpToAPIM : public mlir::ConversionPattern {
	PIMSIMDADDOpToAPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_ADD_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(0);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		SIMD_ADD_OpAdaptor operandAdaptor(operands);
		auto loadop0 = rewriter.create<LoadOp>(loc, operandAdaptor.X());
		auto loadop1 = rewriter.create<LoadOp>(loc, operandAdaptor.Y());
		auto vectorop = rewriter.create<VectorOp>(loc, operandAdaptor.X(), inc, operandAdaptor.Y(), inc, width, type);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);

		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDADDOpToAPIMPatterns(RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDADDOpToAPIM>(context);
}

//===----------------------------------------------------------------------===//
//SIMDSUB
//===----------------------------------------------------------------------===//
struct PIMSIMDSUBOpToAPIM : public mlir::ConversionPattern {
	PIMSIMDSUBOpToAPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_SUB_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(1);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		SIMD_SUB_OpAdaptor operandAdaptor(operands);
		auto loadop0 = rewriter.create<LoadOp>(loc, operandAdaptor.X());
		auto loadop1 = rewriter.create<LoadOp>(loc, operandAdaptor.Y());
		auto vectorop = rewriter.create<VectorOp>(loc, operandAdaptor.X(), inc, operandAdaptor.Y(), inc, width, type);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDSUBOpToAPIMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDSUBOpToAPIM>(context);
}

//===----------------------------------------------------------------------===//
//SIMDMUL
//===----------------------------------------------------------------------===//
struct PIMSIMDMULOpToAPIM : public mlir::ConversionPattern {
	PIMSIMDMULOpToAPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_MUL_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(2);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		SIMD_MUL_OpAdaptor operandAdaptor(operands);
		auto loadop0 = rewriter.create<LoadOp>(loc, operandAdaptor.X());
		auto loadop1 = rewriter.create<LoadOp>(loc, operandAdaptor.Y());
		auto vectorop = rewriter.create<VectorOp>(loc, operandAdaptor.X(), inc, operandAdaptor.Y(), inc, width, type);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDMULOpToAPIMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDMULOpToAPIM>(context);
}

//===----------------------------------------------------------------------===//
//SIMDDIV
//===----------------------------------------------------------------------===//
struct PIMSIMDDIVOpToAPIM : public mlir::ConversionPattern {
	PIMSIMDDIVOpToAPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_DIV_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(3);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		SIMD_DIV_OpAdaptor operandAdaptor(operands);
		auto loadop0 = rewriter.create<LoadOp>(loc, operandAdaptor.X());
		auto loadop1 = rewriter.create<LoadOp>(loc, operandAdaptor.Y());
		auto vectorop = rewriter.create<VectorOp>(loc, operandAdaptor.X(), inc, operandAdaptor.Y(), inc, width, type);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDDIVOpToAPIMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDDIVOpToAPIM>(context);
}

//===----------------------------------------------------------------------===//
//SIMDSCALADD
//===----------------------------------------------------------------------===//
struct PIMSIMDSCALADDOpToAPIM : public mlir::ConversionPattern {
	PIMSIMDSCALADDOpToAPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_SCAL_ADD_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(0);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		SIMD_SCAL_ADD_OpAdaptor operandAdaptor(operands);
		auto loadop = rewriter.create<LoadOp>(loc, operandAdaptor.Y());
		auto vectorop = rewriter.create<VectorImmOp>(loc, operandAdaptor.Y(), inc, operandAdaptor.X(), width, type);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDSCALADDOpToAPIMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDSCALADDOpToAPIM>(context);
}

//===----------------------------------------------------------------------===//
//SIMDSCALSUB
//===----------------------------------------------------------------------===//
struct PIMSIMDSCALSUBOpToAPIM : public mlir::ConversionPattern {
	PIMSIMDSCALSUBOpToAPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_SCAL_SUB_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(1);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		SIMD_SCAL_SUB_OpAdaptor operandAdaptor(operands);
		auto loadop = rewriter.create<LoadOp>(loc, operandAdaptor.Y());
		auto vectorop = rewriter.create<VectorImmOp>(loc, operandAdaptor.Y(), inc, operandAdaptor.X(), width, type);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDSCALSUBOpToAPIMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDSCALSUBOpToAPIM>(context);
}

//===----------------------------------------------------------------------===//
//SIMDSCALMUL
//===----------------------------------------------------------------------===//
struct PIMSIMDSCALMULOpToAPIM : public mlir::ConversionPattern {
	PIMSIMDSCALMULOpToAPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_SCAL_MUL_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(2);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		SIMD_SCAL_MUL_OpAdaptor operandAdaptor(operands);
		auto loadop = rewriter.create<LoadOp>(loc, operandAdaptor.Y());
		auto vectorop = rewriter.create<VectorImmOp>(loc, operandAdaptor.Y(), inc, operandAdaptor.X(), width, type);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDSCALMULOpToAPIMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDSCALMULOpToAPIM>(context);
}

//===----------------------------------------------------------------------===//
//SIMDSCALDIV
//===----------------------------------------------------------------------===//
struct PIMSIMDSCALDIVOpToAPIM : public mlir::ConversionPattern {
	PIMSIMDSCALDIVOpToAPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_SCAL_DIV_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(3);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		SIMD_SCAL_DIV_OpAdaptor operandAdaptor(operands);
		auto loadop = rewriter.create<LoadOp>(loc, operandAdaptor.Y());
		auto vectorop = rewriter.create<VectorImmOp>(loc, operandAdaptor.Y(), inc, operandAdaptor.X(), width, type);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDSCALDIVOpToAPIMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDSCALDIVOpToAPIM>(context);
}

//===----------------------------------------------------------------------===//

namespace{
	struct ConvertPIMToAPIMPass
		: public PassWrapper<ConvertPIMToAPIMPass, OperationPass<ModuleOp>>{
			void getDependentDialects(mlir::DialectRegistry &registry) const override {
				registry.insert<PIMOpsDialect, APIMOpsDialect>();
			}
			void runOnOperation() final;
			StringRef getArgument() const override { return "convert-apim";}
		};
}

void ConvertPIMToAPIMPass::runOnOperation() {
	ModuleOp module = getOperation();
	ConversionTarget target(getContext());

	target.addIllegalDialect<PIMOpsDialect>();
	target.addLegalDialect<APIMOpsDialect>();

	RewritePatternSet patterns(&getContext());

	// ----------- Adding Patterns for Lowering Pass ----------- //
	populateLoweringPIMSIMDADDOpToAPIMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDSUBOpToAPIMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDMULOpToAPIMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDDIVOpToAPIMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDSCALADDOpToAPIMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDSCALSUBOpToAPIMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDSCALMULOpToAPIMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDSCALDIVOpToAPIMPatterns(patterns, &getContext());
	// --------------------------------------------------------- //
	if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
		signalPassFailure();
	}
}

std::unique_ptr<mlir::Pass> mlir::createConvertPIMToAPIMPass() {
	return std::make_unique<ConvertPIMToAPIMPass>();
}

