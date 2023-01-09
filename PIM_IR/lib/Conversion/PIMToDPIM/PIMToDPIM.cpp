//===- PIMToDPIM.cpp - conversion from PIM Dialect to DPIM Dialect ----------===//
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
#include "Conversion/PIMToDPIM/PIMToDPIM.h"

#include "Dialect/PIM/IR/PIMOps.hpp"
#include "Dialect/DPIM/IR/DPIMOps.hpp"

#include <iostream>

using namespace mlir;
using namespace dpim;

//===----------------------------------------------------------------------===//
//SIMDADD
//===----------------------------------------------------------------------===//
struct PIMSIMDADDOpToDPIM : public mlir::ConversionPattern {
	PIMSIMDADDOpToDPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_ADD_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(0);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		SIMD_ADD_OpAdaptor operandAdaptor(operands);
		auto setop0 = rewriter.create<SetOp>(loc, operandAdaptor.X(), width);
		auto setop1 = rewriter.create<SetOp>(loc, operandAdaptor.Y(), width);
		auto vectorop = rewriter.create<VectorCompOp>(loc, operandAdaptor.X(), operandAdaptor.Y(), width, type);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDADDOpToDPIMPatterns(RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDADDOpToDPIM>(context);
}

//===----------------------------------------------------------------------===//
//SIMDSUB
//===----------------------------------------------------------------------===//
struct PIMSIMDSUBOpToDPIM : public mlir::ConversionPattern {
	PIMSIMDSUBOpToDPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_SUB_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(1);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		SIMD_SUB_OpAdaptor operandAdaptor(operands);
		auto setop0 = rewriter.create<SetOp>(loc, operandAdaptor.X(), width);
		auto setop1 = rewriter.create<SetOp>(loc, operandAdaptor.Y(), width);
		auto vectorop = rewriter.create<VectorCompOp>(loc, operandAdaptor.X(), operandAdaptor.Y(), width, type);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDSUBOpToDPIMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDSUBOpToDPIM>(context);
}

//===----------------------------------------------------------------------===//
//SIMDMUL
//===----------------------------------------------------------------------===//
struct PIMSIMDMULOpToDPIM : public mlir::ConversionPattern {
	PIMSIMDMULOpToDPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_MUL_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(2);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		SIMD_MUL_OpAdaptor operandAdaptor(operands);
		auto setop0 = rewriter.create<SetOp>(loc, operandAdaptor.X(), width);
		auto setop1 = rewriter.create<SetOp>(loc, operandAdaptor.Y(), width);
		auto vectorop = rewriter.create<VectorCompOp>(loc, operandAdaptor.X(), operandAdaptor.Y(), width, type);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDMULOpToDPIMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDMULOpToDPIM>(context);
}

//===----------------------------------------------------------------------===//
//SIMDDIV
//===----------------------------------------------------------------------===//
struct PIMSIMDDIVOpToDPIM : public mlir::ConversionPattern {
	PIMSIMDDIVOpToDPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_DIV_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(3);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		SIMD_DIV_OpAdaptor operandAdaptor(operands);
		auto setop0 = rewriter.create<SetOp>(loc, operandAdaptor.X(), width);
		auto setop1 = rewriter.create<SetOp>(loc, operandAdaptor.Y(), width);
		auto vectorop = rewriter.create<VectorCompOp>(loc, operandAdaptor.X(), operandAdaptor.Y(), width, type);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDDIVOpToDPIMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDDIVOpToDPIM>(context);
}

//===----------------------------------------------------------------------===//
//SIMDSCALADD
//===----------------------------------------------------------------------===//
struct PIMSIMDSCALADDOpToDPIM : public mlir::ConversionPattern {
	PIMSIMDSCALADDOpToDPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_SCAL_ADD_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(0);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		SIMD_SCAL_ADD_OpAdaptor operandAdaptor(operands);
		auto setop0 = rewriter.create<SetImmOp>(loc, operandAdaptor.X(), width);
		auto setop1 = rewriter.create<SetOp>(loc, operandAdaptor.Y(), width);
		auto vectorop = rewriter.create<VectorCompOp>(loc, setop0->getResult(0), operandAdaptor.Y(), width, type);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDSCALADDOpToDPIMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDSCALADDOpToDPIM>(context);
}

//===----------------------------------------------------------------------===//
//SIMDSCALSUB
//===----------------------------------------------------------------------===//
struct PIMSIMDSCALSUBOpToDPIM : public mlir::ConversionPattern {
	PIMSIMDSCALSUBOpToDPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_SCAL_SUB_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(1);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		SIMD_SCAL_SUB_OpAdaptor operandAdaptor(operands);
		auto setop0 = rewriter.create<SetImmOp>(loc, operandAdaptor.X(), width);
		auto setop1 = rewriter.create<SetOp>(loc, operandAdaptor.Y(), width);
		auto vectorop = rewriter.create<VectorCompOp>(loc, setop0->getResult(0), operandAdaptor.Y(), width, type);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDSCALSUBOpToDPIMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDSCALSUBOpToDPIM>(context);
}

//===----------------------------------------------------------------------===//
//SIMDSCALMUL
//===----------------------------------------------------------------------===//
struct PIMSIMDSCALMULOpToDPIM : public mlir::ConversionPattern {
	PIMSIMDSCALMULOpToDPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_SCAL_MUL_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(2);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		SIMD_SCAL_MUL_OpAdaptor operandAdaptor(operands);
		auto setop0 = rewriter.create<SetImmOp>(loc, operandAdaptor.X(), width);
		auto setop1 = rewriter.create<SetOp>(loc, operandAdaptor.Y(), width);
		auto vectorop = rewriter.create<VectorCompOp>(loc, setop0->getResult(0), operandAdaptor.Y(), width, type);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDSCALMULOpToDPIMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDSCALMULOpToDPIM>(context);
}

//===----------------------------------------------------------------------===//
//SIMDSCALDIV
//===----------------------------------------------------------------------===//
struct PIMSIMDSCALDIVOpToDPIM : public mlir::ConversionPattern {
	PIMSIMDSCALDIVOpToDPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_SCAL_DIV_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(3);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		SIMD_SCAL_DIV_OpAdaptor operandAdaptor(operands);
		auto setop0 = rewriter.create<SetImmOp>(loc, operandAdaptor.X(), width);
		auto setop1 = rewriter.create<SetOp>(loc, operandAdaptor.Y(), width);
		auto vectorop = rewriter.create<VectorCompOp>(loc, setop0->getResult(0), operandAdaptor.Y(), width, type);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);
		rewriter.replaceOp(op, vectorop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMSIMDSCALDIVOpToDPIMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMSIMDSCALDIVOpToDPIM>(context);
}

//===----------------------------------------------------------------------===//
//MAC
//===----------------------------------------------------------------------===//
struct PIMMACOpToDPIM : public mlir::ConversionPattern {
	PIMMACOpToDPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_SCAL_DIV_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(3);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		MAC_OpAdaptor operandAdaptor(operands);
		auto setop0 = rewriter.create<SetImmOp>(loc, operandAdaptor.X(), width);
		auto setop1 = rewriter.create<SetOp>(loc, operandAdaptor.Y(), width);
		auto vectorop = rewriter.create<VectorCompOp>(loc, operandAdaptor.X(), operandAdaptor.Y(), width, type);
		auto vectoraccop = rewriter.create<VectorAccOp>(loc, operandAdaptor.Y(), width);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.Y(), width);
		rewriter.replaceOp(op, vectoraccop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMMACOpToDPIMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMMACOpToDPIM>(context);
}

//===----------------------------------------------------------------------===//
//ACC
//===----------------------------------------------------------------------===//
struct PIMACCOpToDPIM : public mlir::ConversionPattern {
	PIMACCOpToDPIM(MLIRContext *context)
		: ConversionPattern(mlir::SIMD_SCAL_DIV_Op::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr width = rewriter.getI32IntegerAttr(16);
		IntegerAttr type = rewriter.getI32IntegerAttr(3);
		IntegerAttr inc = rewriter.getI32IntegerAttr(1);

		ACC_OpAdaptor operandAdaptor(operands);
		auto setop0 = rewriter.create<SetImmOp>(loc, operandAdaptor.X(), width);
		auto vectoraccop = rewriter.create<VectorAccOp>(loc, operandAdaptor.X(), width);
		auto storeop = rewriter.create<StoreOp>(loc, operandAdaptor.X(), width);
		rewriter.replaceOp(op, vectoraccop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringPIMACCOpToDPIMPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMACCOpToDPIM>(context);
}

//===----------------------------------------------------------------------===//

namespace{
	struct ConvertPIMToDPIMPass
		: public PassWrapper<ConvertPIMToDPIMPass, OperationPass<ModuleOp>>{
			void getDependentDialects(mlir::DialectRegistry &registry) const override {
				registry.insert<PIMOpsDialect, DPIMOpsDialect>();
			}
			void runOnOperation() final;
			StringRef getArgument() const override { return "convert-dpim";}
		};
}

void ConvertPIMToDPIMPass::runOnOperation() {
	ModuleOp module = getOperation();
	ConversionTarget target(getContext());

	target.addIllegalDialect<PIMOpsDialect>();
	target.addLegalDialect<DPIMOpsDialect>();

	RewritePatternSet patterns(&getContext());

	// ----------- Adding Patterns for Lowering Pass ----------- //
	populateLoweringPIMSIMDADDOpToDPIMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDSUBOpToDPIMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDMULOpToDPIMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDDIVOpToDPIMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDSCALADDOpToDPIMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDSCALSUBOpToDPIMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDSCALMULOpToDPIMPatterns(patterns, &getContext());
	populateLoweringPIMSIMDSCALDIVOpToDPIMPatterns(patterns, &getContext());
	populateLoweringPIMMACOpToDPIMPatterns(patterns, &getContext());
	populateLoweringPIMACCOpToDPIMPatterns(patterns, &getContext());
	// --------------------------------------------------------- //
	if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
		signalPassFailure();
	}
}

std::unique_ptr<mlir::Pass> mlir::createConvertPIMToDPIMPass() {
	return std::make_unique<ConvertPIMToDPIMPass>();
}

