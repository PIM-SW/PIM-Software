//===- PIMToAPIM.cpp - conversion from PIM to APIM dialect ----------===//
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
#include "Conversion/PIMCommon/PIMCommon.h"

#include "Dialect/PIM/IR/PIMOps.hpp"

#include <iostream>

using namespace mlir;

//===----------------------------------------------------------------------===//
//MVMulOp
//===----------------------------------------------------------------------===//
struct PIMMMMulOpToPIMMVMul : public mlir::ConversionPattern {
	PIMMMMulOpToPIMMVMul(MLIRContext *context)
		: ConversionPattern(mlir::MMMulOp::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		auto loc = op->getLoc();
		IntegerAttr zero = rewriter.getI64IntegerAttr(0);	// task_id = 0
	 
		MMMulOpAdaptor operandAdaptor(operands);
		auto setxbarop = rewriter.create<SetXbarOp>(loc, operandAdaptor.A());
		auto mvop = rewriter.create<MVMulOp>(loc, loadop0.getResult(), operandAdaptor.lda(), loadop1.getResult(), operandAdaptor.incX());
		return success();
	}
};

void mlir::populateConvertingPIMMMMulOpToPIMMVMulPatterns(
		RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<PIMMMMulOpToPIMMVMUL>(context);
}
//===----------------------------------------------------------------------===//

namespace{
	struct ConvertPIMCommonPass
		: public PassWrapper<ConvertPIMCommonPass, OperationPass<ModuleOp>>{
			void getDependentDialects(mlir::DialectRegistry &registry) const override {
				registry.insert<PIMOpsDialect>();
			}  
			void runOnOperation() final;
		};
}

void ConvertPIMCommonPass::runOnOperation() {
	ModuleOp module = getOperation();
	ConversionTarget target(getContext());

	RewritePatternSet patterns(&getContext());

	// ----------- Adding Patterns for Lowering Pass ----------- //
	populateConvertingPIMMMMulOpToPIMMVMulPatterns(patterns, &getContext());
	// --------------------------------------------------------- //
	if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
		signalPassFailure();
	}	
}
std::unique_ptr<mlir::Pass> mlir::createConvertPIMCommonMPass() {
	return std::make_unique<ConvertPIMCommonPass>();
}
