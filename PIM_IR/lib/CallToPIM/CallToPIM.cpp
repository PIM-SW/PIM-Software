#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Pass/Passes.h"
#include "Dialect/PIM/IR/PIMOps.hpp"

#include <iostream>

using namespace mlir;

struct CallToPIMOps : public OpRewritePattern<CallOp> {
    using OpRewritePattern<CallOp>::OpRewritePattern;
    
    LogicalResult matchAndRewrite(CallOp op, PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        StringRef opName = op.getCallee();
        CallOpAdaptor opAdaptor(op);
        auto vr = ValueRange();
        if (opName.equals("SIMD_ADD_16")) {
            SIMD_ADD_OpAdaptor pimOperand(opAdaptor.getOperands(), opAdaptor.getAttributes(), {});
            rewriter.create<SIMD_ADD_Op>(loc, pimOperand.X(), pimOperand.Y());
        } else if (opName.equals("SIMD_SUB_16")) {
            SIMD_SUB_OpAdaptor pimOperand(opAdaptor.getOperands(), opAdaptor.getAttributes(), {});
            rewriter.create<SIMD_SUB_Op>(loc, pimOperand.X(), pimOperand.Y());
        } else if (opName.equals("SIMD_MUL_16")) {
            SIMD_MUL_OpAdaptor pimOperand(opAdaptor.getOperands(), opAdaptor.getAttributes(), {});
            rewriter.create<SIMD_MUL_Op>(loc, pimOperand.X(), pimOperand.Y());
        } else if (opName.equals("SIMD_DIV_16")) {
            SIMD_DIV_OpAdaptor pimOperand(opAdaptor.getOperands(), opAdaptor.getAttributes(), {});
            rewriter.create<SIMD_DIV_Op>(loc, pimOperand.X(), pimOperand.Y());
        } else if (opName.equals("SIMD_SCAL_ADD_16")) {
            SIMD_SCAL_ADD_OpAdaptor pimOperand(opAdaptor.getOperands(), opAdaptor.getAttributes(), {});
            rewriter.create<SIMD_SCAL_ADD_Op>(loc, pimOperand.X(), pimOperand.Y());
        } else if (opName.equals("SIMD_SCAL_SUB_16")) {
            SIMD_SCAL_SUB_OpAdaptor pimOperand(opAdaptor.getOperands(), opAdaptor.getAttributes(), {});
            rewriter.create<SIMD_SCAL_SUB_Op>(loc, pimOperand.X(), pimOperand.Y());
        } else if (opName.equals("SIMD_SCAL_MUL_16")) {
            SIMD_SCAL_MUL_OpAdaptor pimOperand(opAdaptor.getOperands(), opAdaptor.getAttributes(), {});
            rewriter.create<SIMD_SCAL_MUL_Op>(loc, pimOperand.X(), pimOperand.Y());
        } else if (opName.equals("SIMD_SCAL_DIV_16")) {
            SIMD_SCAL_DIV_OpAdaptor pimOperand(opAdaptor.getOperands(), opAdaptor.getAttributes(), {});
            rewriter.create<SIMD_SCAL_DIV_Op>(loc, pimOperand.X(), pimOperand.Y());
        } else if (opName.equals("MAC_16")) {
            MAC_OpAdaptor pimOperand(opAdaptor.getOperands(), opAdaptor.getAttributes(), {});
            auto tmpOp = rewriter.create<MAC_Op>(loc, pimOperand.X(), pimOperand.Y());
            vr = ValueRange(tmpOp->getResult(0));
            op.replaceAllUsesWith(tmpOp);
        } else if (opName.equals("ACC_16")) {
            ACC_OpAdaptor pimOperand(opAdaptor.getOperands(), opAdaptor.getAttributes(), {});
            auto tmpOp = rewriter.create<ACC_Op>(loc, pimOperand.X());
            vr = ValueRange(tmpOp->getResult(0));
            op.replaceAllUsesWith(tmpOp);
        } else {
            return failure();
        }
        rewriter.replaceOp(op, vr);
        return success();
    }
};

namespace {
    struct ReplaceCallToPIMOpsPass
		: public PassWrapper<ReplaceCallToPIMOpsPass, FunctionPass>{
			void getDependentDialects(mlir::DialectRegistry &registry) const override {
				registry.insert<PIMOpsDialect>();
			}  
            void runOnFunction() override;
            StringRef getArgument() const {return "pim-avail";};
        };
}

void ReplaceCallToPIMOpsPass::runOnFunction() {
    RewritePatternSet patterns(getFunction().getContext());
    patterns.add<CallToPIMOps>(getFunction().getContext());
    GreedyRewriteConfig config;
    (void)applyPatternsAndFoldGreedily(getFunction().getOperation(), std::move(patterns), config);
}

std::unique_ptr<mlir::Pass> mlir::replaceCallToPIMOps() {
    return std::make_unique<ReplaceCallToPIMOpsPass>();
}
