//===- DPIMToLLVM.cpp - conversion from PIM to DPIM dialect ----------===//
//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

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
#include "Conversion/DPIMToLLVM/DPIMToLLVM.h"

#include "Dialect/PIM/IR/PIMOps.hpp"
#include "Dialect/DPIM/IR/DPIMOps.hpp"

#include <iostream>
#include <cstring>

using namespace mlir;
using namespace dpim;

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

static LLVM::LLVMFuncOp getOrInsertPrintf(PatternRewriter &rewriter,
		ModuleOp module) {

	MLIRContext *context = rewriter.getContext();
	auto printfRef = module.lookupSymbol<LLVM::LLVMFuncOp>("printf");
	if(!printfRef) {
		auto llvmI32Ty = IntegerType::get(context, 32);
		auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
		auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
				/*isVarArg=*/true);

		// Insert the printf function into the body of the parent module.
		PatternRewriter::InsertionGuard insertGuard(rewriter);
		rewriter.setInsertionPointToStart(module.getBody());
		printfRef=rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
	}
	return printfRef;
}

static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
		StringRef name, StringRef value,
		ModuleOp module) {
	// Create the global at the entry of the module.
	LLVM::GlobalOp global;
	if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
		OpBuilder::InsertionGuard insertGuard(builder);
		builder.setInsertionPointToStart(module.getBody());
		auto type = LLVM::LLVMArrayType::get(
				IntegerType::get(builder.getContext(), 8), value.size());
		global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
				LLVM::Linkage::Internal, name,
				builder.getStringAttr(value),
				/*alignment=*/0);
	}

	// Get the pointer to the first character in the global string.
	Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
	Value cst0 = builder.create<LLVM::ConstantOp>(
			loc, IntegerType::get(builder.getContext(), 64),
			builder.getIntegerAttr(builder.getIndexType(), 0));
	return builder.create<LLVM::GEPOp>(
			loc,
			LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
			globalPtr, ArrayRef<Value>({cst0, cst0}));
}

static LLVM::CallOp insertPrintfCall(PatternRewriter &rewriter, ModuleOp module, char* funcName, uint32_t operand0, uint32_t operand1) {
	auto loc = module.getLoc();
	auto printfRef = getOrInsertPrintf(rewriter, module);
	LLVM::ConstantOp arg0 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(operand0));
	LLVM::ConstantOp arg1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(operand1));
	char opNameStr[50] = "dpim.op\t";
	char placeholder[20] = "\targ %d\targ %d\n\0";
	strcat(opNameStr, funcName);
	strcat(opNameStr, placeholder);
	Value global = getOrCreateGlobalString(loc, rewriter, funcName, StringRef(opNameStr, strlen(opNameStr)+1), module);
	Value nl = getOrCreateGlobalString(loc, rewriter, "nl", StringRef("\n\0", 2), module);
	SmallVector<Value, 4> args;
	args.push_back(global);
	args.push_back(arg0);
	args.push_back(arg1);
	args.push_back(nl);
	return rewriter.create<LLVM::CallOp>(loc, printfRef, args);
}

//===----------------------------------------------------------------------===//
//Vector
//===----------------------------------------------------------------------===//
struct DPIMVectorCompOpToLLVM : public mlir::ConversionPattern {
	DPIMVectorCompOpToLLVM(MLIRContext *context)
		: ConversionPattern(mlir::dpim::VectorCompOp::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		ModuleOp module = op->getParentOfType<ModuleOp>();
		auto callop = insertPrintfCall(rewriter, module, "vector", 0, 1);
		rewriter.replaceOp(op, callop->getResult(0));
		return success();
	}
};
void mlir::populateLoweringDPIMVectorCompOpToLLVMPatterns(RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<DPIMVectorCompOpToLLVM>(context);
}


//===----------------------------------------------------------------------===//
//VectorImm
//===----------------------------------------------------------------------===//
struct DPIMVectorImmOpToLLVM : public mlir::ConversionPattern {
	DPIMVectorImmOpToLLVM(MLIRContext *context)
		: ConversionPattern(mlir::dpim::VectorImmOp::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		ModuleOp module = op->getParentOfType<ModuleOp>();
		VectorImmOpAdaptor opAdaptor(operands);
		auto callop = insertPrintfCall(rewriter, module, "vectorimm", 0, 1);
		rewriter.replaceOp(op, callop->getResult(0));
		return success();
	}
};
void mlir::populateLoweringDPIMVectorImmOpToLLVMPatterns(RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<DPIMVectorImmOpToLLVM>(context);
}

//===----------------------------------------------------------------------===//
//Set
//===----------------------------------------------------------------------===//
struct DPIMSetOpToLLVM : public mlir::ConversionPattern {
	DPIMSetOpToLLVM(MLIRContext *context)
		: ConversionPattern(mlir::dpim::SetOp::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		ModuleOp module = op->getParentOfType<ModuleOp>();
		SetOpAdaptor opAdaptor(operands);
		auto callop = insertPrintfCall(rewriter, module, "set", 0, 1);
		//rewriter.replaceOp(op, callop->getResult(0));
		rewriter.eraseOp(op);//, callop->getResult(0));
		return success();
	}
};

struct DPIMSetImmOpToLLVM : public mlir::ConversionPattern {
	DPIMSetImmOpToLLVM(MLIRContext *context)
		: ConversionPattern(mlir::dpim::SetImmOp::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		ModuleOp module = op->getParentOfType<ModuleOp>();
		SetOpAdaptor opAdaptor(operands);
		auto callop = insertPrintfCall(rewriter, module, "set", 0, 1);
		//rewriter.replaceOp(op, callop->getResult(0));
		rewriter.eraseOp(op);//, callop->getResult(0));
		return success();
	}
};

void mlir::populateLoweringDPIMSetOpToLLVMPatterns(RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<DPIMSetOpToLLVM>(context);
	patterns.insert<DPIMSetImmOpToLLVM>(context);
}

//===----------------------------------------------------------------------===//
//Store
//===----------------------------------------------------------------------===//
struct DPIMStoreOpToLLVM : public mlir::ConversionPattern {
	DPIMStoreOpToLLVM(MLIRContext *context)
		: ConversionPattern(mlir::dpim::StoreOp::getOperationName(), 1, context) {}

	LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<Value> operands,
			mlir::ConversionPatternRewriter &rewriter) const final {
		ModuleOp module = op->getParentOfType<ModuleOp>();
		StoreOpAdaptor opAdaptor(operands);
		auto callop = insertPrintfCall(rewriter, module, "store", 0, 1);
		rewriter.eraseOp(op);//, callop->getResult(0));
		return success();
	}
};
void mlir::populateLoweringDPIMStoreOpToLLVMPatterns(RewritePatternSet &patterns, MLIRContext *context) {
	patterns.insert<DPIMStoreOpToLLVM>(context);
}

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

namespace{
	struct ConvertDPIMToLLVMPass
		: public PassWrapper<ConvertDPIMToLLVMPass, OperationPass<ModuleOp>>{
			void getDependentDialects(mlir::DialectRegistry &registry) const override {
				registry.insert<PIMOpsDialect, DPIMOpsDialect>();
			}  
			void runOnOperation() final;
			StringRef getArgument() const override {return "convert-dpim-llvm";}
		};
}

void ConvertDPIMToLLVMPass::runOnOperation() {
	ModuleOp module = getOperation();
	ConversionTarget target(getContext());

	target.addIllegalDialect<DPIMOpsDialect>();
	target.addLegalDialect<LLVM::LLVMDialect>();

	RewritePatternSet patterns(&getContext());

	// ----------- Adding Patterns for Lowering Pass ----------- //
	populateLoweringDPIMVectorCompOpToLLVMPatterns(patterns, &getContext());
	populateLoweringDPIMVectorImmOpToLLVMPatterns(patterns, &getContext());
	populateLoweringDPIMSetOpToLLVMPatterns(patterns, &getContext());
	populateLoweringDPIMStoreOpToLLVMPatterns(patterns, &getContext());
	// --------------------------------------------------------- //
	if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
		signalPassFailure();
	}	
}
std::unique_ptr<mlir::Pass> mlir::createConvertDPIMToLLVMPass() {
	return std::make_unique<ConvertDPIMToLLVMPass>();
}
