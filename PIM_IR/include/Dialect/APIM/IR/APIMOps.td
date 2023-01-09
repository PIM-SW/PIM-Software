//===----------- APIMOps.td - APIM Dialect Operation Definitions ------------===//
//
//===-------------------------- corelab heelim ----------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef APIM_OPS
#define APIM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// APIM dialect definition.
//===----------------------------------------------------------------------===//

def APIM_Dialect : Dialect {
	let name = "apim";
	let cppNamespace = "::mlir::apim";
}

//===----------------------------------------------------------------------===//
//  APIM operations definition.
//===----------------------------------------------------------------------===//

def APIM_VectorImmOp : Op<APIM_Dialect, "vector_imm",[]> {
	let summary = "Single element addition operation";
	let description = [{
		"apim.vector_imm"
	}];

	let arguments = (ins 
			F32MemRef:$vec,
			I32Attr:$inc,
			F32:$imm,
			I32Attr:$width,
			I32Attr:$type
			);
	let results = (outs 
			F32MemRef:$res
		      );
	let builders = [
		OpBuilder<(ins 
			"Value":$vec, 
			"IntegerAttr":$inc, 
			"Value":$imm,
			"IntegerAttr":$width, 
			"IntegerAttr":$type
			)>
	];
}

def APIM_VectorOp : Op<APIM_Dialect, "vector"> {
	let summary = "vector";
	let description = [{ 
		"apim.vector".
	}];
	let arguments = (ins 
			F32MemRef:$X,
			I32Attr:$incX,
			F32MemRef:$Y,
			I32Attr:$incY,
			I32Attr:$width,
			I32Attr:$type
			);
	let results = (outs F32MemRef:$res);
	let builders = [
		OpBuilder<(ins 
			"Value":$X, 
			"IntegerAttr":$incX, 
			"Value":$Y, 
			"IntegerAttr":$incY,
			"IntegerAttr":$width, 
			"IntegerAttr":$type
			)>
	];
}

def APIM_MVOp : Op<APIM_Dialect, "mv"> {
	let summary = "Matrix-Vector multiplication operation";
	let description = [{
		The "mv_mul" operation performs Matrix-Vector multiplication between two
			tensors. The shapes of the tensor operands are expected to match in specific way.
	}];

	let arguments = (ins
			F32MemRef:$A,
			I32Attr:$lda,
			F32MemRef:$X,
			I32Attr:$incX
			);
	let results = (outs
			F32MemRef:$Y
		      );
	let builders = [
		OpBuilder<(ins "Value":$A, "IntegerAttr":$lda, "Value":$X, "IntegerAttr":$incX)>
	];  		
}

def APIM_SetImmOp : Op<APIM_Dialect, "set_imm", []> {
	let summary = "set immediate value operation";
	let description = [{
		The "set_imm" operation
	}];

	let arguments = (ins F32:$input);
	let results = (outs F32:$output);
	let builders = [
		OpBuilder<(ins "Value":$input)>
	];
}

def APIM_SetXbarOp : Op<APIM_Dialect, "set_xbar", []> {
	let summary = "set crossbar value operation";
	let description = [{
		The "set_xbar" operation
	}];

	let arguments = (ins F32MemRef:$input);
}

def APIM_CopyOp : Op<APIM_Dialect, "copy", []> {
	let summary = "copy operation";
	let description = [{
		The "copy" operation
	}];

	let arguments = (ins F32MemRef:$src, F32MemRef:$dst);
}

def APIM_LoadOp : Op<APIM_Dialect, "load", []> {
	let summary = "load operation";
	let description = [{
		The "load" operation
	}];

	let arguments = (ins F32MemRef:$input);
//	let results = (outs F32MemRef:$val);
//	let builders = [
//		OpBuilder<(ins "Value":$input)>
//	];
}

def APIM_StoreOp : Op<APIM_Dialect, "store", []> {
	let summary = "store operation";
	let description = [{
		The "store" operation
	}];

	let arguments = (ins 
			F32MemRef:$val,
			I32Attr:$width 
			);
	//  let builders = [
	//    OpBuilder<(ins "IntegerAttr":$input, "Value":$val)>
	//	];
}

def APIM_SendOp : Op<APIM_Dialect, "send", []> {
	let summary = "send operation";
	let description = [{
		The "send" operation
	}];

	let arguments = (ins 
			F32MemRef:$input, 
			I32Attr:$tid, 
			I32Attr:$width
			);
}

def APIM_ReceiveOp : Op<APIM_Dialect, "receive", []> {
	let summary = "receive operation";
	let description = [{
		The "receive" operation
	}];

	let arguments = (ins 
			I32Attr:$tid, 
			//I32:$tid, 
			I32Attr:$width
			//I32:$width
			);
	let results = (outs F32MemRef:$res);
}

#endif // APIM_OPS
