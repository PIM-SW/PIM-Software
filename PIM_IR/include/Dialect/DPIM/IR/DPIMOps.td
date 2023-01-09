//===----------- DPIMOps.td - DPIM Dialect Operation Definitions ------------===//
//
//===-------------------------- corelab heelim ----------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef DPIM_OPS
#define DPIM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// DPIM dialect definition.
//===----------------------------------------------------------------------===//

def DPIM_Dialect : Dialect {
	let name = "dpim";
	let cppNamespace = "::mlir::dpim";
}

//===----------------------------------------------------------------------===//
//  DPIM operations definition.
//===----------------------------------------------------------------------===//

def DPIM_VectorCompOp : Op<DPIM_Dialect, "vector_comp"> {
	let summary = "vector_comp";
	let description = [{ 
		"dpim.vector_comp".
	}];
	let arguments = (ins 
			F32MemRef:$X,
			F32MemRef:$Y,
			I32Attr:$width,
			I32Attr:$type
			);
	let results = (outs F32MemRef:$res);
	let builders = [
		//	OpBuilder<(ins "Value":$X, "Value":$Y, "IntegerAttr":$width)>
		OpBuilder<(ins "Value":$X, "Value":$Y, "IntegerAttr":$width, "IntegerAttr":$type)>
	];
	let assemblyFormat = [{
		`(`$type $X $Y $width`)` attr-dict `(` type($X) type($Y) type($res)`)`
	}];
}

def DPIM_VectorImmOp : Op<DPIM_Dialect, "vector_imm"> {
	let summary = "Single element addition operation";
	let description = [{
		"pnm.vector_imm".
	}];

	let arguments = (ins 
			F32MemRef:$vec,
			F32:$imm,
			I32Attr:$width,
			I32Attr:$type
			);
	let results = (outs 
			F32MemRef:$res
		      );
	let builders = [
		//  OpBuilder<(ins "Value":$vec, "Value":$imm, "Value":$width)>
		OpBuilder<(ins "Value":$vec, "Value":$imm, "IntegerAttr":$width, "IntegerAttr":$type)>
	];
	let assemblyFormat = [{
		`(`$type $imm $vec $width`)` attr-dict `(` type($imm) type($vec) type($res)`)`
	}];
}

def DPIM_VectorAccOp : Op<DPIM_Dialect, "vector_acc"> {
	let summary = "Single element addition operation";
	let description = [{
		"pnm.vector_imm".
	}];

	let arguments = (ins 
			F32MemRef:$vec,
			I32Attr:$width
			);
	let results = (outs 
			F32:$res
		      );
	let builders = [
		OpBuilder<(ins "Value":$vec, "IntegerAttr":$width)>
	];
}

def DPIM_SetOp : Op<DPIM_Dialect, "set", []> {
	let summary = "set immediate value operation";
	let description = [{
		The "set" operation, set immediate value to register
	}];

	let arguments = (ins 
			F32MemRef:$input,
			I32Attr:$width
			);
	let results = (outs F32MemRef:$output);
	let builders = [
		OpBuilder<(ins "Value":$input, "IntegerAttr":$width)>
	];
}

def DPIM_SetImmOp : Op<DPIM_Dialect, "set_imm", []> {
	let summary = "set immediate value operation";
	let description = [{
		The "set" operation, set immediate value to register
	}];

	let arguments = (ins 
			F32:$input,
			I32Attr:$width
			);
	let results = (outs F32MemRef:$output);
	let builders = [
		OpBuilder<(ins "Value":$input, "IntegerAttr":$width)>
	];
}

def DPIM_StoreOp : Op<DPIM_Dialect, "store", []> {
	let summary = "store operation";
	let description = [{
		The "store" operation
	}];

	let arguments = (ins F32MemRef:$src, I32Attr:$size);
}

#endif // DPIM_OPS
