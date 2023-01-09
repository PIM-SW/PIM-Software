//===----------- PIMOps.td - PIM Dialect Operation Definitions ------------===//
//
//===-------------------------- corelab heelim ----------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PIM_OPS
#define PIM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
//===----------------------------------------------------------------------===//
// PIM dialect definition.
//===----------------------------------------------------------------------===//

def PIM_Dialect : Dialect {
	let name = "pim";
	let cppNamespace = "::mlir";
}

//===----------------------------------------------------------------------===//
// Base PIM operation definition.
//===----------------------------------------------------------------------===//

//class Op<PIM_Dialect, string mnemonic, list<OpTrait> traits = []> :
//        Op<PIM_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
//  PIM operations definition.
//===----------------------------------------------------------------------===//

def PIM_SIMD_ADD_Op: Op<PIM_Dialect, "simd_add"> {
	let summary = "simd add, size 16";
	let description = [{ 
		"pim.simd_add".
	}];
	let arguments = (ins 
			F32MemRef:$X,
			F32MemRef:$Y
			);
	let results = (outs F32MemRef:$res);
	let builders = [
		OpBuilder<(ins "Value":$X, "Value":$Y)>,
	];
}

def PIM_SIMD_SUB_Op: Op<PIM_Dialect, "simd_sub"> {
	let summary = "simd sub, size 16";
	let description = [{ 
		"pim.simd_sub".
	}];
	let arguments = (ins 
			F32MemRef:$X,
			F32MemRef:$Y
			);
	let results = (outs F32MemRef:$res);
	let builders = [
		OpBuilder<(ins "Value":$X, "Value":$Y)>,
	];
}

def PIM_SIMD_MUL_Op: Op<PIM_Dialect, "simd_mul"> {
	let summary = "simd mul, size 16";
	let description = [{ 
		"pim.simd_mul".
	}];
	let arguments = (ins 
			F32MemRef:$X,
			F32MemRef:$Y
			);
	let results = (outs F32MemRef:$res);
	let builders = [
		OpBuilder<(ins "Value":$X, "Value":$Y)>,
	];
}

def PIM_SIMD_DIV_Op: Op<PIM_Dialect, "simd_div"> {
	let summary = "simd div, size 16";
	let description = [{ 
		"pim.simd_div".
	}];
	let arguments = (ins 
			F32MemRef:$X,
			F32MemRef:$Y
			);
	let results = (outs F32MemRef:$res);
	let builders = [
		OpBuilder<(ins "Value":$X, "Value":$Y)>,
	];
}

def PIM_SIMD_SCAL_ADD_Op: Op<PIM_Dialect, "simd_scal_add"> {
	let summary = "simd add scalar value, size 16";
	let description = [{ 
		"pim.simd_".
	}];
	let arguments = (ins 
			F32:$X,
			F32MemRef:$Y
			);
	let results = (outs F32MemRef:$res);
	let builders = [
		OpBuilder<(ins "Value":$X, "Value":$Y)>,
	];
}

def PIM_SIMD_SCAL_SUB_Op: Op<PIM_Dialect, "simd_scal_sub"> {
	let summary = "simd substrate scalar value, size 16";
	let description = [{ 
		"pim.simd_scal_sub".
	}];
	let arguments = (ins 
			F32:$X,
			F32MemRef:$Y
			);
	let results = (outs F32MemRef:$res);
	let builders = [
		OpBuilder<(ins "Value":$X, "Value":$Y)>,
	];
}

def PIM_SIMD_SCAL_MUL_Op: Op<PIM_Dialect, "simd_scal_mul"> {
	let summary = "simd multiply scalar value, size 16";
	let description = [{ 
		"pim.simd_scal_mul".
	}];
	let arguments = (ins 
			F32:$X,
			F32MemRef:$Y
			);
	let results = (outs F32MemRef:$res);
	let builders = [
		OpBuilder<(ins "Value":$X, "Value":$Y)>,
	];
}

def PIM_SIMD_SCAL_DIV_Op: Op<PIM_Dialect, "simd_scal_div"> {
	let summary = "simd , size 16";
	let description = [{ 
		"pim.simd_scal_div".
	}];
	let arguments = (ins 
			F32:$X,
			F32MemRef:$Y
			);
	let results = (outs F32MemRef:$res);
	let builders = [
		OpBuilder<(ins "Value":$X, "Value":$Y)>,
	];
}

def PIM_MAC_Op: Op<PIM_Dialect, "simd_mac"> {
	let summary = "mac, size 16";
	let description = [{ 
		"pim.mac".
	}];
	let arguments = (ins 
			F32MemRef:$X,
			F32MemRef:$Y
			);
	let results = (outs F32:$res);
	let builders = [
		OpBuilder<(ins "Value":$X, "Value":$Y)>,
	];
}

def PIM_ACC_Op: Op<PIM_Dialect, "simd_acc"> {
	let summary = "accumulate , size 16";
	let description = [{ 
		"pim.acc".
	}];
	let arguments = (ins 
			F32MemRef:$X
			);
	let results = (outs F32:$res);
	let builders = [
		OpBuilder<(ins "Value":$X)>,
	];
}

def PIM_VEC_ACC_Op: Op<PIM_Dialect, "mat_acc"> {
	let summary = "vector accumulate";
	let description = [{ 
		"pim.mat_acc".
	}];
	let arguments = (ins 
			F32MemRef:$X,
			I32Attr:$width
			);
	let results = (outs F32:$res);
	let builders = [
		OpBuilder<(ins "Value":$X)>,
	];
}

def PIM_VEC_ADD_Op: Op<PIM_Dialect, "mat_add"> {
	let summary = "elementwise vector addition";
	let description = [{ 
	}];
	let arguments = (ins 
			F32MemRef:$X,
			F32MemRef:$Y,
			I32Attr:$width
			);
	let results = (outs F32:$res);
	let builders = [
		OpBuilder<(ins 
				"Value":$X,
				"Value":$Y,
				"IntegerAttr":$width
			  )>,
	];
}

def PIM_VEC_MUL_Op: Op<PIM_Dialect, "mat_mul"> {
	let summary = "elementwise vector addition";
	let description = [{ 
	}];
	let arguments = (ins 
			F32MemRef:$X,
			F32MemRef:$Y,
			I32Attr:$width
			);
	let results = (outs F32:$res);
	let builders = [
		OpBuilder<(ins 
				"Value":$X,
				"Value":$Y,
				"IntegerAttr":$width
			  )>,
	];
}

def PIM_VEC_IMM_ADD_Op: Op<PIM_Dialect, "mat_imm_add"> {
	let summary = "elementwise vector addition";
	let description = [{ 
	}];
	let arguments = (ins 
			F32:$imm,
			F32MemRef:$X,
			I32Attr:$width
			);
	let results = (outs F32:$res);
	let builders = [
		OpBuilder<(ins 
				"Value":$imm,
				"Value":$X,
				"IntegerAttr":$width
			  )>,
	];
}

def PIM_VEC_IMM_MUL_Op: Op<PIM_Dialect, "mat_imm_mul"> {
	let summary = "elementwise vector addition";
	let description = [{ 
	}];
	let arguments = (ins 
			F32:$imm,
			F32MemRef:$X,
			I32Attr:$width
			);
	let results = (outs F32:$res);
	let builders = [
		OpBuilder<(ins 
				"Value":$imm,
				"Value":$Y,
				"IntegerAttr":$width
			  )>,
	];
}

def PIM_MV_MUL_Op: Op<PIM_Dialect, "mat_mvmul"> {
	let summary = "matrix vector multiplication";
	let description = [{ 
	}];
	let arguments = (ins 
			F32MemRef:$X,
			F32MemRef:$A,
			F32MemRef:$Y,
			I32Attr:$N,
			I32Attr:$M
			);
	let results = (outs F32:$res);
	let builders = [
		OpBuilder<(ins 
				"Value":$X,
				"Value":$A,
				"Value":$Y,
				"IntegerAttr":$N,
				"IntegerAttr":$M
			  )>,
	];
}



#endif // PIM_OPS
