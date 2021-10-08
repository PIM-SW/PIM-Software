//===----------- PNMOps.td - PNM Dialect Operation Definitions ------------===//
//
//===-------------------------- corelab heelim ----------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PNM_OPS
#define PNM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// PNM dialect definition.
//===----------------------------------------------------------------------===//

def PNM_Dialect : Dialect {
    let name = "pnm";
    let cppNamespace = "::mlir";
}

//===----------------------------------------------------------------------===//
//  PNM operations definition.
//===----------------------------------------------------------------------===//

def PNM_VectorOp : Op<PNM_Dialect, "vector"> {
  let summary = "vector";
  let description = [{ 
  "pnm.vector".
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

def PNM_VectorImmOp : Op<PNM_Dialect, "vector_imm"> {
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
    OpBuilder<(ins "Value":$X, "Value":$imm, "IntegerAttr":$width, "IntegerAttr":$type)>
  ];
}

def PNM_SetOp : Op<PNM_Dialect, "set", []> {
  let summary = "set immediate value operation";
  let description = [{
    The "set" operation, set immediate value to register
  }];

  let arguments = (ins F32:$input);
  let results = (outs F32:$output);
  let builders = [
    OpBuilder<(ins "Value":$input)>
	];
}

def PNM_CopyOp : Op<PNM_Dialect, "copy", []> {
  let summary = "copy operation";
  let description = [{
    The "copy" operation
  }];

  let arguments = (ins F32MemRef:$src, F32MemRef:$dst);
}

#endif // PNM_OPS
