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

def PIM_MAC_Op: Op<PIM_Dialect, "mac"> {
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

def PIM_ACC_Op: Op<PIM_Dialect, "acc"> {
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

/*
def PIM_PrintOp : Op<PIM_Dialect, "pimprint",[]> {
    let summary = "print operation";
    let description = [{
        The "print" builtin operation prints a given input tensor, and produces
        no results.
    }];
    
    // The print operation takes an input tensor to print.
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
    let assemblyFormat = "$input attr-dict `:` type($input)";
}
*/

def PIMConvertOp : Op<PIM_Dialect,"convert", [NoSideEffect]> {
  let summary = "convert operation";
  let description = [{
    The "convert" operation represents a convert operation within a function.
    Only for testing.
  }];

  let arguments = (ins F32:$input);
  let results = (outs F64Tensor:$res);

  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];
}

def PIMReturnOp : Op<PIM_Dialect, "return", [NoSideEffect, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      func @foo() -> tensor<2xf64> {
        ...
        toy.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<AnyTensor>:$input);
  //Variadic<F64Tensor>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];
}
#endif // PIM_OPS
