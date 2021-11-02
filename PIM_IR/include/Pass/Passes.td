include "mlir/Pass/PassBase.td"

def CallToPIMOps : FunctionPass<"pim-avail"> {
  let summary = "replace call to PIM ops";
  let constructor = "mlir::replaceCallToPIMOps()";
  //let dependentDialects = [""];
}