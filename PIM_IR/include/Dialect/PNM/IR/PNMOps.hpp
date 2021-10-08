//===-------------------- PNMOps.hpp - PNM Ops Header ---------------------===//
//
//===-------------------------- corelab heelim ----------------------------===//
//
//===----------------------------------------------------------------------===//
#ifndef __PNM_OPS_H__
#define __PNM_OPS_H__

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Support/TypeID.h"

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {

class PNMOpsDialect : public Dialect {
public:
    PNMOpsDialect(MLIRContext *context);
    static StringRef getDialectNamespace() { return "apim"; }
};
} // end of namespace mlir

#define GET_OP_CLASSES
#include "Dialect/PNM/IR/PNMOps.hpp.inc"

#endif

