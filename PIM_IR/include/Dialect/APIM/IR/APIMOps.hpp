//===-------------------- APIMOps.hpp - APIM Ops Header ---------------------===//
//
//===-------------------------- corelab heelim ----------------------------===//
//
//===----------------------------------------------------------------------===//
#ifndef __APIM_OPS_H__
#define __APIM_OPS_H__

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

class APIMOpsDialect : public Dialect {
public:
    APIMOpsDialect(MLIRContext *context);
    static StringRef getDialectNamespace() { return "apim"; }
};
} // end of namespace mlir

#define GET_OP_CLASSES
#include "Dialect/APIM/IR/APIMOps.hpp.inc"

#endif

