//===-------------------- DPIMOps.hpp - DPIM Ops Header ---------------------===//
//
//===-------------------------- corelab heelim ----------------------------===//
//
//===----------------------------------------------------------------------===//
#ifndef __DPIM_OPS_H__
#define __DPIM_OPS_H__

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

class DPIMOpsDialect : public Dialect {
public:
    DPIMOpsDialect(MLIRContext *context);
    static StringRef getDialectNamespace() { return "dpim"; }
};
} // end of namespace mlir

#define GET_OP_CLASSES
#include "Dialect/DPIM/IR/DPIMOps.hpp.inc"

#endif

