//===-------------------- PIMOps.hpp - PIM Ops Header ---------------------===//
//
//===-------------------------- corelab heelim ----------------------------===//
//
//===----------------------------------------------------------------------===//
#ifndef __PIM_OPS_H__
#define __PIM_OPS_H__

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

//#include "flexc/Dialect/CUDNN/IR/CUDNNOpsHelper.hpp"

namespace mlir {

class PIMOpsDialect : public Dialect {
public:
    PIMOpsDialect(MLIRContext *context);
    static StringRef getDialectNamespace() { return "pim"; }
};
} // end of namespace mlir

#define GET_OP_CLASSES
#include "Dialect/PIM/IR/PIMOps.hpp.inc"

#endif

