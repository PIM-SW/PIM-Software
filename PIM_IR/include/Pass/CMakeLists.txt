set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name pim "-I${PROJECT_SOURCE_DIR}/include")
add_public_tablegen_target(CallToPIMOpsIncGen)