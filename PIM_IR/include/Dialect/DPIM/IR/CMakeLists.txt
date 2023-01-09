set(LLVM_TARGET_DEFINITIONS DPIMOps.td)
mlir_tablegen(DPIMOps.hpp.inc -gen-op-decls "-I${PROJECT_SOURCE_DIR}/include")
mlir_tablegen(DPIMOps.cpp.inc -gen-op-defs "-I${PROJECT_SOURCE_DIR}/include")

set(GEN_DOC_FILE ${CMAKE_BINARY_DIR}/docs/Dialects/apim.md)
add_public_tablegen_target(DPIMOpsIncGen)

# Header dependancies target for DPIMOps.hpp
add_custom_target(DPIMOpsInc
        DEPENDS DPIMOpsIncGen)
