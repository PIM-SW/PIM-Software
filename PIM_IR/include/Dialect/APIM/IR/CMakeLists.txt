set(LLVM_TARGET_DEFINITIONS APIMOps.td)
mlir_tablegen(APIMOps.hpp.inc -gen-op-decls "-I${PROJECT_SOURCE_DIR}/include")
mlir_tablegen(APIMOps.cpp.inc -gen-op-defs "-I${PROJECT_SOURCE_DIR}/include")

set(GEN_DOC_FILE ${CMAKE_BINARY_DIR}/docs/Dialects/apim.md)
add_public_tablegen_target(APIMOpsIncGen)

# Header dependancies target for APIMOps.hpp
add_custom_target(APIMOpsInc
        DEPENDS APIMOpsIncGen)
