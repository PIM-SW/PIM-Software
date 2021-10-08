set(LLVM_TARGET_DEFINITIONS PNMOps.td)
mlir_tablegen(PNMOps.hpp.inc -gen-op-decls "-I${PROJECT_SOURCE_DIR}/include")
mlir_tablegen(PNMOps.cpp.inc -gen-op-defs "-I${PROJECT_SOURCE_DIR}/include")

set(GEN_DOC_FILE ${CMAKE_BINARY_DIR}/docs/Dialects/pnm.md)
add_public_tablegen_target(PNMOpsIncGen)

# Header dependancies target for APIMOps.hpp
add_custom_target(PNMOpsInc
	DEPENDS PNMOpsIncGen)
