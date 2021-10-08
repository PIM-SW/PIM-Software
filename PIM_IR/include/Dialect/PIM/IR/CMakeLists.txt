set(LLVM_TARGET_DEFINITIONS PIMOps.td)
#flexc_tablegen(PIMOps.hpp.inc -gen-op-decls "-I${PROJECT_SOURCE_DIR}/src/include")
#flexc_tablegen(PIMOps.cpp.inc -gen-op-defs "-I${PROJECT_SOURCE_DIR}/src/include")
mlir_tablegen(PIMOps.hpp.inc -gen-op-decls "-I${PROJECT_SOURCE_DIR}/include")
mlir_tablegen(PIMOps.cpp.inc -gen-op-defs "-I${PROJECT_SOURCE_DIR}/include")

set(GEN_DOC_FILE ${CMAKE_BINARY_DIR}/docs/Dialects/pim.md)
add_public_tablegen_target(PIMOpsIncGen)

# Header dependancies target for PIMOps.hpp
add_custom_target(PIMOpsInc
        DEPENDS PIMOpsIncGen)

#add_flexc_dialect_doc(cudnn CUDNNOps.td)
#add_mlir_doc(pim PIMOps.td -gen-op-doc)


#add_mlir_dialect(PIMOps pim)
#add_mlir_doc(PIMDialect PIMDialect PIM/ -gen-dialect-doc)
#add_mlir_doc(PIMOps PIMOps PIM/ )
