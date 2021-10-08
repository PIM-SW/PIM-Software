get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
	${dialect_libs}
	${conversion_libs}
	MLIROptLib
	PIMOps
	PNMOps
	PIMToPNM
	)
add_executable(PIM-IR 
	PIM-IR.cpp
	)
add_dependencies(PIM-IR 
	PIMOpsInc
	PNMOpsInc
	)
add_dependencies(PIM-IR 
	PIMOps
	PNMOps
	PIMToPNM
	)
target_include_directories(PIM-IR
	PRIVATE
	${PROJECT_SOURCE_DIR}
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}
	${PROJECT_BINARY_DIR}/include
	)
llvm_update_compile_flags(PIM-IR)
target_link_libraries(PIM-IR 
	PRIVATE 
	MLIRIR
	${LIBS} 
	${PIMLIBS}
	)

mlir_check_all_link_libraries(PIM-IR)
