add_mlir_dialect_library(DPIMToLLVM
	DPIMToLLVM.cpp

	ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/
	${PROJECT_BINARY_DIR}/include

	DEPENDS
	DPIMOpsInc

	LINK_LIBS PUBLIC
	MLIRIR
	#LLVMOps 
	DPIMOps
	)
target_include_directories(DPIMToLLVM
	PRIVATE
	${PROJECT_SOURCE_DIR}/
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/
	${PROJECT_BINARY_DIR}/include
	)

#Header dependancies
add_dependencies(DPIMToLLVM
	DPIMOpsInc
	)

# Linking dependancies
add_dependencies(DPIMToLLVM 
	#	LLVMOps 
	DPIMOps
	)
install(TARGETS DPIMToLLVM DESTINATION lib)

