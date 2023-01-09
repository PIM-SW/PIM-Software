add_mlir_dialect_library(PNMToLLVM
	PNMToLLVM.cpp

	ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/
	${PROJECT_BINARY_DIR}/include

	DEPENDS
	PNMOpsInc

	LINK_LIBS PUBLIC
	MLIRIR
	#LLVMOps 
	PNMOps
	)
target_include_directories(PNMToLLVM
	PRIVATE
	${PROJECT_SOURCE_DIR}/
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/
	${PROJECT_BINARY_DIR}/include
	)

#Header dependancies
add_dependencies(PNMToLLVM
	PNMOpsInc
	)

# Linking dependancies
add_dependencies(PNMToLLVM 
	#	LLVMOps 
	PNMOps
	)
install(TARGETS PNMToLLVM DESTINATION lib)

