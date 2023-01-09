add_mlir_dialect_library(APIMToLLVM
	APIMToLLVM.cpp

	ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/
	${PROJECT_BINARY_DIR}/include

	DEPENDS
	APIMOpsInc

	LINK_LIBS PUBLIC
	MLIRIR
	#LLVMOps 
	APIMOps
	)
target_include_directories(APIMToLLVM
	PRIVATE
	${PROJECT_SOURCE_DIR}/
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/
	${PROJECT_BINARY_DIR}/include
	)

#Header dependancies
add_dependencies(APIMToLLVM
	APIMOpsInc
	)

# Linking dependancies
add_dependencies(APIMToLLVM 
	#	LLVMOps 
	APIMOps
	)
install(TARGETS APIMToLLVM DESTINATION lib)

