add_mlir_dialect_library(PIMToAPIM
	PIMToAPIM.cpp

	ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/
	${PROJECT_BINARY_DIR}/include

	DEPENDS
	PIMOpsInc
	APIMOpsInc

	LINK_LIBS PUBLIC
	MLIRIR
	MLIRArithmetic
	APIMOps 
	PIMOps
	)
target_include_directories(PIMToAPIM
	PRIVATE
	${PROJECT_SOURCE_DIR}/
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/
	${PROJECT_BINARY_DIR}/include
	)

#Header dependancies
add_dependencies(
	PIMToAPIM
	PIMOpsInc
	APIMOpsInc
	)

# Linking dependancies
add_dependencies(
	PIMToAPIM 
	APIMOps 
	PIMOps
	)
install(TARGETS PIMToAPIM DESTINATION lib)

