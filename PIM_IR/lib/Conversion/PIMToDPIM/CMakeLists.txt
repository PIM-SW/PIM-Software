add_mlir_dialect_library(PIMToDPIM
	PIMToDPIM.cpp

	ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/
	${PROJECT_BINARY_DIR}/include

	DEPENDS
	PIMOpsInc
	DPIMOpsInc

	LINK_LIBS PUBLIC
	MLIRIR
	MLIRArithmetic
	DPIMOps 
	PIMOps
	)
target_include_directories(PIMToDPIM
	PRIVATE
	${PROJECT_SOURCE_DIR}/
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/
	${PROJECT_BINARY_DIR}/include
	)

#Header dependancies
add_dependencies(
	PIMToDPIM
	PIMOpsInc
	DPIMOpsInc
	)

# Linking dependancies
add_dependencies(
	PIMToDPIM 
	DPIMOps 
	PIMOps
	)
install(TARGETS PIMToDPIM DESTINATION lib)

