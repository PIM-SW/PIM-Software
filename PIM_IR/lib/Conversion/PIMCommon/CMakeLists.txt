add_mlir_dialect_library(PIMToAPIM
	PIMToAPIM.cpp

	ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/
	${PROJECT_BINARY_DIR}/include

	DEPENDS
	APIMOpsInc
	PIMOpsInc

	LINK_LIBS PUBLIC
	MLIRIR
	APIMOps 
	PIMOps
	PIMAPIOps
	)
target_include_directories(PIMToAPIM
	PRIVATE
	${PROJECT_SOURCE_DIR}/
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/
	${PROJECT_BINARY_DIR}/include
	)

#Header dependancies
add_dependencies(PIMToAPIM
	PIMOpsInc
	PIMAPIOpsInc
	APIMOpsInc)

# Linking dependancies
add_dependencies(PIMToAPIM 
	APIMOps 
	PIMOps
	PIMAPIOps)
install(TARGETS PIMToAPIM DESTINATION lib)

