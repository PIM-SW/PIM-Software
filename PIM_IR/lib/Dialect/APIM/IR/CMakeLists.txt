add_mlir_dialect_library(APIMOps
	APIMOps.cpp

	ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/include

	DEPENDS
	APIMOpsInc

	LINK_LIBS PUBLIC
	MLIRIR
	)
target_include_directories(APIMOps
	PRIVATE
	${PROJECT_SOURCE_DIR}
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}
	${PROJECT_BINARY_DIR}/include
	)
add_dependencies(APIMOps APIMOpsInc)
install(TARGETS APIMOps DESTINATION lib)

