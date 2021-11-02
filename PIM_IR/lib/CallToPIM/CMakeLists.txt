add_mlir_dialect_library(CallToPIM
	CallToPIM.cpp

	ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/
	${PROJECT_BINARY_DIR}/include

	DEPENDS
	PIMOpsInc

	LINK_LIBS PUBLIC
	MLIRIR
	PIMOps
	)

target_include_directories(CallToPIM
	PRIVATE
	${PROJECT_SOURCE_DIR}/
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/
	${PROJECT_BINARY_DIR}/include
	)

#Header dependancies
add_dependencies(CallToPIM
	PIMOpsInc
	)

# Linking dependancies
add_dependencies(CallToPIM 
	PIMOps
	)
install(TARGETS CallToPIM DESTINATION lib)