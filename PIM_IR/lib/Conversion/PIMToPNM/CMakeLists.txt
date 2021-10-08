add_mlir_dialect_library(PIMToPNM
	PIMToPNM.cpp

	ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/
	${PROJECT_BINARY_DIR}/include

	DEPENDS
	PIMOpsInc
	PNMOpsInc

	LINK_LIBS PUBLIC
	MLIRIR
	PNMOps 
	PIMOps
	)
target_include_directories(PIMToPNM
	PRIVATE
	${PROJECT_SOURCE_DIR}/
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/
	${PROJECT_BINARY_DIR}/include
	)

#Header dependancies
add_dependencies(PIMToPNM
	PIMOpsInc
	PNMOpsInc
	)

# Linking dependancies
add_dependencies(PIMToPNM 
	PNMOps 
	PIMOps
	)
install(TARGETS PIMToPNM DESTINATION lib)

