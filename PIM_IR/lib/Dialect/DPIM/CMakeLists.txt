
add_subdirectory(IR)

#add_mlir_dialect_library(DPIMOps
#	DPIMOps.cpp
#
#	ADDITIONAL_HEADER_DIRS
#	${PROJECT_SOURCE_DIR}/include
#
#	DEPENDS
#	DPIMOpsInc
#
#	LINK_LIBS PUBLIC
#	MLIRIR
#	)
#target_include_directories(DPIMOps
#	PRIVATE
#	${PROJECT_SOURCE_DIR}
#	${PROJECT_SOURCE_DIR}/include
#	${PROJECT_BINARY_DIR}
#	${PROJECT_BINARY_DIR}/include
#	)
#add_dependencies(DPIMOps DPIMOpsInc)
#install(TARGETS DPIMOps DESTINATION lib)

