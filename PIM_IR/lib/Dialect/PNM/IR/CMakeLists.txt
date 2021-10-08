add_mlir_dialect_library(PNMOps
	PNMOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include

        DEPENDS
				PNMOpsInc

	LINK_LIBS PUBLIC
	MLIRIR
	)
target_include_directories(PNMOps
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/include
		)
add_dependencies(PNMOps PNMOpsInc)
install(TARGETS PNMOps DESTINATION lib)

