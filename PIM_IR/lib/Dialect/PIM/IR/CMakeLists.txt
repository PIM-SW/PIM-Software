#add_library(PIMOps
#    PIMOps.cpp
#	PUBLIC
#	MLIRIR
#	)
#
add_mlir_dialect_library(PIMOps
	PIMOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include

        DEPENDS
				PIMOpsInc

	LINK_LIBS PUBLIC
	MLIRIR
	)
target_include_directories(PIMOps
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/include
		)
add_dependencies(PIMOps PIMOpsInc)
install(TARGETS PIMOps DESTINATION lib)

