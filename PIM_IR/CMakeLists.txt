cmake_minimum_required(VERSION 3.13.4)
project(pim-dialect LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(LLVM_ENABLE_PROJECTS mlir)
#set(LLVM_BUILD_EXAMPLES ON)
set(LLVM_TARGETS_TO_BUILD "X86;NVPTX;AMDGPU")
set(CMAKE_BUILD_TYPE Release)
#set(LLVM_ENABLE_ASSERTIONS ON)
#set(CMAKE_C_COMPILER clang)
#set(CMAKE_CXX_COMPILER clang++)
#set(LLVM_ENABLE_LLD ON)
#set(LLVM_DIR ~/llvm-project/build/lib/cmake/llvm)
#set(MLIR_DIR ~/llvm-project/build/lib/cmake/mlir)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
set(PIMLIBS
PIMOps
PNMOps
APIMOps
DPIMOps

PIMToPNM
PNMToLLVM

PIMToAPIM
APIMToLLVM

PIMToDPIM
DPIMToLLVM
)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
#add_subdirectory(test)
add_subdirectory(PIM-IR)
