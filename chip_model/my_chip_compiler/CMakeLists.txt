cmake_minimum_required(VERSION 3.20)
project(my_chip_compiler LANGUAGES CXX C)

# 1. Find the packages
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
# Top-level CMakeLists.txt
find_package(MLIR REQUIRED COMPONENTS
  MLIROptLib
  MLIRAllDialects # <-- Ensure this component is explicitly requested here
)
message(STATUS "Using LLVMConfig.cmake from: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake from: ${MLIR_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

# 2. Append their CMake module directories to search path
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

# 3. Include the helper macros
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# === THE C++ INCLUDE FIX: Tell the C++ compiler where your headers live ===
include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include        # Finds: include/MyChip/MyChipDialect.h
  ${PROJECT_BINARY_DIR}/include        # Finds: build/include/ (where TableGen places files)
  ${PROJECT_BINARY_DIR}/lib/MyChip     # Finds generated .inc files
)
# =========================================================================

# Global TableGen flag settings (which we established in our previous step)
set(MLIR_TABLEGEN_FLAGS 
  "-I${MLIR_INCLUDE_DIRS}"
  "-I${LLVM_INCLUDE_DIRS}"
  "-I${PROJECT_SOURCE_DIR}/include"
)

# Add our custom subdirectories -must have CmakeLists.txt
add_subdirectory(lib/MyChip)
add_subdirectory(tools/mychip-opt)

# ==============================================================================
# Add Subdirectories for Compilation -must have CmakeLists.txt
# ==============================================================================

# 1. Include your library logic (make sure your new emitter library folder is added)
add_subdirectory(lib/TargetSilicon)

# 2. Add this line to include the tools folder where your executable lives
add_subdirectory(tools/TargetSilicon)
