# lib/MyChip/CMakeLists.txt

set(MYCHIP_TBLGEN_INCLUDES
  "-I${MLIR_INCLUDE_DIRS}"
  "-I${LLVM_INCLUDE_DIRS}"
  "-I${PROJECT_SOURCE_DIR}/include"
)

# Define the target definitions file
set(LLVM_TARGET_DEFINITIONS ../../include/MyChip/MyChipDialect.td)

# Output directly into include/MyChip/ so they match your C++ include directives
mlir_tablegen(../../include/MyChip/MyChipDialect.h.inc -gen-dialect-decls ${MYCHIP_TBLGEN_INCLUDES})
mlir_tablegen(../../include/MyChip/MyChipDialect.cpp.inc -gen-dialect-defs ${MYCHIP_TBLGEN_INCLUDES})
add_public_tablegen_target(MyChipDialectIncGen)

set(LLVM_TARGET_DEFINITIONS ../../include/MyChip/MyChipOps.td)
mlir_tablegen(../../include/MyChip/MyChipOps.h.inc -gen-op-decls ${MYCHIP_TBLGEN_INCLUDES})
mlir_tablegen(../../include/MyChip/MyChipOps.cpp.inc -gen-op-defs ${MYCHIP_TBLGEN_INCLUDES})
add_public_tablegen_target(MyChipOpsIncGen)

add_mlir_dialect_library(MLIRMyChip
  MyChipDialect.cpp
  LowerToMyChip.cpp

  DEPENDS
  MyChipDialectIncGen
  MyChipOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRTransforms
  MLIRPass
)

