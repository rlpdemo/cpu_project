# my_chip_compiler/lib/CMakeLists.txt

add_mlir_dialect_library(MLIRMyChip
  MyChip/MyChipDialect.cpp
  MyChip/MyChipOps.cpp
  MyChip/LowerToMyChip.cpp
  
  # ====== ADD YOUR NEW EMITTER FILE HERE ======
  Target/EmitHardwareCmds.cpp

  DEPENDS
  MLIRMyChipOpsIncGen

  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRLinalgDialect
)
