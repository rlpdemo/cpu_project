# my_chip_compiler/lib/TargetSilicon/CMakeLists.txt

add_mlir_library(MLIRMyChipTargetSilicon
  EmitHardwareCmds.cpp

  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRSupport
  MLIRMyChip  # Links against your core dialect definitions
)
