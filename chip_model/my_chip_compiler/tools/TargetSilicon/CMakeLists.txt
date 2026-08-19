# my_chip_compiler/tools/TargetSilicon/CMakeLists.txt

add_mlir_tool(mychip-translate
  mychip-translate.cpp
)

target_link_libraries(mychip-translate
  PRIVATE
  # Use the explicit target definition so CMake resolves the absolute path
  MLIRTranslateLib
  MLIRMyChipTargetSilicon
  MLIRMyChip
  MLIRIR
  MLIRSupport
  MLIRBufferizationDialect
)
