// my_chip_compiler/tools/TargetSilicon/mychip-translate.cpp
// my_chip_compiler/tools/TargetSilicon/mychip-translate.cpp
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
void registerToHardwareTranslation();
}

int main(int argc, char **argv) {
    // Registers the command AND its structural dialect requirements
    mlir::registerToHardwareTranslation();
    
    return failed(mlir::mlirTranslateMain(argc, argv, "MyChip Translation Driver\n"));
}
