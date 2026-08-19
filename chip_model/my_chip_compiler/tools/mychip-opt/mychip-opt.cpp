// tools/mychip-opt/mychip-opt.cpp
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "MyChip/MyChipDialect.h"

namespace mlir {
namespace my_chip {
    std::unique_ptr<Pass> createLowerToMyChipPass();
}
}

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    registry.insert<mlir::my_chip::MyChipDialect>();

    mlir::PassPipelineRegistration<>(
        "lower-to-mychip",
        "Lowers linalg matmuls to my_chip hardware matrix tiles.",
        [](mlir::OpPassManager &pm) {
            pm.addPass(mlir::my_chip::createLowerToMyChipPass());
        }
    );

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "MyChip Optimizer Driver\n", registry)
    );
}
