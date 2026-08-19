#include "MyChip/MyChipDialect.h"
#include "MyChip/MyChipOps.h"
#include "mlir/IR/BuiltinOps.h"          // Fixes ModuleOp declaration
#include "mlir/IR/Operation.h"           // Fixes Operation type checking
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h" // <--- ADD THIS HEADER

using namespace mlir;

// Change input type to standard Operation* for the macro interface
LogicalResult emitHardwareCommands(Operation *op, llvm::raw_ostream &os) {
    auto module = dyn_cast<ModuleOp>(op);
    if (!module) {
        return failure();
    }

    // Walk through every operation inside the MLIR module and emit a hardware command.
    module.walk([&](my_chip::MatMulOp matmulOp) {
        os << "# --- BEGIN MY_CHIP EXECUTION COMMAND ---\n";
        os << "CMD_TYPE: EXEC_TILE_MATMUL\n";
        os << "REG_LHS_PTR:  " << matmulOp.getLhs() << "\n";
        os << "REG_RHS_PTR:  " << matmulOp.getRhs() << "\n";
        os << "REG_DEST_PTR: " << matmulOp.getOutput() << "\n";
        os << "# --- END COMMAND ---\n\n";
    });
    return success();
}

// ... keep your emitHardwareCommands function as it is ...
namespace mlir {
void registerToHardwareTranslation() {
    TranslateFromMLIRRegistration(
        "mlir-to-mychip-cmds",
        "Translates MyChip dialect operations into physical hardware command structures",
        emitHardwareCommands,
        [](DialectRegistry &registry) {
            // Register your custom dialect AND standard upstream components here!
            registry.insert<mlir::my_chip::MyChipDialect,
                            mlir::func::FuncDialect,
                            mlir::memref::MemRefDialect,
                            mlir::arith::ArithDialect,
                            mlir::linalg::LinalgDialect,
                            mlir::bufferization::BufferizationDialect>();
        }
    );
}
} // namespace mlir
