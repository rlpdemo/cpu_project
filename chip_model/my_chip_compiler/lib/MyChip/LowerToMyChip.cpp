#include "MyChip/MyChipDialect.h"
#include "MyChip/MyChipOps.h"
#include "MyChip/MyChipPasses.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct MatMulLoweringPattern : public OpConversionPattern<linalg::MatmulOp> { // <-- Fixed: lowercase m
    using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;        // <-- Fixed: lowercase m
LogicalResult matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    
    // Use the adaptor values to ensure you grab the newly transformed 
    // buffer/memref types rather than the original un-bufferized operands.
    Value lhs = adaptor.getInputs()[0];
    Value rhs = adaptor.getInputs()[1];
    Value outputBuffer = adaptor.getOutputs()[0];

    // 1. Emit your physical memory-mapped hardware instruction op
    rewriter.create<my_chip::MatMulOp>(loc, lhs, rhs, outputBuffer);

    // 2. Notify the conversion framework that this 0-result op is cleanly replaced
    rewriter.replaceOp(op, ValueRange()); 
    return success();
}

};

struct LowerToMyChipPass : public PassWrapper<LowerToMyChipPass, OperationPass<ModuleOp>> {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToMyChipPass)

        llvm::StringRef getArgument() const override { return "lower-to-mychip"; }
        llvm::StringRef getDescription() const override { return "Lowers linalg matmuls to my_chip hardware matrix tiles."; }

        // ====== ADD THIS METHOD RIGHT HERE ======
        void getDependentDialects(mlir::DialectRegistry &registry) const override {
            registry.insert<my_chip::MyChipDialect>();
        }
        // =======================================

        void runOnOperation() override {
            ConversionTarget target(getContext());
            RewritePatternSet patterns(&getContext());

            target.addLegalDialect<my_chip::MyChipDialect>();
            target.addIllegalOp<linalg::MatmulOp>();

            patterns.add<MatMulLoweringPattern>(&getContext());

            if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
                signalPassFailure();
            }
        }
    };
} // namespace


namespace mlir {
namespace my_chip {
    std::unique_ptr<Pass> createLowerToMyChipPass() {
        return std::make_unique<LowerToMyChipPass>();
    }
} // namespace my_chip
} // namespace mlir
