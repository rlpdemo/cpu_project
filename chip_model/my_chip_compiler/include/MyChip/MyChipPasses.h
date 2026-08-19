#ifndef MY_CHIP_PASSES_H
#define MY_CHIP_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp; // Forward declaration
namespace my_chip {

std::unique_ptr<Pass> createLowerToMyChipPass();

} // namespace my_chip
} // namespace mlir

#endif // MY_CHIP_PASSES_H
