#include "MyChip/MyChipDialect.h"
#include "MyChip/MyChipOps.h"

// === ADD THESE THREE BUILDER INCLUDES HERE ===
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpImplementation.h"
// ============================================

using namespace mlir;
using namespace mlir::my_chip;

#include "MyChip/MyChipDialect.cpp.inc"

#define GET_OP_CLASSES
#include "MyChip/MyChipOps.cpp.inc"

void MyChipDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MyChip/MyChipOps.cpp.inc"
  >();
}
