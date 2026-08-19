// include/MyChip/MyChipOps.h
#ifndef MYCHIP_MYCHIPOPS_H
#define MYCHIP_MYCHIPOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"

// Define the namespace and pull in the ops declarations
#define GET_OP_CLASSES
#include "MyChip/MyChipOps.h.inc"

#endif // MYCHIP_MYCHIPOPS_H
