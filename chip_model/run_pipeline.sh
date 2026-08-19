#!/bin/bash

# Exit immediately if any command fails
set -e

# Define absolute paths based on your environment
WORKSPACE_DIR="$HOME/Desktop/ws/pytorch_model"
LLVM_INSTALL_BIN="/home/us/llvm-mlir-install/bin"
BUILD_DIR="$WORKSPACE_DIR/my_chip_compiler/build"
CUSTOM_OPT="$BUILD_DIR/bin/mychip-opt"

echo "========================================"
echo "1. Building Custom MyChip Compiler..."
echo "========================================"
cd "$BUILD_DIR"
ninja

echo -e "\n========================================"
echo "2. Running Upstream Bufferization..."
echo "========================================"
cd "$WORKSPACE_DIR"
./pipeline_run.py

$LLVM_INSTALL_BIN/mlir-opt \
  --pass-pipeline="builtin.module(empty-tensor-to-alloc-tensor,one-shot-bufferize)" \
  math_graph.mlir \
  -o buffer_graph.mlir

echo "Generated: buffer_graph.mlir (Tensors -> MemRefs)"

echo -e "\n========================================"
echo "3. Lowering to Custom Target Dialect..."
echo "========================================"
$CUSTOM_OPT \
  --lower-to-mychip \
  buffer_graph.mlir \
  -o final_hardware_ir.mlir

echo "Generated: final_hardware_ir.mlir (Target Dialect Complete)"
echo "========================================"
echo "Pipeline execution successful!"
