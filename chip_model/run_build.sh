# 1. Unpack the compiler structure
tar -xvf my_chip_compiler.tar.gz
cd my_chip_compiler

# 2. Setup standard build environment
mkdir build && cd build

# 3. Configure CMake pointing directly to your installation prefix
cmake -G Ninja .. \
  -DMLIR_DIR=/home/us/llvm-mlir-install/lib/cmake/mlir \
  -DLLVM_DIR=/home/us/llvm-mlir-install/lib/cmake/llvm

# 4. Compile your driver binary
ninja mychip-opt
