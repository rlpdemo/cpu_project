# tools/mychip-opt/CMakeLists.txt

add_llvm_executable(mychip-opt
  mychip-opt.cpp
)

target_link_libraries(mychip-opt
  PRIVATE
    MLIRMyChip
    
    # Core initialization and registration support
    MLIRRegisterAllDialects
    MLIROptLib
    MLIRParser
    MLIRSupport
    MLIRIR
    
    # The actual structural dialects checked by registerAllDialects
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRLLVMDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRVectorDialect
)
