#ifndef MY_CHIP_OPS
#define MY_CHIP_OPS

include "MyChip/MyChipDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Changed "define" to "def"
def MyChip_MatMulOp : MyChip_Op<"matmul", []> {
    let summary = "Hardware matrix multiplication tile operation.";
    
    // Accepts three physical memory address spaces (LHS, RHS, and Dest)
    let arguments = (ins 
        Arg<AnyMemRef, "LHS matrix buffer">:$lhs,
        Arg<AnyMemRef, "RHS matrix buffer">:$rhs,
        Arg<AnyMemRef, "Destination accumulation buffer">:$output
    );

    // ZERO result outputs—it mutates the destination buffer in-place
    let results = (outs); 
}

// Changed "define" to "def"
def MyChip_VectorFusedActivationOp : MyChip_Op<"vector_fused_act", [Pure]> {
    let summary = "Streams an SRAM tile through TVU for inline activation (SiLU/GeLU).";
    let arguments = (ins Arg<AnyMemRef, "Accumulator tile block", [MemRead]>:$input,
                         StrAttr:$activation_type);
    let results = (outs AnyMemRef:$output);
    
    let assemblyFormat = "$input `,` $activation_type attr-dict `:` type($input) `->` type($output)";
}

#endif // MY_CHIP_OPS
