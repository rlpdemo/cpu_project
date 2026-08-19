#ifndef MY_CHIP_DIALECT
#define MY_CHIP_DIALECT

include "mlir/IR/OpBase.td"

// Changed "define" to "def"
def MyChip_Dialect : Dialect {
    let name = "my_chip";
    let summary = "A software-compliant dialect for 80x80 MMC execution architectures.";
    let description = [{
        This dialect implements hardware-specific instructions for mapping
        tiled linear algebra operators directly onto our physical ASIC compute clusters.
    }];
    let cppNamespace = "::mlir::my_chip";
}

class MyChip_Op<string mnemonic, list<Trait> traits = []> :
    Op<MyChip_Dialect, mnemonic, traits>;

#endif // MY_CHIP_DIALECT

