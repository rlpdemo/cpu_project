#!/usr/bin/env python3
"""
Streaming-CPU cosimulation track (MNIST MLP 784 -> 128 -> 64 -> 10).

Same MLIR compilation pipeline as `pipeline_run_cosim.py`
    PyTorch -> Torch-MLIR -> bufferize -> mychip-opt -> mychip-translate
    -> commands.txt
but the compiled firmware is executed on the **streaming CPU** model
(`StreamingCPUHardwareSimulator` / `StreamingCPUSimulator` from str.py)
instead of the plain systolic array.

The streaming model still executes each tile functionally (numerical parity is
asserted against PyTorch) and additionally reports streaming-architecture
metrics: latency, DRAM/SRAM traffic, energy breakdown and TOPS efficiency.
"""
import re
import subprocess
import torch
import torch_mlir
from torch_mlir.fx import export_and_import
from my_chip_cosim.streaming_cpu import (
    StreamingCPUHardwareSimulator,
    StreamingCPUSimulator,
    PrecisionMode,
)

# ==============================================================================
# Step 1: Define the 3-layer MLP for MNIST (no bias / activations)
# ==============================================================================
class MNISTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 128, bias=False)
        self.fc2 = torch.nn.Linear(128, 64,  bias=False)
        self.fc3 = torch.nn.Linear(64,  10,  bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


model = MNISTModel().eval()
example_input = torch.randn(1, 784)

# ==============================================================================
# Step 2: Export PyTorch Graph to Torch-MLIR (linalg-on-tensors)
# ==============================================================================
print("--- Compiling PyTorch Graph to Torch-MLIR (Linalg Dialect) ---")
mlir_module = export_and_import(model, example_input, output_type="linalg-on-tensors")
with open("math_graph.mlir", "w") as f:
    f.write(str(mlir_module))
print("Saved: math_graph.mlir")

# ==============================================================================
# Step 3: Normal lowering pipeline
#   math_graph.mlir
#     -> (one-shot-bufferize)   buffer_graph.mlir
#     -> (lower-to-mychip)      final_hardware_ir.mlir
#     -> (mychip-translate)     commands.txt
# ==============================================================================
LLVM_BIN   = "/home/us/llvm-mlir-install/bin"
MYCHIP_OPT = "./my_chip_compiler/build/bin/mychip-opt"
MYCHIP_TR  = "./my_chip_compiler/build/bin/mychip-translate"

print("--- Running Upstream Bufferization ---")
subprocess.run([
    f"{LLVM_BIN}/mlir-opt",
    "--pass-pipeline=builtin.module(empty-tensor-to-alloc-tensor,one-shot-bufferize)",
    "math_graph.mlir", "-o", "buffer_graph.mlir"
], check=True)

print("--- Lowering to Custom Target Dialect ---")
subprocess.run([MYCHIP_OPT, "--lower-to-mychip",
                "buffer_graph.mlir", "-o", "final_hardware_ir.mlir"], check=True)

print("--- Translating to ISA Hardware Commands ---")
subprocess.run([MYCHIP_TR, "--mlir-to-mychip-cmds",
                "final_hardware_ir.mlir", "-o", "commands.txt"], check=True)
print("Saved: commands.txt")

# ==============================================================================
# Step 4: PyTorch golden reference
# ==============================================================================
input_tensor = torch.randn(1, 784, dtype=torch.float32)
with torch.no_grad():
    golden_output = model(input_tensor)   # shape [1, 10]

# ==============================================================================
# Step 5: Execute firmware on the streaming CPU model
# ==============================================================================
print("\n--- Executing Firmware on Streaming CPU Model (str.py) ---")

sim = StreamingCPUHardwareSimulator(
    clock_freq_ghz=1.0,
    dram_bw_gbps=128.0,
    num_engines=2000,
    precision=PrecisionMode.INT8,
)

with open("commands.txt", "r") as f:
    cmds_text = f.read()

cmd_blocks = re.findall(
    r'# --- BEGIN MY_CHIP EXECUTION COMMAND ---\n(.*?)\n# --- END COMMAND ---',
    cmds_text, re.DOTALL
)
lhs_vars  = [re.search(r'REG_LHS_PTR:\s*(%\w+)', b).group(1) for b in cmd_blocks
             if re.search(r'REG_LHS_PTR:\s*(%\w+)', b)]
rhs_vars  = [re.search(r'REG_RHS_PTR:\s*(%\w+)', b).group(1) for b in cmd_blocks
             if re.search(r'REG_RHS_PTR:\s*(%\w+)', b)]
dest_vars = [re.search(r'REG_DEST_PTR:\s*(%\w+)', b).group(1) for b in cmd_blocks
             if re.search(r'REG_DEST_PTR:\s*(%\w+)', b)]

sim.bind_buffer(lhs_vars[0], input_tensor)
sim.bind_buffer(rhs_vars[0], model.fc1.weight.t().contiguous())
sim.bind_buffer(rhs_vars[1], model.fc2.weight.t().contiguous())
sim.bind_buffer(rhs_vars[2], model.fc3.weight.t().contiguous())

sim.execute_commands("commands.txt")
sim_output = sim.get_buffer(dest_vars[-1])

# ==============================================================================
# Step 6: Numerical parity
# ==============================================================================
try:
    torch.testing.assert_close(sim_output, golden_output, rtol=1e-4, atol=1e-4)
    print("\n🟢 VERIFICATION SUCCESSFUL: PyTorch Golden output and Streaming CPU "
          "execution are numerically equivalent!")
except AssertionError as e:
    print("\n🔴 VERIFICATION FAILED: Numerical divergence between PyTorch and "
          "streaming CPU execution.")
    print(e)

# ==============================================================================
# Step 7: Streaming-architecture metrics report
# ==============================================================================
print("\n=== Streaming CPU Architecture Metrics (accumulated over firmware) ===")
metrics = sim.report()
print(f"Tiles executed        : {metrics['tiles_executed']}")
print(f"Total MACs            : {metrics['total_macs']}")
print(f"Simulated latency     : {metrics['simulated_latency_us']} us")
print(f"DRAM transferred      : {metrics['dram_transferred_bytes']} bytes")
print(f"SRAM transferred      : {metrics['sram_transferred_bytes']} bytes")
energy = metrics["energy_breakdown_uJ"]
print(f"Energy (uJ)           : dram_fetch={energy['dram_fetch']}, "
      f"sram_buffer={energy['sram_buffer']}, routing={energy['direct_wire_routing']}, "
      f"compute={energy['engine_compute']}, total={energy['total']}")
print(f"Throughput            : {metrics['throughput_tops']} TOPS")
print(f"Power                 : {metrics['power_watts']} W")
print(f"Efficiency            : {metrics['efficiency_tops_per_watt']} TOPS/W")
print(f"Precision mode        : {metrics['precision_mode']}")
print("\nPer-tile geometry:")
for tile in metrics["per_tile"]:
    print(f"  tile {tile['tile']}: M={tile['M']}, K={tile['K']}, N={tile['N']} | "
          f"MACs={tile['macs']} | latency={tile['latency_us']} us | energy={tile['energy_total_uJ']} uJ")