# Manual Verification Cheatsheet — Streaming CPU

All commands assume you're in the project root:
`/home/us/Desktop/ws/cpu_project/chip_model`

## 0. Activate the environment

The venv with torch / torch-mlir is **not** the system Python.

```bash
source /home/us/Desktop/mlir_venv/bin/activate
cd /home/us/Desktop/ws/cpu_project/chip_model
```

## 1. Track 1 — streaming CPU model on the compiled firmware

Runs the MLIR pipeline (PyTorch → Torch-MLIR → bufferize → `mychip-opt` →
`mychip-translate` → `commands.txt`) and executes the resulting tile commands
on the stationary-weight **streaming CPU** model (`StreamingCPUHardwareSimulator`
in `my_chip_cosim/streaming_cpu.py`). Every tile executes functionally (golden
parity is asserted) and it reports streaming-architecture metrics: tiles,
MACs, latency, DRAM/SRAM traffic, energy breakdown and TOPS efficiency.

```bash
python pipeline_run_streaming.py
```

Expect:
`🟢 VERIFICATION SUCCESSFUL: PyTorch Golden output and Streaming CPU execution
are numerically equivalent!`
followed by a streaming-CPU metrics report (tiles, latency, energy, TOPS/W).

Intermediate artifacts are regenerated in the project root: `math_graph.mlir`,
`buffer_graph.mlir`, `final_hardware_ir.mlir`, `commands.txt`.

## 2. Track 2 — streaming-CPU RTL equivalence (4-way)

End-to-end 4-way check for the streaming-CPU MLP (784→128→64→10): one
seed-0 model, one input, and four representations that must all agree —

  1. PyTorch golden            (float32)
  2. Compiler model            (MLIR pipeline → commands.txt → MyChipHardwareSimulator)
  3. Streaming-CPU model       (StreamingCPUHardwareSimulator, same core + metrics)
  4. Generated Verilog RTL     (vsim.py interpreter always; iverilog when found)

```bash
cd verilog
python run_streaming_equivalence.py   # compiles MLIR, regenerates .v/.tb, simulates
```

Expect: `🟢 STREAMING-CPU EQUIVALENCE VERIFIED: PyTorch, compiler model,
streaming model, and Verilog RTL all agree.` plus a per-pair worst-|diff|
table (RTL-vs-reference ~1e-12, reference-vs-torch ~1e-7) and the streaming
metrics (3 tiles, 109184 MACs, latency, energy).

`vsim.py` is a pure-Python Verilog-subset interpreter (no external deps), so
the RTL check works even without iverilog; iverilog is auto-detected on PATH or
at `~/.local/opt/iverilog/root/usr/bin` and cross-checks the interpreter.

To regenerate the streaming RTL/testbench by hand:

```bash
python streaming_cpu_model.py         # writes streaming_cpu_top.v / _tb.v
python vsim.py tb_streaming_cpu streaming_cpu_top.v streaming_cpu_top_tb.v
```

To drive iverilog by hand:

```bash
PATH=~/.local/opt/iverilog/root/usr/bin:$PATH iverilog -g2012 -o /tmp/sim \
  streaming_cpu_top.v streaming_cpu_top_tb.v
PATH=~/.local/opt/iverilog/root/usr/bin:$PATH vvp /tmp/sim
```

## 3. (Optional) install a real HDL simulator

iverilog is already installed rootless at
`~/.local/opt/iverilog/root/usr/bin` (12.0; the `ivl` lib symlink under
`root/usr/x86_64-linux-gnu` is required and already fixed).
`run_streaming_equivalence.py` detects it automatically. To install via
package manager instead:

```bash
sudo apt-get install iverilog
```

## 4. Rebuild the custom compiler (optional)

The compiler binaries `mychip-opt` / `mychip-translate` live in
`my_chip_compiler/build/bin/`. If they need rebuilding:

```bash
cd my_chip_compiler/build && ninja mychip-opt
```
