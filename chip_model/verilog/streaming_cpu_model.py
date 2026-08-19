"""Source-of-truth model + Verilog generator for the streaming CPU core.

This is the RTL counterpart of ``pipeline_run_streaming.py`` / the
``StreamingCPUHardwareSimulator`` from ``my_chip_cosim/streaming_cpu.py``.  The
MLIR compiler lowers the MNIST MLP (784 -> 128 -> 64 -> 10) into three
``EXEC_TILE_MATMUL`` tiles; every tile is executed on a weight-stationary
streaming MAC fabric.  That fabric is exactly what the generated RTL models:

  * stationary weights, preloaded once per stage (DRAM -> PE preload),
  * activations streamed in one per clock cycle (1 activation/cycle/stage),
  * per-PE left-to-right multiply-accumulate in IEEE-754 double precision.

Consumers:
  * ``run_streaming_equivalence.py`` -- PyTorch golden + compiler-model outputs
    (``MyChipHardwareSimulator`` / ``StreamingCPUHardwareSimulator``).
  * ``streaming_cpu_top.v``      -- generated weight-stationary streaming RTL.
  * ``streaming_cpu_top_tb.v``   -- generated self-checking testbench.
  * ``python_reference``         -- bit-faithful double-precision software
    model of the generated RTL (used when no HDL simulator is present and by
    the generated testbench for its expected values).

Run:  python streaming_cpu_model.py   (regenerates the .v files)
"""

from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn

# Decimals used when a float is printed into a Verilog `real` literal.
# ``python_reference`` re-parses with the SAME precision so the software model
# and the RTL see bit-identical operands.
REAL_DECIMALS = 12


def format_real(value: float) -> str:
    return f"{float(value):.{REAL_DECIMALS}f}"


def _as_real(value: float) -> float:
    """Value of a Verilog `real` literal as emitted by ``format_real``."""
    return float(format_real(value))


# ============================================================================
# Model (mirrors pipeline_run_cosim.py / pipeline_run_streaming.py)
# ============================================================================
class MNISTModel(nn.Module):
    """Pure-linear 3-layer MLP: 784 -> 128 -> 64 -> 10 (no bias / activations)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.fc3 = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def build_model(seed: int = 0) -> MNISTModel:
    torch.manual_seed(seed)
    return MNISTModel().eval()


def linear_layers(model: nn.Module) -> List[nn.Linear]:
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def python_reference(model: nn.Module, x: Sequence[float]) -> List[float]:
    """Bit-faithful software model of ``streaming_cpu_top.v``.

    For each stage computes ``out[n] = sum_k x[k] * W[n][k]`` in IEEE-754 double
    precision with left-to-right accumulation and 12-decimal literals -- exactly
    the arithmetic the generated RTL performs (and therefore what a correct
    iverilog / vsim run must reproduce).
    """
    act = [_as_real(v) for v in x]
    for fc in linear_layers(model):
        weights = fc.weight.detach().cpu().numpy()  # (N, K)
        nxt: List[float] = []
        for row in weights:  # row == W[n]
            acc = 0.0
            for k in range(len(row)):
                acc += act[k] * _as_real(row[k])
            nxt.append(acc)
        act = nxt
    return act


# ============================================================================
# Verilog RTL generator
# ============================================================================
def generate_rtl(model: nn.Module, top_name: str = "streaming_cpu_top") -> str:
    """Generate the weight-stationary streaming MAC fabric RTL.

    Layout:  stage0 (1x784 @ 784x128) -> stage1 (1x128 @ 128x64)
             -> stage2 (1x64 @ 64x10).
    Each stage is a one-row MAC array: N PEs, each holding a column of the
    transposed weight matrix, accumulating one streamed activation per cycle.
    """
    layers = linear_layers(model)
    shapes = [(int(l.weight.shape[0]), int(l.weight.shape[1])) for l in layers]
    outs = shapes[-1][0]
    weights = [l.weight.detach().cpu().numpy() for l in layers]
    w_names = ["w0", "w1", "w2"]

    L: List[str] = []
    L.append("// Auto-generated from streaming_cpu_model.py -- DO NOT EDIT BY HAND.")
    L.append(f"// Streaming CPU core: pure-linear MLP "
             f"{' -> '.join(str(s[1]) for s in shapes)} -> {outs}.")
    L.append("// Weight-stationary streaming MAC fabric (1 activation/cycle/stage).")
    L.append("")
    L.append("`timescale 1ns/1ps")
    L.append("")
    L.append(f"module {top_name}(")
    ports = [
        "    input wire clk,",
        "    input wire rst,",
        "    input wire start,",
        "    input real act_in,",
    ]
    for n in range(outs):
        ports.append(f"    output real out{n},")
    ports.append("    output reg done")
    L.append("\n".join(ports))
    L.append(");")
    L.append("")

    L.append("    // ---- Stage weight arrays (stationary weights, preloaded once) ----")
    for nm, (nn_, kk) in zip(w_names, shapes):
        L.append(f"    reg real {nm}[0:{nn_ - 1}][0:{kk - 1}];")
    L.append("")
    L.append("    // ---- Stage accumulator arrays ----")
    for si, (nn_, _kk) in enumerate(shapes):
        L.append(f"    reg real acc{si}[0:{nn_ - 1}];")
    L.append("")
    L.append("    // ---- Cross-stage handoff registers ----")
    L.append(f"    reg real buf0[0:{shapes[0][0] - 1}];")
    L.append(f"    reg real buf1[0:{shapes[1][0] - 1}];")
    L.append("")
    L.append("    integer state;")
    L.append("    integer cnt;")
    L.append("    integer i;")
    L.append("")

    L.append("    // ---- Stationary weight preload (DRAM -> PE preload) ----")
    L.append("    initial begin")
    for nm, W in zip(w_names, weights):
        for n in range(W.shape[0]):
            for k in range(W.shape[1]):
                L.append(f"        {nm}[{n}][{k}] = {format_real(W[n][k])};")
    L.append("    end")
    L.append("")

    L.append("    // ---- Streaming datapath + control FSM ----")
    L.append("    always @(posedge clk) begin")
    L.append("        if (rst) begin")
    L.append("            state <= 0; cnt <= 0; done <= 0;")
    for si, (nn_, _kk) in enumerate(shapes):
        L.append(f"            for (i=0;i<{nn_};i=i+1) acc{si}[i] <= 0.0;")
    L.append("        end else begin")
    L.append("            done <= 0;")
    L.append("            if (state == 0) begin")
    L.append("                if (start) begin")
    L.append("                    state <= 1; cnt <= 0;")
    L.append(f"                    for (i=0;i<{shapes[0][0]};i=i+1) acc0[i] <= 0.0;")
    L.append("                end")
    L.append("            end")
    # --- stage 0: consume 784 streamed activations -------------------------
    L.append("            else if (state == 1) begin")
    L.append(f"                for (i=0;i<{shapes[0][0]};i=i+1)")
    L.append("                    acc0[i] <= acc0[i] + act_in * w0[i][cnt];")
    L.append(f"                if (cnt == {shapes[0][1] - 1}) begin")
    L.append("                    state <= 2; cnt <= 0;")
    L.append(f"                    for (i=0;i<{shapes[0][0]};i=i+1) begin")
    L.append("                        buf0[i] <= acc0[i] + act_in * w0[i][cnt];")
    L.append("                    end")
    L.append(f"                    for (i=0;i<{shapes[1][0]};i=i+1) acc1[i] <= 0.0;")
    L.append("                end else begin")
    L.append("                    cnt <= cnt + 1;")
    L.append("                end")
    L.append("            end")
    # --- stage 1: consume the 128 held activations -------------------------
    L.append("            else if (state == 2) begin")
    L.append(f"                for (i=0;i<{shapes[1][0]};i=i+1)")
    L.append("                    acc1[i] <= acc1[i] + buf0[cnt] * w1[i][cnt];")
    L.append(f"                if (cnt == {shapes[1][1] - 1}) begin")
    L.append("                    state <= 3; cnt <= 0;")
    L.append(f"                    for (i=0;i<{shapes[1][0]};i=i+1) begin")
    L.append("                        buf1[i] <= acc1[i] + buf0[cnt] * w1[i][cnt];")
    L.append("                    end")
    L.append(f"                    for (i=0;i<{shapes[2][0]};i=i+1) acc2[i] <= 0.0;")
    L.append("                end else begin")
    L.append("                    cnt <= cnt + 1;")
    L.append("                end")
    L.append("            end")
    # --- stage 2: consume the 64 held activations, publish logits ----------
    L.append("            else if (state == 3) begin")
    L.append(f"                for (i=0;i<{shapes[2][0]};i=i+1)")
    L.append("                    acc2[i] <= acc2[i] + buf1[cnt] * w2[i][cnt];")
    L.append(f"                if (cnt == {shapes[2][1] - 1}) begin")
    L.append("                    state <= 0; cnt <= 0; done <= 1;")
    for n in range(outs):
        L.append(f"                    out{n} <= acc2[{n}] + buf1[cnt] * w2[{n}][cnt];")
    L.append("                end else begin")
    L.append("                    cnt <= cnt + 1;")
    L.append("                end")
    L.append("            end")
    L.append("        end")
    L.append("    end")
    L.append("endmodule")
    return "\n".join(L) + "\n"


# ============================================================================
# Verilog testbench generator
# ============================================================================
def generate_testbench(model: nn.Module, x: Sequence[float],
                       top_name: str = "streaming_cpu_top",
                       tol: float = 1e-9) -> str:
    """Generate a self-checking testbench streaming one input vector through.

    Expected outputs are the *Verilog reference* values (double precision) --
    what a correct RTL simulation must reproduce exactly.
    """
    layers = linear_layers(model)
    outs = layers[-1].weight.shape[0]
    k0 = layers[0].weight.shape[1]
    expected = python_reference(model, x)

    L: List[str] = []
    L.append("// Auto-generated self-checking testbench for " + top_name)
    L.append("`timescale 1ns/1ps")
    L.append("")
    L.append("module tb_streaming_cpu;")
    L.append("    reg clk = 0;")
    L.append("    reg rst = 1;")
    L.append("    reg start = 0;")
    L.append("    real act_in;")
    L.append("    reg done;")
    for n in range(outs):
        L.append(f"    real out{n};")
    L.append(f"    real invec[0:{k0 - 1}];")
    L.append("    integer i;")
    L.append("    integer errors;")
    for n in range(outs):
        L.append(f"    real e{n};")
    L.append("")
    conns = [".clk(clk)", ".rst(rst)", ".start(start)",
             ".act_in(act_in)", ".done(done)"]
    for n in range(outs):
        conns.append(f".out{n}(out{n})")
    L.append(f"    {top_name} dut (")
    L.append("        " + ",\n        ".join(conns))
    L.append("    );")
    L.append("")
    L.append("    initial begin")
    for i, v in enumerate(x):
        L.append(f"        invec[{i}] = {format_real(v)};")
    L.append("")
    for n in range(outs):
        L.append(f"        e{n} = {format_real(expected[n])};")
    L.append("")
    L.append("        errors = 0;")
    L.append("        #10 rst = 0;")
    L.append("        #10 start = 1;")
    L.append("        #10 start = 0;")
    L.append(f"        for (i=0;i<{k0};i=i+1) begin act_in = invec[i]; #10; end")
    L.append("        while (!done) #10;")
    L.append("        #10;")
    L.append("")
    for n in range(outs):
        L.append(f'        $display("out{n} = %0.12f (expected %0.12f)", out{n}, e{n});')
        L.append(f"        if (($abs(out{n} - e{n})) > {format_real(tol)}) errors = errors + 1;")
    L.append("")
    L.append("        if (errors == 0)")
    L.append('            $display("PASS: streaming CPU RTL matches the Python reference on all outputs.");')
    L.append("        else")
    L.append('            $display("FAIL: %0d mismatching outputs.", errors);')
    L.append("        $finish;")
    L.append("    end")
    L.append("")
    L.append("    initial begin")
    L.append("        repeat(5000) begin #5 clk = ~clk; end")
    L.append("    end")
    L.append("endmodule")
    return "\n".join(L) + "\n"


def write_verilog_files(model: nn.Module, x: Sequence[float],
                        out_dir: Path,
                        top_name: str = "streaming_cpu_top") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{top_name}.v").write_text(generate_rtl(model, top_name))
    (out_dir / f"{top_name}_tb.v").write_text(
        generate_testbench(model, x, top_name))


if __name__ == "__main__":
    model = build_model(seed=0)
    x = torch.randn(1, 784, dtype=torch.float32)
    out_dir = Path(__file__).resolve().parent
    write_verilog_files(model, x.flatten().tolist(), out_dir)
    print(f"Wrote streaming_cpu_top.v / streaming_cpu_top_tb.v to {out_dir}")
    ref = python_reference(model, x.flatten().tolist())
    print("python_reference (10 logits):", [round(v, 6) for v in ref])