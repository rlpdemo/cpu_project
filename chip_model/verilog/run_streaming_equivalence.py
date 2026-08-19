"""4-way end-to-end equivalence for the streaming-CPU MNIST MLP.

For one seeded model and one input vector this checks that all four
representations agree:

  1. PyTorch golden model output                 (float32)
  2. Compiler model output                       (MLIR pipeline -> commands.txt
                                                 -> MyChipHardwareSimulator)
  3. Streaming-CPU model output                  (StreamingCPUHardwareSimulator,
                                                 same functional core + metrics)
  4. Generated Verilog RTL simulation            (vsim.py interpreter always;
                                                 iverilog when available)

(4) vs the embedded Python reference in the testbench is the RTL-level claim;
(1)-(3) vs the same reference is the compiler/architecture-level claim.  The RTL
and the testbench are regenerated from the current model weights each run, so
the seed-0 model, the compiler binary pipeline, and the simulator all see
identical weights and input.

Run:  python run_streaming_equivalence.py
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch_mlir
from torch_mlir.fx import export_and_import

HERE = Path(__file__).resolve().parent
CHIP_MODEL = HERE.parent
sys.path.insert(0, str(CHIP_MODEL))
sys.path.insert(0, str(HERE))

from my_chip_cosim.my_chip_sim import MyChipHardwareSimulator          # noqa: E402
from my_chip_cosim.streaming_cpu import StreamingCPUHardwareSimulator  # noqa: E402
from streaming_cpu_model import (                                     # noqa: E402
    build_model,
    python_reference,
    write_verilog_files,
)
import vsim  # noqa: E402

LLVM_BIN = Path("/home/us/llvm-mlir-install/bin")
MC_BIN = CHIP_MODEL / "my_chip_compiler" / "build" / "bin"
IVL_BIN = Path.home() / ".local" / "opt" / "iverilog" / "root" / "usr" / "bin"
WORK = HERE / "_streaming_equiv_work"

TOP = "streaming_cpu_top"     # DUT module name (also used for file names)
TB_TOP = "tb_streaming_cpu"   # testbench module name (simulation entry point)
SEED = 0
N_IN, N_H1, N_H2, N_OUT = 784, 128, 64, 10

TOL_RTL_REF = 1e-9    # RTL vs python_reference: identical 12-dec literals, double
TOL_MODEL = 2e-4      # python_reference vs PyTorch golden (float32 + rounding)
TOL_SIM = 1e-4        # compiler / streaming model vs golden


def run_compile_pipeline(model, x, workdir: Path) -> Path:
    """PyTorch -> Torch-MLIR -> bufferize -> mychip -> commands.txt."""
    mlir = export_and_import(model, torch.randn(1, N_IN),
                             output_type="linalg-on-tensors")
    math_graph = workdir / "math_graph.mlir"
    math_graph.write_text(str(mlir))

    subprocess.run(
        [str(LLVM_BIN / "mlir-opt"),
         "--pass-pipeline=builtin.module(empty-tensor-to-alloc-tensor,"
         "one-shot-bufferize)",
         str(math_graph), "-o", str(workdir / "buffer_graph.mlir")],
        check=True, capture_output=True, text=True)
    subprocess.run(
        [str(MC_BIN / "mychip-opt"), "--lower-to-mychip",
         str(workdir / "buffer_graph.mlir"),
         "-o", str(workdir / "final_hardware_ir.mlir")],
        check=True, capture_output=True, text=True)
    commands = workdir / "commands.txt"
    subprocess.run(
        [str(MC_BIN / "mychip-translate"), "--mlir-to-mychip-cmds",
         str(workdir / "final_hardware_ir.mlir"), "-o", str(commands)],
        check=True, capture_output=True, text=True)
    return commands


def bind_and_execute(sim, commands: Path, model, x):
    content = commands.read_text()
    blocks = re.findall(
        r'# --- BEGIN MY_CHIP EXECUTION COMMAND ---\n(.*?)\n# --- END COMMAND ---',
        content, re.DOTALL)
    lhs_vars = [re.search(r'REG_LHS_PTR:\s*(%\w+)', b).group(1)
                for b in blocks]
    rhs_vars = [re.search(r'REG_RHS_PTR:\s*(%\w+)', b).group(1)
                for b in blocks]
    dest_vars = [re.search(r'REG_DEST_PTR:\s*(%\w+)', b).group(1)
                 for b in blocks]
    assert len(blocks) == 3, f"expected 3 matmul tiles, got {len(blocks)}"
    sim.bind_buffer(lhs_vars[0], x)
    sim.bind_buffer(rhs_vars[0], model.fc1.weight.t().contiguous())
    sim.bind_buffer(rhs_vars[1], model.fc2.weight.t().contiguous())
    sim.bind_buffer(rhs_vars[2], model.fc3.weight.t().contiguous())
    sim.execute_commands(str(commands))
    return sim.get_buffer(dest_vars[-1]).flatten().tolist()


def parse_outputs(lines) -> list:
    vals = {}
    pat = re.compile(r"out(\d+)\s*=\s*(-?[\d.]+)")
    for m in pat.finditer("\n".join(lines)):
        vals[int(m.group(1))] = float(m.group(2))
    return [vals[i] for i in range(N_OUT)]


def run_vsim(rtl: Path, tb: Path) -> list:
    return parse_outputs(vsim.simulate_files([str(rtl), str(tb)], TB_TOP))


def run_iverilog(rtl: Path, tb: Path):
    if not (IVL_BIN / "iverilog").exists() and shutil.which("iverilog") is None:
        return None
    env = dict(__import__("os").environ)
    env["PATH"] = str(IVL_BIN) + ":" + env.get("PATH", "")
    build = WORK / "iverilog_build"
    build.mkdir(parents=True, exist_ok=True)
    vvp = build / f"{TOP}.vvp"
    subprocess.run(["iverilog", "-g2012", "-o", str(vvp), str(rtl), str(tb)],
                   check=True, capture_output=True, text=True, env=env)
    result = subprocess.run(["vvp", str(vvp)], check=True,
                            capture_output=True, text=True, env=env)
    return parse_outputs(result.stdout.splitlines())


def cmp(name, got, want, tol, ref_name):
    n = min(len(got), len(want))
    diffs = [abs(got[i] - want[i]) for i in range(n)]
    worst = max(diffs) if diffs else float("inf")
    ok = worst <= tol
    status = "PASS" if ok else "FAIL"
    print(f"    {name:28s} worst |diff| = {worst:.3e}  (tol {tol:.0e})  ->  {status}")
    return ok


def main() -> int:
    torch.manual_seed(SEED)
    model = build_model(seed=SEED)
    x = torch.randn(1, N_IN, dtype=torch.float32)

    WORK.mkdir(parents=True, exist_ok=True)
    print(f"Compiling model (seed {SEED}) through the MLIR toolchain...")
    commands = run_compile_pipeline(model, x, WORK)
    print(f"    -> {commands.relative_to(CHIP_MODEL)}")

    with torch.no_grad():
        golden = model(x).flatten().tolist()
    print("Golden (PyTorch, float32)       :",
          " ".join(f"{v:+.6f}" for v in golden))

    compiler_out = bind_and_execute(MyChipHardwareSimulator(), commands, model, x)
    stream_sim = StreamingCPUHardwareSimulator()
    stream_out = bind_and_execute(stream_sim, commands, model, x)
    metrics = stream_sim.report()

    print("Regenerating streaming-cpu RTL + testbench from the same weights...")
    rtl = HERE / f"{TOP}.v"
    tb = HERE / f"{TOP}_tb.v"
    write_verilog_files(model, x.flatten().tolist(), HERE, TOP)
    ref = python_reference(model, x.flatten().tolist())

    rtl_sim = run_vsim(rtl, tb)
    print(f"    vsim.py      : {' '.join(f'{v:+.9f}' for v in rtl_sim)}")
    ivl_sim = run_iverilog(rtl, tb)
    if ivl_sim is not None:
        print(f"    iverilog     : {' '.join(f'{v:+.9f}' for v in ivl_sim)}")
    else:
        print("    iverilog     : (not found -- using vsim.py only)")

    print("\nCross-check report")
    print("=" * 78)
    ok = True
    ok &= cmp("RTL(vsim)  vs python_ref", rtl_sim, ref, TOL_RTL_REF, "python_ref")
    if ivl_sim is not None:
        ok &= cmp("RTL(iverilog) vs python_ref", ivl_sim, ref, TOL_RTL_REF,
                  "python_ref")
    ok &= cmp("python_ref vs golden(torch)", ref, golden, TOL_MODEL, "golden")
    ok &= cmp("compiler model vs golden", compiler_out, golden, TOL_SIM, "golden")
    ok &= cmp("streaming model vs golden", stream_out, golden, TOL_SIM, "golden")
    print("=" * 78)

    print(f"\nStreaming-CPU metrics: {metrics['tiles_executed']} tiles, "
          f"{metrics['total_macs']} MACs, "
          f"latency {metrics['simulated_latency_us']} us, "
          f"energy {metrics['energy_breakdown_uJ']['total']} uJ (model)")

    if ok:
        print("\n\U0001F7E2 STREAMING-CPU EQUIVALENCE VERIFIED: PyTorch, compiler "
              "model, streaming model, and Verilog RTL all agree.")
        return 0
    print("\n\U0001F534 EQUIVALENCE FAILED: divergence detected.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
