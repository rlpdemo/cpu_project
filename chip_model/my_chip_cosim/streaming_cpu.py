"""
Streaming CPU model (from cpu_project/str.py) integrated with the MLIR
hardware-cosimulation workflow.

The MLIR compiler lowers the MNIST MLP into `EXEC_TILE_MATMUL` commands.  This
module keeps the architecture model from str.py — the stationary-weight
streaming fabric (`StreamingCPUSimulator`) — as the target hardware, and adds a
`StreamingCPUHardwareSimulator` adapter that:

  1. executes each compiled tile functionally (systolic matmul) so numerical
     parity against PyTorch can still be asserted, and
  2. feeds every tile's (M, K, N) geometry into the streaming CPU model,
     accumulating latency / DRAM / SRAM / energy / TOPS metrics for the whole
     compiled firmware image.
"""

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import torch

from my_chip_cosim.my_chip_sim import MyChipHardwareSimulator


# ============================================================================
# str.py model (kept verbatim: stationary-weight streaming fabric)
# ============================================================================
class PrecisionMode(Enum):
    FP16 = (2.0, 0.45)
    INT8 = (1.0, 0.12)
    INT4 = (0.5, 0.03)

    def __init__(self, bytes_per_elem: float, pj_per_mac: float):
        self.bytes_per_elem = bytes_per_elem
        self.pj_per_mac = pj_per_mac


class InstructionType(Enum):
    CUSTOM_STREAM = 7
    I_TYPE = 2


@dataclass
class DecodedInstruction:
    raw_bits: int
    mnemonic: str
    inst_type: InstructionType
    imm: int = 0


class LocalSRAMBuffer:
    def __init__(self, capacity_bytes: int = 512 * 1024):
        self.capacity_bytes = capacity_bytes
        self.allocated_bytes = 0

    def allocate(self, size_bytes: float) -> bool:
        if self.allocated_bytes + size_bytes <= self.capacity_bytes:
            self.allocated_bytes += size_bytes
            return True
        return False


class SpatialFabric2000:
    def __init__(self, num_engines: int = 2000, mesh_rows: int = 40, mesh_cols: int = 50):
        self.num_engines = num_engines
        self.mesh_rows = mesh_rows
        self.mesh_cols = mesh_cols
        self.configured_active_engines = 0

    def map_tensor_kernel(self, K: int, N: int) -> int:
        self.configured_active_engines = min(self.num_engines, K * N)
        return self.configured_active_engines


class StreamingCPUSimulator:
    def __init__(
        self,
        clock_freq_ghz: float = 1.0,
        dram_bw_gbps: float = 128.0,
        num_engines: int = 2000,
        precision: PrecisionMode = PrecisionMode.INT8,
    ):
        self.clock_freq_ghz = clock_freq_ghz
        self.clock_period_ns = 1.0 / clock_freq_ghz
        self.dram_bw_bytes_per_ns = dram_bw_gbps
        self.precision = precision

        self.fabric = SpatialFabric2000(num_engines=num_engines)
        self.sram_buffer = LocalSRAMBuffer(capacity_bytes=512 * 1024)

        self.total_macs_executed = 0
        self.dram_bytes_transferred = 0.0
        self.sram_bytes_transferred = 0.0
        self.simulated_latency_ns = 0.0
        self.tiles_executed = 0

        # Energy Parameters (pJ)
        self.pj_per_dram_byte = 160.0
        self.pj_per_sram_byte = 2.0
        self.pj_per_direct_neighbor_hop = 0.08  # Direct wire driving (1-hop)

    def run_streaming_pass(self, M: int, K: int, N: int) -> Dict:
        start_time = time.perf_counter()
        bytes_per_elem = self.precision.bytes_per_elem

        # 1. Map Array
        self.fabric.map_tensor_kernel(K, N)

        # 2. Stationary Weight Preload (Loaded ONCE from DRAM to Micro-Engines)
        weight_bytes = self.fabric.configured_active_engines * bytes_per_elem
        self.dram_bytes_transferred += weight_bytes
        self.simulated_latency_ns += weight_bytes / self.dram_bw_bytes_per_ns

        # 3. Stream Activations (Buffered in SRAM)
        act_bytes = M * K * bytes_per_elem
        out_bytes = M * N * bytes_per_elem

        if self.sram_buffer.allocate(act_bytes + out_bytes):
            self.sram_bytes_transferred += act_bytes + out_bytes
        else:
            self.dram_bytes_transferred += act_bytes + out_bytes

        total_macs = M * K * N
        self.total_macs_executed += total_macs
        self.tiles_executed += 1

        pipeline_depth = self.fabric.mesh_rows + self.fabric.mesh_cols - 1
        compute_cycles = (total_macs / max(1, self.fabric.configured_active_engines)) + pipeline_depth
        compute_latency_ns = compute_cycles * self.clock_period_ns

        self.simulated_latency_ns += compute_latency_ns

        return self._generate_metrics(time.perf_counter() - start_time)

    def _generate_metrics(self, wall_time_sec: float) -> Dict:
        total_flops = 2 * self.total_macs_executed

        dram_energy_uJ = (self.dram_bytes_transferred * self.pj_per_dram_byte) / 1e6
        sram_energy_uJ = (self.sram_bytes_transferred * self.pj_per_sram_byte) / 1e6
        mac_energy_uJ = (self.total_macs_executed * self.precision.pj_per_mac) / 1e6

        # Direct Neighbor Driving: 1 hop per spatial shift instead of 16 NoC hops
        neighbor_routing_uJ = (self.total_macs_executed * 1 * self.pj_per_direct_neighbor_hop) / 1e6

        total_energy_uJ = dram_energy_uJ + sram_energy_uJ + mac_energy_uJ + neighbor_routing_uJ

        sim_sec = self.simulated_latency_ns / 1e9
        tops = (total_flops / 1e12) / sim_sec if sim_sec > 0 else 0.0
        power_watts = (total_energy_uJ / 1e6) / sim_sec if sim_sec > 0 else 0.0
        tops_per_watt = tops / power_watts if power_watts > 0 else 0.0

        return {
            "precision_mode": self.precision.name,
            "simulated_latency_us": round(self.simulated_latency_ns / 1000.0, 3),
            "dram_transferred_bytes": round(self.dram_bytes_transferred, 1),
            "sram_transferred_bytes": round(self.sram_bytes_transferred, 1),
            "energy_breakdown_uJ": {
                "dram_fetch": round(dram_energy_uJ, 4),
                "sram_buffer": round(sram_energy_uJ, 4),
                "direct_wire_routing": round(neighbor_routing_uJ, 4),
                "engine_compute": round(mac_energy_uJ, 4),
                "total": round(total_energy_uJ, 4),
            },
            "throughput_tops": round(tops, 3),
            "power_watts": round(power_watts, 4),
            "efficiency_tops_per_watt": round(tops_per_watt, 4),
            "python_wall_time_ms": round(wall_time_sec * 1000.0, 2),
        }


# ============================================================================
# MLIR workflow adapter: execute translated tiles on the streaming CPU
# ============================================================================
class StreamingCPUHardwareSimulator(MyChipHardwareSimulator):
    """
    Executes MLIR-generated tile commands (commands.txt) on the streaming CPU.

    Inherits buffer binding / functional matmul execution from the systolic
    model (so golden-vs-silicon parity is still verifiable) and additionally
    feeds each resolved tile's (M, K, N) into `StreamingCPUSimulator` to
    accumulate streaming-architecture metrics for the whole compiled firmware.
    """

    def __init__(self, **streaming_kwargs):
        super().__init__()
        self.streaming_model = StreamingCPUSimulator(**streaming_kwargs)
        self.tile_metrics: List[Dict] = []

    def _sim_matmul_tile(self, lhs_line, rhs_line, dest_line):
        # Functional systolic execution (inherited): binds / auto-allocates
        # buffers and computes the tile so parity checks still work.
        super()._sim_matmul_tile(lhs_line, rhs_line, dest_line)

        # Extract the resolved tile geometry and accumulate streaming metrics.
        lhs = self.resolve_buffer(lhs_line)
        rhs = self.resolve_buffer(rhs_line)
        M, K = lhs.shape[0], lhs.shape[1]
        N = rhs.shape[1]

        total_macs_before = self.streaming_model.total_macs_executed
        latency_before = self.streaming_model.simulated_latency_ns
        dram_before = self.streaming_model.dram_bytes_transferred
        sram_before = self.streaming_model.sram_bytes_transferred

        tile_metrics = self.streaming_model.run_streaming_pass(M, K, N)

        # Incremental delta for this tile (the model accumulates state).
        macs = self.streaming_model.total_macs_executed - total_macs_before
        sim = self.streaming_model
        tile_energy = (
            (sim.dram_bytes_transferred - dram_before) * sim.pj_per_dram_byte
            + (sim.sram_bytes_transferred - sram_before) * sim.pj_per_sram_byte
            + macs * (sim.precision.pj_per_mac + sim.pj_per_direct_neighbor_hop)
        ) / 1e6

        self.tile_metrics.append({
            "tile": self.streaming_model.tiles_executed,
            "M": M, "K": K, "N": N,
            "latency_us": round(
                (self.streaming_model.simulated_latency_ns - latency_before) / 1000.0, 3),
            "macs": macs,
            "energy_total_uJ": round(tile_energy, 4),
        })

    def report(self) -> Dict:
        """Rolls up the accumulated streaming-CPU metrics for the whole program."""
        metrics = self.streaming_model._generate_metrics(time.perf_counter())
        metrics["tiles_executed"] = self.streaming_model.tiles_executed
        metrics["total_macs"] = self.streaming_model.total_macs_executed
        metrics["per_tile"] = list(self.tile_metrics)
        return metrics


def extract_tile_shapes(commands_file_path: str) -> List[Dict]:
    """Reads commands.txt and returns per-tile (M, K, N) geometry parsed from
    the memref types on the LHS / RHS / DEST registers."""
    with open(commands_file_path, "r") as f:
        content = f.read()

    blocks = re.findall(
        r'# --- BEGIN MY_CHIP EXECUTION COMMAND ---\n(.*?)\n# --- END COMMAND ---',
        content, re.DOTALL
    )

    def dims(text: str) -> List[int]:
        # Matches both plain (memref<1x784xf32>) and strided
        # (memref<1x784xf32, strided<[?, ?], offset: ?>>) memref types.
        m = re.search(r'memref<([\d]+(?:x[\d]+)*)x\w+(?:,|>)', text)
        if not m:
            return []
        return [int(d) for d in m.group(1).split('x')]

    shapes = []
    for block in blocks:
        lhs = re.search(r'REG_LHS_PTR:\s*(.*)', block)
        rhs = re.search(r'REG_RHS_PTR:\s*(.*)', block)
        ld = dims(lhs.group(1)) if lhs else []
        rd = dims(rhs.group(1)) if rhs else []
        if len(ld) == 2 and len(rd) == 2:
            shapes.append({"M": ld[0], "K": ld[1], "N": rd[1]})
    return shapes