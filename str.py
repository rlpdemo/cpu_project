import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
import numpy as np


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


if __name__ == "__main__":
    print("\n=========================================================================")
    print("  STATIONARY WEIGHT STREAMING BENCHMARK (M=1024, Nearest-Neighbor)      ")
    print("=========================================================================")

    for mode in [PrecisionMode.FP16, PrecisionMode.INT8, PrecisionMode.INT4]:
        sim = StreamingCPUSimulator(
            clock_freq_ghz=1.0, dram_bw_gbps=128.0, num_engines=2000, precision=mode
        )
        res = sim.run_streaming_pass(M=1024, K=40, N=50)

        print(f"\n--- Precision Mode: {res['precision_mode']} ---")
        print(f"  Latency: {res['simulated_latency_us']} us | DRAM Fetch: {res['dram_transferred_bytes']} B | SRAM Transfer: {res['sram_transferred_bytes']} B")
        print(f"  Energy Breakdown (uJ): {res['energy_breakdown_uJ']}")
        print(f"  Throughput: {res['throughput_tops']} TOPS | Power: {res['power_watts']} W | Efficiency: {res['efficiency_tops_per_watt']} TOPS/W")

    print("=========================================================================\n")