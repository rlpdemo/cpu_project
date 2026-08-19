import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict
import numpy as np


class PrecisionMode(Enum):
    FP16 = (2.0, 0.45)   # (bytes_per_element, pJ_per_MAC)
    INT8 = (1.0, 0.12)
    INT4 = (0.5, 0.03)

    def __init__(self, bytes_per_elem: float, pj_per_mac: float):
        self.bytes_per_elem = bytes_per_elem
        self.pj_per_mac = pj_per_mac


class VonNeumannGEMMSimulator:
    """
    Simulates a traditional Von Neumann hardware execution unit for GEMM.
    Models instruction/data fetches through central SRAM cache and DRAM buses.
    """
    def __init__(
        self,
        num_pes: int = 2000,             # Aligned to 2,000 PEs (40x50 array scale)
        clock_freq_ghz: float = 1.0,     # Core clock frequency in GHz
        dram_bw_gbps: float = 128.0,     # DRAM memory bandwidth in GB/s
        sram_bw_gbps: float = 512.0,     # On-chip SRAM cache bandwidth in GB/s
        sram_capacity_kb: int = 512,     # SRAM Cache capacity in KB
        precision: PrecisionMode = PrecisionMode.INT8
    ):
        self.num_pes = num_pes
        self.clock_freq_ghz = clock_freq_ghz
        self.clock_period_ns = 1.0 / clock_freq_ghz
        self.dram_bw_bytes_per_ns = dram_bw_gbps
        self.sram_bw_bytes_per_ns = sram_bw_gbps
        self.sram_capacity_bytes = sram_capacity_kb * 1024
        self.precision = precision

        # Energy coefficients (pJ per Byte / MAC)
        self.pj_per_dram_byte = 160.0
        self.pj_per_sram_byte = 2.0

    def execute_and_benchmark(self, M: int, K: int, N: int) -> Dict:
        bytes_per_elem = self.precision.bytes_per_elem
        pj_per_mac = self.precision.pj_per_mac

        total_macs = M * K * N
        total_flops = 2 * total_macs

        # 1. Memory Traffic Sizing
        bytes_act = M * K * bytes_per_elem
        bytes_weights = K * N * bytes_per_elem
        bytes_output = M * N * bytes_per_elem

        # Off-chip DRAM transfers (Activations in, Weights in, Outputs out)
        total_dram_bytes = bytes_act + bytes_weights + bytes_output

        # Von Neumann SRAM Reads/Writes:
        # Activations fetched into PE registers, Weights fetched per tile/pass, Outputs written back
        sram_read_act = M * K * bytes_per_elem
        sram_read_weights = K * N * M * bytes_per_elem  # Re-read weights per activation row from SRAM
        sram_write_output = M * N * bytes_per_elem
        total_sram_bytes = sram_read_act + sram_read_weights + sram_write_output

        # 2. Latency & Bottleneck Modeling
        compute_cycles = total_macs / self.num_pes
        compute_latency_ns = compute_cycles * self.clock_period_ns
        dram_latency_ns = total_dram_bytes / self.dram_bw_bytes_per_ns
        sram_latency_ns = total_sram_bytes / self.sram_bw_bytes_per_ns

        # Total Execution Latency
        simulated_latency_ns = max(compute_latency_ns, dram_latency_ns) + (sram_latency_ns * 0.05)

        # 3. Energy Breakdown (uJ)
        energy_dram_uJ = (total_dram_bytes * self.pj_per_dram_byte) / 1e6
        energy_sram_uJ = (total_sram_bytes * self.pj_per_sram_byte) / 1e6
        energy_compute_uJ = (total_macs * pj_per_mac) / 1e6
        total_energy_uJ = energy_dram_uJ + energy_sram_uJ + energy_compute_uJ

        # 4. Performance Metrics
        sim_sec = simulated_latency_ns / 1e9
        tops = (total_flops / 1e12) / sim_sec if sim_sec > 0 else 0.0
        power_watts = (total_energy_uJ / 1e6) / sim_sec if sim_sec > 0 else 0.0
        tops_per_watt = tops / power_watts if power_watts > 0 else 0.0

        return {
            "precision_mode": self.precision.name,
            "simulated_latency_us": round(simulated_latency_ns / 1000.0, 3),
            "dram_transferred_bytes": round(total_dram_bytes, 1),
            "sram_transferred_bytes": round(total_sram_bytes, 1),
            "energy_breakdown_uJ": {
                "dram_fetch": round(energy_dram_uJ, 4),
                "sram_cache": round(energy_sram_uJ, 4),
                "engine_compute": round(energy_compute_uJ, 4),
                "total": round(total_energy_uJ, 4),
            },
            "throughput_tops": round(tops, 3),
            "power_watts": round(power_watts, 4),
            "efficiency_tops_per_watt": round(tops_per_watt, 4),
        }


if __name__ == "__main__":
    print("\n=========================================================================")
    print("  VON NEUMANN GEMM BENCHMARK (M=1024, K=40, N=50, 2000 PEs)              ")
    print("=========================================================================")

    for mode in [PrecisionMode.FP16, PrecisionMode.INT8, PrecisionMode.INT4]:
        sim = VonNeumannGEMMSimulator(
            num_pes=2000, dram_bw_gbps=128.0, sram_capacity_kb=512, precision=mode
        )
        res = sim.execute_and_benchmark(M=1024, K=40, N=50)

        print(f"\n--- Precision Mode: {res['precision_mode']} ---")
        print(f"  Latency: {res['simulated_latency_us']} us | DRAM Fetch: {res['dram_transferred_bytes']} B | SRAM Transfer: {res['sram_transferred_bytes']} B")
        print(f"  Energy Breakdown (uJ): {res['energy_breakdown_uJ']}")
        print(f"  Throughput: {res['throughput_tops']} TOPS | Power: {res['power_watts']} W | Efficiency: {res['efficiency_tops_per_watt']} TOPS/W")

    print("=========================================================================\n")