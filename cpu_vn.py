import time
import numpy as np

class VonNeumannGEMMSimulator:
    """
    Simulates a traditional Von Neumann hardware execution unit for General Matrix Multiplication (GEMM).
    Models off-chip DRAM traffic, SRAM cache buffering, PE array compute cycles, and energy breakdown.
    """
    def __init__(
        self,
        num_pes: int = 1024,            # Number of Processing Elements (e.g., 32x32 MAC array)
        clock_freq_ghz: float = 1.0,    # Core clock frequency in GHz
        dram_bw_gbps: float = 64.0,     # Off-chip DRAM memory bandwidth in GB/s
        sram_bw_gbps: float = 512.0,    # On-chip SRAM cache bandwidth in GB/s
        sram_capacity_kb: int = 512,    # SRAM Cache capacity in KB
        word_bits: int = 16,            # Element precision (16-bit FP16/INT16)
        dram_energy_pj_bit: float = 20.0, # Energy per bit fetched from DRAM (pJ)
        sram_energy_pj_bit: float = 2.0,  # Energy per bit fetched from SRAM (pJ)
        mac_energy_pj: float = 0.5        # Energy per MAC operation (pJ)
    ):
        self.num_pes = num_pes
        self.clock_freq_ghz = clock_freq_ghz
        self.clock_period_ns = 1.0 / clock_freq_ghz
        self.dram_bw_bytes_per_ns = dram_bw_gbps
        self.sram_bw_bytes_per_ns = sram_bw_gbps
        self.sram_capacity_bytes = sram_capacity_kb * 1024
        self.bytes_per_element = word_bits // 8
        
        # Energy coefficients
        self.dram_energy_pj_bit = dram_energy_pj_bit
        self.sram_energy_pj_bit = sram_energy_pj_bit
        self.mac_energy_pj = mac_energy_pj

    def execute_and_benchmark(self, A: np.ndarray, W: np.ndarray) -> dict:
        """
        Executes functional GEMM (Y = A @ W) and returns cycle-accurate latency and energy metrics.
        
        Args:
            A: Input activation matrix of shape (M, K)
            W: Weight matrix of shape (K, N)
        """
        M, K = A.shape
        K_w, N = W.shape
        assert K == K_w, f"Dimension mismatch: A shape {A.shape} vs W shape {W.shape}"
        
        # 1. Functional Golden Output
        start_time = time.perf_counter()
        Y = np.matmul(A, W)
        sw_execution_time_ms = (time.perf_counter() - start_time) * 1000.0

        # 2. Workload Data Size and Intensity
        total_macs = M * K * N
        total_flops = 2 * total_macs
        
        bytes_act = M * K * self.bytes_per_element
        bytes_weights = K * N * self.bytes_per_element
        bytes_output = M * N * self.bytes_per_element
        nominal_memory_bytes = bytes_act + bytes_weights + bytes_output
        
        # Operational Intensity (FLOPs / Byte)
        operational_intensity = total_flops / nominal_memory_bytes
        
        # 3. Latency & Bottleneck Modeling
        # Compute latency based on available Processing Elements
        compute_cycles = total_macs / self.num_pes
        compute_latency_ns = compute_cycles * self.clock_period_ns
        
        # Tiling & DRAM thrashing overhead if working set exceeds SRAM cache size
        working_set_bytes = bytes_weights + bytes_act
        if working_set_bytes <= self.sram_capacity_bytes:
            effective_dram_bytes = nominal_memory_bytes
        else:
            # Weight thrashing penalty when tiling is required
            tile_iterations = np.ceil(working_set_bytes / self.sram_capacity_bytes)
            effective_dram_bytes = (bytes_weights * tile_iterations) + bytes_act + bytes_output
            
        dram_latency_ns = effective_dram_bytes / self.dram_bw_bytes_per_ns
        sram_latency_ns = (nominal_memory_bytes * 2) / self.sram_bw_bytes_per_ns
        
        # Execution Latency (Determined by memory vs compute bottleneck)
        is_memory_bound = dram_latency_ns > compute_latency_ns
        total_simulated_latency_ns = max(compute_latency_ns, dram_latency_ns) + (sram_latency_ns * 0.05)
        stall_percentage = max(0.0, ((dram_latency_ns - compute_latency_ns) / total_simulated_latency_ns) * 100) if is_memory_bound else 0.0

        # 4. Energy Breakdown
        energy_dram_uJ = (effective_dram_bytes * 8 * self.dram_energy_pj_bit) / 1e6
        energy_sram_uJ = (nominal_memory_bytes * 8 * self.sram_energy_pj_bit) / 1e6
        energy_compute_uJ = (total_macs * self.mac_energy_pj) / 1e6
        total_energy_uJ = energy_dram_uJ + energy_sram_uJ + energy_compute_uJ

        return {
            "output": Y,
            "shape": {"M": M, "K": K, "N": N},
            "total_macs": total_macs,
            "operational_intensity_flops_per_byte": round(operational_intensity, 2),
            "simulated_latency_us": round(total_simulated_latency_ns / 1000.0, 3),
            "compute_latency_us": round(compute_latency_ns / 1000.0, 3),
            "dram_fetch_latency_us": round(dram_latency_ns / 1000.0, 3),
            "is_memory_bound": is_memory_bound,
            "memory_stall_percentage": round(stall_percentage, 2),
            "energy_uJ": {
                "dram_fetch": round(energy_dram_uJ, 3),
                "sram_cache": round(energy_sram_uJ, 3),
                "compute_mac": round(energy_compute_uJ, 3),
                "total": round(total_energy_uJ, 3)
            },
            "compute_energy_ratio_pct": round((energy_compute_uJ / total_energy_uJ) * 100, 2),
            "efficiency_tops_per_watt": round((total_flops / 1e12) / (total_energy_uJ / (total_simulated_latency_ns * 1e-3)), 3)
        }

# Example Usage
if __name__ == "__main__":
    sim = VonNeumannGEMMSimulator(
        num_pes=1024,          # 32x32 MAC array
        dram_bw_gbps=64.0,     # LPDDR5 / HBM bandwidth constraint
        sram_capacity_kb=512
    )

    # Test matrix: M=16 (batch/sequence), K=1024, N=1024 (FC / Projection layer)
    A = np.random.randn(16, 1024).astype(np.float32)
    W = np.random.randn(1024, 1024).astype(np.float32)

    results = sim.execute_and_benchmark(A, W)
    for k, v in results.items():
        if k != "output":
            print(f"{k}: {v}")