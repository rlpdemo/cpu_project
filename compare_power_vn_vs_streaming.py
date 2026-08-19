#!/usr/bin/env python3
"""Power / energy comparison: Von Neumann GEMM (cpu_vn.py) vs streaming CPU (str.py).

Both simulators run identical GEMM tiles. Energy + memory coefficients are
matched (documented below) so the result isolates the architectural difference:

  * Von Neumann  : weights re-fetched from DRAM for every tile, operands
                   streamed through SRAM cache -> DRAM traffic dominates.
  * Streaming CPU: weights loaded ONCE and held stationary in the engine
                   array; only activations stream through SRAM.

Power (W) is computed consistently as  total_energy_uJ / total_latency_us.

Run:  python compare_power_vn_vs_streaming.py
"""

import numpy as np

from cpu_vn import VonNeumannGEMMSimulator
from str import StreamingCPUSimulator, PrecisionMode

# Matched assumptions (both INT8, 1 byte/element)
#   DRAM energy   160 pJ/byte   (cpu_vn: 20 pJ/bit * 8)
#   SRAM energy     2 pJ/byte   (cpu_vn: 0.25 pJ/bit * 8)
#   MAC energy    0.45 pJ
#   DRAM bandwidth 128 GB/s, core clock 1 GHz
#   cpu_vn keeps its nominal 1024-PE array; streaming its 2000-engine mesh.
VN_CFG = dict(
    num_pes=1024, clock_freq_ghz=1.0, dram_bw_gbps=128.0,
    sram_capacity_kb=512, word_bits=8,
    dram_energy_pj_bit=20.0, sram_energy_pj_bit=0.25, mac_energy_pj=0.45,
)
STREAM_CFG = dict(
    clock_freq_ghz=1.0, dram_bw_gbps=128.0, num_engines=2000,
    precision=PrecisionMode.INT8,
)

# (name, [(M, K, N), ...])  -- layer tiles per workload
WORKLOADS = {
    "MNIST MLP (784->128->64->10)": [(1, 784, 128), (1, 128, 64), (1, 64, 10)],
    "Mini-batch 64 MNIST (M=64)":   [(64, 784, 128), (64, 128, 64), (64, 64, 10)],
    "Large GEMM (M=128, K=N=1024)": [(128, 1024, 1024)],
}


def run_vn(tiles):
    sim = VonNeumannGEMMSimulator(**VN_CFG)
    tot_macs = tot_energy = tot_lat = 0.0
    tot_dram_bytes = 0.0
    for (m, k, n) in tiles:
        a = np.random.randn(m, k).astype(np.float32)
        w = np.random.randn(k, n).astype(np.float32)
        r = sim.execute_and_benchmark(a, w)
        tot_macs += r["total_macs"]
        tot_energy += r["energy_uJ"]["total"]
        tot_lat += r["simulated_latency_us"]
        tot_dram_bytes += r["energy_uJ"]["dram_fetch"] \
            / (VN_CFG["dram_energy_pj_bit"] * 8) * 1e6
    return tot_macs, tot_energy, tot_lat, tot_dram_bytes


def run_stream(tiles):
    sim = StreamingCPUSimulator(**STREAM_CFG)
    for (m, k, n) in tiles:
        sim.run_streaming_pass(m, k, n)
    m = sim._generate_metrics(0.0)
    e = m["energy_breakdown_uJ"]
    return (sim.total_macs_executed, e["total"], m["simulated_latency_us"],
            sim.dram_bytes_transferred)


def main() -> int:
    print("=" * 78)
    print("  POWER COMPARISON  --  Von Neumann GEMM vs Streaming CPU")
    print("=" * 78)
    print("Matched INT8 coefficients: DRAM 160 pJ/B | SRAM 2 pJ/B | "
          "MAC 0.45 pJ | 128 GB/s | 1 GHz")
    print()

    tot_vn = {"energy": 0.0, "lat": 0.0}
    tot_st = {"energy": 0.0, "lat": 0.0}

    for wname, tiles in WORKLOADS.items():
        macs, vn_e, vn_lat, vn_dram = run_vn(tiles)
        s_macs, st_e, st_lat, st_dram = run_stream(tiles)
        tot_vn["energy"] += vn_e
        tot_vn["lat"] += vn_lat
        tot_st["energy"] += st_e
        tot_st["lat"] += st_lat

        vn_pwr = vn_e / vn_lat if vn_lat else 0.0
        st_pwr = st_e / st_lat if st_lat else 0.0
        vn_tops_w = (2 * macs / (vn_lat * 1e-6)) / 1e12 / vn_pwr if vn_pwr else 0.0
        st_tops_w = (2 * s_macs / (st_lat * 1e-6)) / 1e12 / st_pwr if st_pwr else 0.0

        print(f"  {wname}  ({macs:,} MACs)")
        print(f"    {'':15s} {'Latency(us)':>11s} {'Energy(uJ)':>11s} "
              f"{'Power(W)':>9s} {'TOPS/W':>9s} {'DRAM(B)':>12s}")
        print(f"    {'Von Neumann':15s} {vn_lat:>11.3f} {vn_e:>11.2f} "
              f"{vn_pwr:>9.3f} {vn_tops_w:>9.4f} {vn_dram:>12,.0f}")
        print(f"    {'Streaming CPU':15s} {st_lat:>11.3f} {st_e:>11.2f} "
              f"{st_pwr:>9.3f} {st_tops_w:>9.4f} {st_dram:>12,.0f}")
        print(f"    -> streaming energy is {st_e / vn_e * 100:.1f}% of "
              f"Von Neumann; power {st_pwr / vn_pwr * 100:.1f}%")
        print()

    print("  TOTALS (all workloads)")
    print(f"    {'':15s} {'Energy(uJ)':>11s} {'Power(W)':>9s}")
    vn_p = tot_vn["energy"] / tot_vn["lat"]
    st_p = tot_st["energy"] / tot_st["lat"]
    print(f"    {'Von Neumann':15s} {tot_vn['energy']:>11.2f} {vn_p:>9.3f}")
    print(f"    {'Streaming CPU':15s} {tot_st['energy']:>11.2f} {st_p:>9.3f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())