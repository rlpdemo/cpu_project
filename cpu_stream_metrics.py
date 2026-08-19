#!/usr/bin/env python3
"""
Spatial Dataflow CPU Core (derivative of cpu_stream.py) with Metric Instrumentation.
- Retains the full cycle-accurate streaming datapath (TCM, elastic channels,
  ME fabric, pipeline assembler) of the original model.
- Adds per-cycle instrumentation (active/stall, TCM accesses, channel flits)
  plus clock/energy coefficients so execute_and_benchmark() returns the same
  style of report as cpu_vn.py:
      latency, operational intensity, stall/backpressure, energy breakdown,
      compute energy ratio and efficiency (TOPS/W).
"""

from collections import deque
import enum

# =====================================================================
# 1. Memory Subsystem (Tightly Coupled Memory / TCM)
# =====================================================================
class TightlyCoupledMemory:
    """Byte/Word-addressable dual-port memory shared by CPU & MEs."""
    def __init__(self, size_words=1024):
        self.mem = [0.0] * size_words

    def read(self, addr):
        return self.mem[addr]

    def write(self, addr, val):
        self.mem[addr] = float(val)


# =====================================================================
# 2. Elastic Interconnect (Valid/Ready Handshake with Backpressure)
# =====================================================================
class ElasticChannel:
    """Cycle-buffered queue with explicit valid/ready handshake logic."""
    def __init__(self, capacity=2):
        self.capacity = capacity
        self.buffer = deque()
        self.push_count = 0   # Metric: flits transported across this link

    def is_full(self):
        return len(self.buffer) >= self.capacity

    def is_empty(self):
        return len(self.buffer) == 0

    def push(self, val):
        if not self.is_full():
            self.buffer.append(val)
            self.push_count += 1
            return True
        return False  # Backpressure asserted to upstream ME

    def pop(self):
        if not self.is_empty():
            return self.buffer.popleft()
        return None


# =====================================================================
# 3. Micro-Engine (ME) Hardware Sub-modules
# =====================================================================
class METype(enum.Enum):
    IDLE = 0
    READ_AGU = 1
    MAC = 2
    PWL_ACT = 3
    WRITE_AGU = 4


class MicroEngine:
    """Generic Reconfigurable Micro-Engine Core Slice (with cycle counters)."""
    def __init__(self, me_id):
        self.me_id = me_id
        self.type = METype.IDLE
        self.in_channel = None
        self.out_channel = None
        
        # Reconfigurable Hardware Registers
        self.base_addr = 0
        self.stride = 1
        self.count = 0
        self.weight = 1.0
        self.bias = 0.0
        self.act_mode = "PASSTHROUGH"
        
        # Operational Runtime State
        self.curr_addr = 0
        self.rem_count = 0
        self.processed_count = 0

        # Metric Instrumentation
        self.active_cycles = 0
        self.stall_cycles = 0
        self.tcm_reads = 0
        self.tcm_writes = 0

    def configure(self, me_type, **kwargs):
        """Hardware register configuration routine driven by Pipeline Assembler."""
        self.type = me_type
        self.base_addr = kwargs.get("base_addr", 0)
        self.stride = kwargs.get("stride", 1)
        self.count = kwargs.get("count", 0)
        self.weight = kwargs.get("weight", 1.0)
        self.bias = kwargs.get("bias", 0.0)
        self.act_mode = kwargs.get("act_mode", "PASSTHROUGH")
        
        self.curr_addr = self.base_addr
        self.rem_count = self.count
        self.processed_count = 0

    def tick(self, memory):
        """Execute 1 clock cycle of the configured ME datapath."""
        if self.type == METype.IDLE:
            return

        elif self.type == METype.READ_AGU:
            # Stream memory out to output elastic channel
            if self.rem_count > 0 and self.out_channel and not self.out_channel.is_full():
                data = memory.read(self.curr_addr)
                if self.out_channel.push(data):
                    self.curr_addr += self.stride
                    self.rem_count -= 1
                    self.processed_count += 1
                    self.tcm_reads += 1
                    self.active_cycles += 1
                    return
            self.stall_cycles += 1  # Backpressure full or stream drained

        elif self.type == METype.MAC:
            # Pop input, apply Multiply-Accumulate, push output
            if self.in_channel and not self.in_channel.is_empty():
                if self.out_channel and not self.out_channel.is_full():
                    val = self.in_channel.pop()
                    res = (val * self.weight) + self.bias
                    self.out_channel.push(res)
                    self.processed_count += 1
                    self.active_cycles += 1
                    return
            self.stall_cycles += 1  # Empty input / full output (pipeline bubble)

        elif self.type == METype.PWL_ACT:
            # Non-linear activation evaluation (Piecewise Linear Tanh/ReLU)
            if self.in_channel and not self.in_channel.is_empty():
                if self.out_channel and not self.out_channel.is_full():
                    val = self.in_channel.pop()
                    if self.act_mode == "RELU":
                        res = max(0.0, val)
                    elif self.act_mode == "TANH":
                        # 3-segment PWL Tanh approximation
                        if val < -2.0: res = -1.0
                        elif val > 2.0: res = 1.0
                        else: res = 0.5 * val
                    else:
                        res = val
                    self.out_channel.push(res)
                    self.processed_count += 1
                    self.active_cycles += 1
                    return
            self.stall_cycles += 1  # Empty input / full output (pipeline bubble)

        elif self.type == METype.WRITE_AGU:
            # Pop input stream, write out to TCM memory with striding
            if self.in_channel and not self.in_channel.is_empty():
                val = self.in_channel.pop()
                memory.write(self.curr_addr, val)
                self.curr_addr += self.stride
                self.processed_count += 1
                self.tcm_writes += 1
                self.active_cycles += 1
                return
            self.stall_cycles += 1  # Empty input (pipeline bubble)


# =====================================================================
# 4. Reconfigurable ME Fabric & Crossbar Interconnect
# =====================================================================
class MEFabric:
    """Grid/Array of MEs connected by dynamic Elastic Crossbar routing."""
    def __init__(self, num_mes=4):
        self.mes = [MicroEngine(i) for i in range(num_mes)]
        self.channels = []

    def reset_routes(self):
        self.channels.clear()
        for me in self.mes:
            me.in_channel = None
            me.out_channel = None
            # Metric counters zeroed for a fresh run
            me.active_cycles = 0
            me.stall_cycles = 0
            me.tcm_reads = 0
            me.tcm_writes = 0

    def route_channel(self, src_me_id, dst_me_id, capacity=2):
        """Establishes an elastic handshake channel between two MEs."""
        chan = ElasticChannel(capacity=capacity)
        self.mes[src_me_id].out_channel = chan
        self.mes[dst_me_id].in_channel = chan
        self.channels.append(chan)

    def tick(self, memory):
        """
        Evaluates fabric in reverse topological order (Write -> PWL -> MAC -> Read).
        This models 1-cycle pipeline register boundaries without race conditions.
        """
        for me in reversed(self.mes):
            me.tick(memory)

    def stats(self):
        """Aggregates per-engine instrumentation into fabric-level metrics."""
        total_active = sum(me.active_cycles for me in self.mes)
        total_stall = sum(me.stall_cycles for me in self.mes)
        total_me_cycles = total_active + total_stall
        utilization_pct = (total_active / total_me_cycles * 100.0) if total_me_cycles else 0.0
        return {
            "total_active_cycles": total_active,
            "total_stall_cycles": total_stall,
            "pipeline_utilization_pct": utilization_pct,
            "tcm_reads": sum(me.tcm_reads for me in self.mes),
            "tcm_writes": sum(me.tcm_writes for me in self.mes),
            "flits": sum(c.push_count for c in self.channels)
        }


# =====================================================================
# 5. Spatial Front-End (Instruction Decoder & Pipeline Assembler)
# =====================================================================
class Instruction:
    """Custom Instruction format for spatial pipeline configuration."""
    def __init__(self, opcode, **kwargs):
        self.opcode = opcode
        self.kwargs = kwargs


class PipelineAssembler:
    """Decodes program instructions and configures spatial ME assembly lines."""
    def __init__(self, fabric):
        self.fabric = fabric

    def assemble_and_configure(self, instruction_stream):
        """Parses a graph definition block and maps it to the ME fabric."""
        self.fabric.reset_routes()
        stream_length = 0

        for instr in instruction_stream:
            op = instr.opcode
            args = instr.kwargs

            if op == "CFG_READ_AGU":
                me = self.fabric.mes[args["me_id"]]
                me.configure(METype.READ_AGU, base_addr=args["base_addr"],
                             stride=args["stride"], count=args["count"])
                stream_length = args["count"]

            elif op == "CFG_MAC":
                me = self.fabric.mes[args["me_id"]]
                me.configure(METype.MAC, weight=args["weight"], bias=args["bias"])

            elif op == "CFG_PWL":
                me = self.fabric.mes[args["me_id"]]
                me.configure(METype.PWL_ACT, act_mode=args["act_mode"])

            elif op == "CFG_WRITE_AGU":
                me = self.fabric.mes[args["me_id"]]
                me.configure(METype.WRITE_AGU, base_addr=args["base_addr"],
                             stride=args["stride"])

            elif op == "CONNECT":
                self.fabric.route_channel(args["src"], args["dst"])

        return stream_length


# =====================================================================
# 6. Top-Level Spatial CPU Core (with Clock & Energy Model)
# =====================================================================
class SpatialCPUCore:
    """Replacement CPU Core featuring Spatial Dataflow Execution + Metrics."""
    def __init__(
        self,
        clock_freq_ghz: float = 1.0,          # Core clock frequency in GHz
        word_bits: int = 32,                   # Element precision (FP32 datapath)
        tcm_access_cycles: int = 1,            # TCM SRAM access latency (cycles)
        tcm_energy_pj_bit: float = 2.0,        # Energy per bit for TCM (SRAM) access (pJ)
        mac_energy_pj: float = 0.5,            # Energy per MAC operation (pJ)
        pwl_energy_pj: float = 0.5,            # Energy per PWL activation op (pJ)
        interconnect_energy_pj_bit: float = 0.5  # Energy per bit per flit hop (pJ)
    ):
        self.memory = TightlyCoupledMemory(size_words=1024)
        self.fabric = MEFabric(num_mes=4)
        self.assembler = PipelineAssembler(self.fabric)
        self.pc = 0

        self.clock_freq_ghz = clock_freq_ghz
        self.clock_period_ns = 1.0 / clock_freq_ghz
        self.word_bits = word_bits
        self.tcm_access_cycles = tcm_access_cycles

        # Energy coefficients
        self.tcm_energy_pj_bit = tcm_energy_pj_bit
        self.mac_energy_pj = mac_energy_pj
        self.pwl_energy_pj = pwl_energy_pj
        self.interconnect_energy_pj_bit = interconnect_energy_pj_bit

    def run_spatial_program(self, program):
        print("=== [Spatial CPU] Front-End: Decoding Program & Assembling ME Fabric ===")
        
        # 1. Front-End Assembly Phase (1-2 Cycles)
        stream_len = self.assembler.assemble_and_configure(program)
        print(f"-> MEs Configured: ReadAGU(ME0) -> MAC(ME1) -> PWL_Tanh(ME2) -> WriteAGU(ME3)")
        print(f"-> Crossbar Elastic Channels Routed with Capacity=2")
        print("=== [Spatial CPU] Pipeline Unlocked: Beginning Streaming Execution ===")

        # 2. Streaming Execution Phase (PC Frozen, Dataflow active)
        cycles = 0
        writer_me = self.fabric.mes[3]  # ME3 is our output Write AGU

        while writer_me.processed_count < stream_len:
            cycles += 1
            self.fabric.tick(self.memory)

        print(f"=== [Spatial CPU] Kernel Completed in {cycles} Clock Cycles ===")
        print(f"-> Throughput: {round(stream_len / cycles, 2)} tokens/cycle\n")
        return cycles

    def execute_and_benchmark(self, program) -> dict:
        """
        Runs a spatial pipeline program and returns cycle-accurate latency,
        energy, utilization and efficiency metrics (mirrors cpu_vn.py report style).

        Args:
            program: List[Instruction] describing the spatial pipeline overlay.
        """
        stream_len = self.assembler.assemble_and_configure(program)

        # Streaming Execution Phase (instrumented tick)
        cycles = 0
        writer_me = self.fabric.mes[3]
        while writer_me.processed_count < stream_len:
            cycles += 1
            self.fabric.tick(self.memory)

        num_elements = stream_len
        stats = self.fabric.stats()

        # 1. Workload Data Size and Intensity
        total_macs = num_elements                    # 1 MAC per element (ME1)
        total_act_ops = num_elements                 # 1 PWL activation per element (ME2)
        total_flops = 2 * total_macs + total_act_ops # Mul + Add + PWL eval

        bytes_per_element = self.word_bits // 8
        memory_bytes = (stats["tcm_reads"] + stats["tcm_writes"]) * bytes_per_element
        operational_intensity = total_flops / memory_bytes if memory_bytes else 0.0

        # 2. Latency & Bottleneck Modeling
        total_simulated_latency_ns = cycles * self.clock_period_ns
        memory_latency_ns = (stats["tcm_reads"] + stats["tcm_writes"]) * \
                            self.tcm_access_cycles * self.clock_period_ns

        stall_percentage = round(100.0 - stats["pipeline_utilization_pct"], 2)
        is_memory_bound = stall_percentage > 15.0  # Backpressure / pipeline bubbles dominate

        # 3. Energy Breakdown
        energy_tcm_uJ = (stats["tcm_reads"] + stats["tcm_writes"]) * \
                        self.word_bits * self.tcm_energy_pj_bit / 1e6
        energy_mac_uJ = (total_macs * self.mac_energy_pj) / 1e6
        energy_pwl_uJ = (total_act_ops * self.pwl_energy_pj) / 1e6
        energy_net_uJ = (stats["flits"] * self.word_bits * self.interconnect_energy_pj_bit) / 1e6
        total_energy_uJ = energy_tcm_uJ + energy_mac_uJ + energy_pwl_uJ + energy_net_uJ

        # 4. Output readback via writer AGU striding
        out_base = writer_me.base_addr
        out_stride = max(writer_me.stride, 1)
        output = [self.memory.read(out_base + (i * out_stride)) for i in range(num_elements)]

        return {
            "output": output,
            "shape": {"num_elements": num_elements, "num_pes": len(self.fabric.mes)},
            "total_macs": total_macs,
            "total_flops": total_flops,
            "operational_intensity_flops_per_byte": round(operational_intensity, 2),
            "simulated_latency_us": round(total_simulated_latency_ns / 1000.0, 3),
            "compute_latency_us": round(total_simulated_latency_ns / 1000.0, 3),
            "memory_latency_us": round(memory_latency_ns / 1000.0, 3),
            "total_cycles": cycles,
            "tokens_per_cycle": round(num_elements / cycles, 3),
            "throughput_gflops": round((total_flops / (total_simulated_latency_ns * 1e-9)) / 1e9, 3),
            "is_memory_bound": is_memory_bound,
            "memory_stall_percentage": stall_percentage,
            "energy_uJ": {
                "tcm_read_write": round(energy_tcm_uJ, 6),
                "compute_mac": round(energy_mac_uJ, 6),
                "pwl_activation": round(energy_pwl_uJ, 6),
                "interconnect": round(energy_net_uJ, 6),
                "total": round(total_energy_uJ, 6)
            },
            "compute_energy_ratio_pct": round(
                ((energy_mac_uJ + energy_pwl_uJ) / total_energy_uJ) * 100, 2),
            "efficiency_tops_per_watt": round(
                (total_flops / 1e12) / (total_energy_uJ / (total_simulated_latency_ns * 1e-3)), 3)
        }


# =====================================================================
# 7. Verification Workload Run (Metric Report)
# =====================================================================
if __name__ == "__main__":
    cpu = SpatialCPUCore()

    # Seed Input Data into Memory at Base Address 0x100
    input_tensor = [-2.0, -1.0, 0.0, 1.0, 2.0]
    in_base = 0x100
    out_base = 0x200

    for i, val in enumerate(input_tensor):
        cpu.memory.write(in_base + i, val)

    # Program Instruction Stream (Spatial Assembly Overlay for: Y = Tanh(X * 2.0 + 0.5))
    spatial_program = [
        Instruction("CFG_READ_AGU",  me_id=0, base_addr=in_base, stride=1, count=len(input_tensor)),
        Instruction("CFG_MAC",       me_id=1, weight=2.0, bias=0.5),
        Instruction("CFG_PWL",       me_id=2, act_mode="TANH"),
        Instruction("CFG_WRITE_AGU", me_id=3, base_addr=out_base, stride=2), # Stride=2 performs Transpose/Interleaving
        
        # Dynamic Elastic Routing Configuration
        Instruction("CONNECT", src=0, dst=1),
        Instruction("CONNECT", src=1, dst=2),
        Instruction("CONNECT", src=2, dst=3),
    ]

    # Execute Program on Spatial Core & Collect Metrics
    results = cpu.execute_and_benchmark(spatial_program)

    print("=== Execution Results Verification ===")
    print(f"Input Stream  (Address 0x100): {input_tensor}")
    print(f"Output Stream (Address 0x200): {results['output']}")
    print()
    for k, v in results.items():
        if k != "output":
            print(f"{k}: {v}")