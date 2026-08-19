#!/usr/bin/env python3
"""
Spatial Dataflow CPU Core (Choice B Replacement CPU Architecture)
- Cycle-accurate simulation of a spatial micro-engine CPU core.
- Front-End Pipeline Assembler parses instructions and reconfigures the ME fabric.
- Point-to-point Elastic Channels manage backpressure (valid/ready handshakes).
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

    def is_full(self):
        return len(self.buffer) >= self.capacity

    def is_empty(self):
        return len(self.buffer) == 0

    def push(self, val):
        if not self.is_full():
            self.buffer.append(val)
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
    """Generic Reconfigurable Micro-Engine Core Slice."""
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

        elif self.type == METype.MAC:
            # Pop input, apply Multiply-Accumulate, push output
            if self.in_channel and not self.in_channel.is_empty():
                if self.out_channel and not self.out_channel.is_full():
                    val = self.in_channel.pop()
                    res = (val * self.weight) + self.bias
                    self.out_channel.push(res)
                    self.processed_count += 1

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

        elif self.type == METype.WRITE_AGU:
            # Pop input stream, write out to TCM memory with striding
            if self.in_channel and not self.in_channel.is_empty():
                val = self.in_channel.pop()
                memory.write(self.curr_addr, val)
                self.curr_addr += self.stride
                self.processed_count += 1


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
# 6. Top-Level Spatial CPU Core
# =====================================================================
class SpatialCPUCore:
    """Replacement CPU Core featuring Spatial Dataflow Execution."""
    def __init__(self):
        self.memory = TightlyCoupledMemory(size_words=1024)
        self.fabric = MEFabric(num_mes=4)
        self.assembler = PipelineAssembler(self.fabric)
        self.pc = 0

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


# =====================================================================
# 7. Verification Workload Run
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

    # Execute Program on Spatial Core
    total_cycles = cpu.run_spatial_program(spatial_program)

    # Read back and display results from output memory
    output_results = [cpu.memory.read(out_base + (i * 2)) for i in range(len(input_tensor))]

    print("=== Execution Results Verification ===")
    print(f"Input Stream  (Address 0x100): {input_tensor}")
    print(f"Output Stream (Address 0x200): {output_results}")