import re
import torch

class MyChipHardwareSimulator:
    def __init__(self):
        # Hardware register file / memory map tracking
        self.memory_map = {}

    def bind_buffer(self, ssa_name, tensor_data):
        """Maps an MLIR SSA variable name (e.g., '%0') to an active tensor allocation."""
        clean_name = ssa_name.strip()
        self.memory_map[clean_name] = tensor_data.clone()

    def get_buffer(self, ssa_name):
        clean_name = ssa_name.strip()
        if clean_name not in self.memory_map:
            raise ValueError(f"Hardware Error: Accessing unallocated memory pointer {clean_name}")
        return self.memory_map[clean_name]

    def resolve_buffer(self, ptr_line):
        """
        Dynamically resolves a raw PTR definition line from commands.txt to a tensor.

        Handles three cases:
          1. The primary SSA name is already bound in memory_map  -> return it directly.
          2. The line contains a cast (to_tensor / to_buffer)     -> alias the operand variable.
          3. The line contains memref.alloc()                     -> auto-allocate a zero tensor
             of the correct shape by parsing the memref<...> type.
        """
        # Extract the primary (defined) SSA variable — always the first %name on the line
        defined_var_match = re.search(r'(%\w+)', ptr_line)
        if not defined_var_match:
            raise ValueError(f"Could not find SSA name in PTR line: {ptr_line}")
        defined_var = defined_var_match.group(1)

        # 1. Already bound — fast path
        if defined_var in self.memory_map:
            return self.memory_map[defined_var]

        # 2. Cast alias — to_tensor / to_buffer
        if 'to_tensor' in ptr_line or 'to_buffer' in ptr_line:
            all_vars = re.findall(r'(%\w+)', ptr_line)
            if len(all_vars) >= 2:
                alias_var = all_vars[1]  # operand of the cast
                buf = self.get_buffer(alias_var)
                self.memory_map[defined_var] = buf  # cache alias
                return buf

        # 3. Allocation — parse shape from memref<DxDx...xdtype>
        if 'memref.alloc' in ptr_line:
            shape_match = re.search(r'memref<([\d]+(?:x[\d]+)*)x\w+>', ptr_line)
            if shape_match:
                dims = [int(d) for d in shape_match.group(1).split('x')]
                allocated = torch.zeros(*dims, dtype=torch.float32)
                self.memory_map[defined_var] = allocated
                return allocated

        raise ValueError(
            f"Hardware Error: Cannot resolve memory pointer '{defined_var}' from line: {ptr_line}"
        )

    def execute_commands(self, commands_file_path):
        """Parses commands.txt and models the internal hardware execution graph."""
        with open(commands_file_path, 'r') as f:
            content = f.read()

        # Find all individual tile processing blocks
        commands = re.findall(
            r'# --- BEGIN MY_CHIP EXECUTION COMMAND ---\n(.*?)\n# --- END COMMAND ---',
            content, re.DOTALL
        )

        PARAM_PREFIXES = ('CMD_TYPE:', 'REG_LHS_PTR:', 'REG_RHS_PTR:', 'REG_DEST_PTR:')

        for cmd_block in commands:
            # Reconstruct line-wrapped fragments: lines that do NOT start with a known
            # parameter key are continuations of the previous line.
            cleaned_lines = []
            for raw_line in cmd_block.split('\n'):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                if any(stripped.startswith(p) for p in PARAM_PREFIXES):
                    cleaned_lines.append(stripped)
                else:
                    # Continuation of the previous line (MLIR line wrapping)
                    if cleaned_lines:
                        cleaned_lines[-1] += ' ' + stripped

            cmd_data = {}
            for line in cleaned_lines:
                key, val = line.split(':', 1)
                cmd_data[key.strip()] = val.strip()

            if cmd_data.get('CMD_TYPE') == 'EXEC_TILE_MATMUL':
                self._sim_matmul_tile(
                    lhs_line=cmd_data.get('REG_LHS_PTR', ''),
                    rhs_line=cmd_data.get('REG_RHS_PTR', ''),
                    dest_line=cmd_data.get('REG_DEST_PTR', '')
                )
            else:
                raise NotImplementedError(
                    f"Unsupported ISA hardware token: {cmd_data.get('CMD_TYPE')}"
                )

    def _sim_matmul_tile(self, lhs_line, rhs_line, dest_line):
        """Models the localized systolic array behavior / tile computation loop."""
        lhs  = self.resolve_buffer(lhs_line)
        rhs  = self.resolve_buffer(rhs_line)
        dest = self.resolve_buffer(dest_line)   # auto-allocates if not yet bound

        # Hardware behavioral execution — mirrors the systolic array matmul
        result = torch.matmul(lhs, rhs)

        # Write back to the physical destination register
        dest.copy_(result)

        # Ensure the result is visible under the dest variable name
        dest_var = re.search(r'(%\w+)', dest_line).group(1)
        self.memory_map[dest_var] = dest
