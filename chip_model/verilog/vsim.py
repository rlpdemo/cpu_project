"""Minimal pure-Python interpreter for a well-defined subset of Verilog.

Executes the *exact* generated RTL text (``streaming_cpu_top.v`` /
``streaming_cpu_top_tb.v``) so the RTL can be validated even when no HDL
simulator (iverilog) is installed.  ``run_streaming_equivalence.py`` prefers
iverilog when present and cross-checks this interpreter against it.

Supported subset (everything the generator emits):
  * ``module`` with SystemVerilog-style ``input``/``output`` port lists,
  * ``real``/``integer``/``reg`` scalars, 1-D and 2-D ``real`` arrays,
  * ``initial begin..end`` and ``always @(posedge <sig>) begin..end``,
  * blocking ``=`` and non-blocking ``<=`` assignments (NBA applied at the
    end of the time step), ``if/else``, ``for``, ``while``, ``repeat``,
    ``begin/end``, ``#<delay>``,
  * module instantiation with named ports (``.port(sig)``),
  * ``$display`` (``%f``/``%0.Nf``/``%d``/``%s``/``%%``), ``$finish``,
    ``$abs()``, and ``~`` on 0/1-valued signals.

Semantics follow IEEE-1364 for this subset: non-blocking assignments update at
the end of the current time step, blocking assignments take effect immediately,
``always @(posedge clk)`` re-runs on every rising edge, and delays suspend the
thread.
"""

import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
_OPS2 = ("<=", "==", "!=", "&&", "||", ">=")
_TYPE_KEYS = ("input", "output", "inout", "reg", "wire", "real", "integer",
              "parameter")


class Token:
    __slots__ = ("kind", "text")
    kind: str  # 'ID' | 'NUM' | 'STR' | 'OP'
    text: str

    def __init__(self, kind: str, text: str):
        self.kind = kind
        self.text = text

    def __repr__(self):
        return f"<{self.kind} {self.text}>"


def tokenize(src: str) -> List[Token]:
    toks: List[Token] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 1
            toks.append(Token("STR", src[i + 1:j]))
            i = j + 1
            continue
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            j = i
            while j < n and (src[j].isalnum() or src[j] in "._"):
                j += 1
            toks.append(Token("NUM", src[i:j]))
            i = j
            continue
        if c.isalpha() or c in "_$":
            j = i
            while j < n and (src[j].isalnum() or src[j] in "_$"):
                j += 1
            toks.append(Token("ID", src[i:j]))
            i = j
            continue
        two = src[i:i + 2]
        if two in _OPS2:
            toks.append(Token("OP", two))
            i += 2
            continue
        toks.append(Token("OP", c))
        i += 1
    return toks


# ---------------------------------------------------------------------------
# Parser -> AST
# ---------------------------------------------------------------------------
class Parser:
    def __init__(self, toks: List[Token]):
        self.toks = toks
        self.pos = 0

    def peek(self) -> Optional[Token]:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def nxt(self) -> Token:
        t = self.toks[self.pos]
        self.pos += 1
        return t

    def is_op(self, op: str) -> bool:
        t = self.peek()
        return t is not None and t.kind == "OP" and t.text == op

    def expect_op(self, op: str) -> Token:
        t = self.nxt()
        if t.kind != "OP" or t.text != op:
            raise SyntaxError(f"expected '{op}', got {t}")
        return t

    def expect_id(self) -> str:
        t = self.nxt()
        if t.kind != "ID":
            raise SyntaxError(f"expected identifier, got {t}")
        return t.text

    # --- expressions ------------------------------------------------------
    def parse_expr(self):
        return self.parse_or()

    def parse_or(self):
        e = self.parse_and()
        while self.is_op("||"):
            self.nxt()
            e = ("binop", "||", e, self.parse_and())
        return e

    def parse_and(self):
        e = self.parse_eq()
        while self.is_op("&&"):
            self.nxt()
            e = ("binop", "&&", e, self.parse_eq())
        return e

    def parse_eq(self):
        e = self.parse_rel()
        while True:
            if self.is_op("==") or self.is_op("!="):
                op = self.nxt().text
                e = ("binop", op, e, self.parse_rel())
            else:
                return e

    def parse_rel(self):
        e = self.parse_add()
        while True:
            t = self.peek()
            if t is not None and t.kind == "OP" and t.text in ("<", ">", "<=", ">="):
                op = self.nxt().text
                e = ("binop", op, e, self.parse_add())
            else:
                return e

    def parse_add(self):
        e = self.parse_mul()
        while True:
            t = self.peek()
            if t is not None and t.kind == "OP" and t.text in ("+", "-"):
                op = self.nxt().text
                e = ("binop", op, e, self.parse_mul())
            else:
                return e

    def parse_mul(self):
        e = self.parse_unary()
        while True:
            t = self.peek()
            if t is not None and t.kind == "OP" and t.text in ("*", "/"):
                op = self.nxt().text
                e = ("binop", op, e, self.parse_unary())
            else:
                return e

    def parse_unary(self):
        t = self.peek()
        if t is not None and t.kind == "OP" and t.text in ("-", "!", "~"):
            self.nxt()
            return ("unop", t.text, self.parse_unary())
        return self.parse_primary()

    def parse_primary(self):
        t = self.peek()
        if t is None:
            raise SyntaxError("unexpected end of input in expression")
        if t.kind == "OP" and t.text == "(":
            self.nxt()
            e = self.parse_expr()
            self.expect_op(")")
            return e
        if t.kind == "NUM":
            self.nxt()
            txt = t.text.replace("_", "")
            return ("num", float(txt) if ("." in txt) else int(txt))
        if t.kind == "ID":
            self.nxt()
            name = t.text
            if name == "$abs":
                self.expect_op("(")
                arg = self.parse_expr()
                self.expect_op(")")
                return ("call", "$abs", [arg])
            idx = []
            while self.is_op("["):
                self.nxt()
                idx.append(self.parse_expr())
                self.expect_op("]")
            if idx:
                return ("arr", name, idx)
            return ("var", name)
        if t.kind == "STR":
            raise SyntaxError("string not allowed in expression")
        raise SyntaxError(f"unexpected token in expression: {t}")

    # --- targets (LHS of assignments) --------------------------------------
    def parse_target(self):
        name = self.expect_id()
        idx = []
        while self.is_op("["):
            self.nxt()
            idx.append(self.parse_expr())
            self.expect_op("]")
        if idx:
            return ("arr", name, idx)
        return ("var", name)

    # --- declarations -------------------------------------------------------
    def parse_decl(self, expect_semi: bool) -> Tuple:
        quals: List[str] = []
        while True:
            t = self.peek()
            if t is not None and t.kind == "ID" and t.text in _TYPE_KEYS:
                quals.append(self.nxt().text)
            else:
                break
        name = self.expect_id()
        dims = []
        while self.is_op("["):
            self.nxt()
            lo = self.parse_expr()
            self.expect_op(":")
            hi = self.parse_expr()
            self.expect_op("]")
            dims.append((lo, hi))
        init = None
        if self.is_op("="):
            self.nxt()
            init = self.parse_expr()
        if expect_semi:
            self.expect_op(";")
        return ("decl", quals, name, dims, init)

    # --- statements ---------------------------------------------------------
    def parse_stmt(self):
        t = self.peek()
        if t is None:
            raise SyntaxError("unexpected end of file in statement")
        if t.kind == "ID" and t.text == "begin":
            self.nxt()
            stmts = []
            while not (self.peek() is not None and
                       self.peek().kind == "ID" and self.peek().text == "end"):
                stmts.append(self.parse_stmt())
            self.expect_id()  # 'end'
            return ("block", stmts)
        if t.kind == "ID" and t.text == "if":
            self.nxt()
            self.expect_op("(")
            cond = self.parse_expr()
            self.expect_op(")")
            then_s = self.parse_stmt()
            else_s = None
            if self.peek() is not None and self.peek().kind == "ID" and \
                    self.peek().text == "else":
                self.nxt()
                else_s = self.parse_stmt()
            return ("if", cond, then_s, else_s)
        if t.kind == "ID" and t.text == "for":
            self.nxt()
            self.expect_op("(")
            init = self.parse_simple_assign(expect_semi=False)
            self.expect_op(";")
            cond = self.parse_expr()
            self.expect_op(";")
            upd = self.parse_simple_assign(expect_semi=False)
            self.expect_op(")")
            body = self.parse_stmt()
            return ("for", init, cond, upd, body)
        if t.kind == "ID" and t.text == "while":
            self.nxt()
            self.expect_op("(")
            cond = self.parse_expr()
            self.expect_op(")")
            return ("while", cond, self.parse_stmt())
        if t.kind == "ID" and t.text == "repeat":
            self.nxt()
            self.expect_op("(")
            count = self.parse_expr()
            self.expect_op(")")
            return ("repeat", count, self.parse_stmt())
        if self.is_op("#"):
            self.nxt()
            d = self.parse_expr()
            if self.is_op(";"):
                self.nxt()
                return ("delay", d)
            return ("delay_stmt", d, self.parse_stmt())
        if t.kind == "ID" and t.text in ("$display", "$finish"):
            name = self.nxt().text
            if name == "$finish":
                self.expect_op(";")
                return ("finish",)
            self.expect_op("(")
            fmt = self.expect_id() if self.peek().kind == "ID" else self.peek()
            # format string is a STR token
            fs = self.nxt()
            if fs.kind != "STR":
                raise SyntaxError("$display format must be a string")
            args = []
            while not self.is_op(")"):
                if self.is_op(","):
                    self.nxt()
                args.append(self.parse_expr())
            self.expect_op(")")
            self.expect_op(";")
            return ("display", fs.text, args)
        if self.is_op(";"):
            self.nxt()
            return ("nop",)
        # assignment
        return self.parse_simple_assign(expect_semi=True)
        raise SyntaxError(f"cannot parse statement starting with {t}")

    def parse_simple_assign(self, expect_semi: bool):
        target = self.parse_target()
        t = self.peek()
        if t is not None and t.kind == "OP" and t.text in ("=", "<="):
            op = self.nxt().text
            rhs = self.parse_expr()
            if expect_semi:
                self.expect_op(";")
            return ("assign", target, op == "<=", rhs)
        raise SyntaxError(f"cannot parse assignment starting with {t}")

    # --- module ---------------------------------------------------------------
    def parse_program(self) -> Dict[str, dict]:
        modules: Dict[str, dict] = {}
        while self.peek() is not None:
            t = self.peek()
            if t.kind == "OP" and t.text == "`":
                # backtick directive (e.g. `timescale 1ns/1ps)
                self.nxt()                      # '`'
                if self.peek() is not None and self.peek().kind == "ID":
                    self.nxt()                  # directive name (timescale)
                while self.peek() is not None and self.peek().kind in ("NUM", "OP"):
                    self.nxt()                  # 1ns / 1ps ...
                continue
            if t.kind == "ID" and t.text == "module":
                self.nxt()
                name = self.expect_id()
                ports: List[Tuple] = []
                if self.is_op("#"):
                    self.nxt()
                    self.expect_op("(")
                    while not self.is_op(")"):
                        self.parse_decl(expect_semi=False)
                        if self.is_op(","):
                            self.nxt()
                    self.expect_op(")")
                if self.is_op("("):
                    self.nxt()
                    while not self.is_op(")"):
                        ports.append(self.parse_decl(expect_semi=False))
                        if self.is_op(","):
                            self.nxt()
                    self.expect_op(")")
                self.expect_op(";")
                items = []
                while not (self.peek() is not None and self.peek().kind == "ID"
                           and self.peek().text == "endmodule"):
                    items.append(self.parse_item())
                self.expect_id()  # 'endmodule'
                modules[name] = {"ports": ports, "items": items}
            else:
                raise SyntaxError(f"unexpected token at top level: {t}")
        return modules

    def parse_item(self):
        t = self.peek()
        if t is None:
            raise SyntaxError("unexpected end of file in module")
        if t.kind == "ID" and t.text == "initial":
            self.nxt()
            return ("initial", self.parse_stmt())
        if t.kind == "ID" and t.text == "always":
            self.nxt()
            self.expect_op("@")
            self.expect_op("(")
            self.expect_id()  # 'posedge' / 'negedge'
            sig = self.expect_id()
            self.expect_op(")")
            return ("always", sig, self.parse_stmt())
        if t.kind == "ID" and t.text in _TYPE_KEYS:
            return self.parse_decl(expect_semi=True)
        # module instantiation
        modname = self.expect_id()
        instname = self.expect_id()
        conns = []
        if self.is_op("#"):
            self.nxt()
            self.expect_op("(")
            while not self.is_op(")"):
                # parameter override -- evaluate and ignore (generator emits none)
                if self.peek().kind == "OP" and self.peek().text == ".":
                    self.nxt()
                    self.expect_id()
                    self.expect_op("(")
                    self.parse_expr()
                    self.expect_op(")")
                else:
                    self.parse_expr()
                if self.is_op(","):
                    self.nxt()
            self.expect_op(")")
        self.expect_op("(")
        while not self.is_op(")"):
            self.expect_op(".")
            port = self.expect_id()
            self.expect_op("(")
            sig = self.parse_expr()
            self.expect_op(")")
            conns.append((port, sig))
            if self.is_op(","):
                self.nxt()
        self.expect_op(")")
        self.expect_op(";")
        return ("inst", modname, instname, conns)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
class Var:
    __slots__ = ("name", "vtype", "value")

    def __init__(self, name: str, vtype: str, value=None):
        self.name = name
        self.vtype = vtype
        self.value = 0 if vtype == "int" else 0.0
        if value is not None:
            self.value = value


class ArrayVar:
    __slots__ = ("name", "vtype", "data")

    def __init__(self, name: str, vtype: str):
        self.name = name
        self.vtype = vtype
        self.data: Dict[Tuple[int, ...], object] = {}

    def get(self, idxs: Tuple[int, ...]):
        return self.data.get(idxs, 0.0 if self.vtype == "real" else 0)

    def put(self, idxs: Tuple[int, ...], val):
        self.data[idxs] = val


class Instance:
    __slots__ = ("name", "module", "parent", "ns")

    def __init__(self, name: str, module: dict, parent):
        self.name = name
        self.module = module
        self.parent = parent
        self.ns: Dict[str, object] = {}


class Sim:
    def __init__(self):
        self.stdout: List[str] = []
        self.nba: List[Tuple] = []
        self.finished = False
        self.time = 0.0
        self.modules: Dict[str, dict] = {}
        self.initials: List[Tuple[Instance, object]] = []
        self.always: List[Tuple[Instance, str, object]] = []
        self.trace = False


def eval_expr(e, inst: Instance):
    k = e[0]
    if k == "num":
        return e[1]
    if k == "var":
        return inst.ns[e[1]].value
    if k == "arr":
        v = inst.ns[e[1]]
        idxs = tuple(int(eval_expr(x, inst)) for x in e[2])
        return v.get(idxs)
    if k == "call":
        args = [eval_expr(a, inst) for a in e[2]]
        if e[1] == "$abs":
            return abs(args[0])
        raise RuntimeError(f"unknown function {e[1]}")
    if k == "unop":
        v = eval_expr(e[2], inst)
        op = e[1]
        if op == "-":
            return -v
        if op == "!":
            return 1 if not v else 0
        if op == "~":
            return 1 - v if v in (0, 1) else ~int(v)
        raise RuntimeError(f"unknown unop {op}")
    if k == "binop":
        l = eval_expr(e[2], inst)
        r = eval_expr(e[3], inst)
        op = e[1]
        if op == "+":
            return l + r
        if op == "-":
            return l - r
        if op == "*":
            return l * r
        if op == "/":
            return l / r
        if op == "<":
            return 1 if l < r else 0
        if op == ">":
            return 1 if l > r else 0
        if op == "<=":
            return 1 if l <= r else 0
        if op == ">=":
            return 1 if l >= r else 0
        if op == "==":
            return 1 if l == r else 0
        if op == "!=":
            return 1 if l != r else 0
        if op == "&&":
            return 1 if (l and r) else 0
        if op == "||":
            return 1 if (l or r) else 0
        raise RuntimeError(f"unknown binop {op}")
    raise RuntimeError(f"bad expr {e}")


def assign_to(target, value, inst: Instance):
    if target[0] == "var":
        inst.ns[target[1]].value = value
    else:
        v = inst.ns[target[1]]
        idxs = tuple(int(eval_expr(x, inst)) for x in target[2])
        v.put(idxs, value)


def resolve_target(target, inst: Instance):
    """Return a copy of an assignment target with array indices frozen to
    their current values (needed for NBA, whose LHS indices must be sampled
    when the statement executes, not when the update is applied)."""
    if target[0] == "var":
        return target
    idxs = tuple(("num", float(eval_expr(x, inst))) for x in target[2])
    return (target[0], target[1], idxs)


def decl_vartype(quals: Sequence[str]) -> str:
    return "real" if "real" in quals else "int"


def elaborate(inst: Instance, sim: Sim, make_ports: bool = True):
    mod = inst.module
    if make_ports:
        for d in mod["ports"]:
            _make_decl(inst, d)
    for item in mod["items"]:
        k = item[0]
        if k == "decl":
            _make_decl(inst, item)
        elif k == "initial":
            sim.initials.append((inst, item[1]))
        elif k == "always":
            sim.always.append((inst, item[1], item[2]))
        elif k == "inst":
            child = Instance(item[2], sim.modules[item[1]], inst)
            for d in child.module["ports"]:
                _make_decl(child, d)
            for port, sig in item[3]:
                if sig[0] != "var":
                    raise RuntimeError(
                        f"unsupported port connection for .{port}: {sig}")
                child.ns[port] = inst.ns[sig[1]]
            elaborate(child, sim, make_ports=False)


def _make_decl(inst: Instance, d: Tuple):
    _, quals, name, dims, init = d
    vtype = decl_vartype(quals)
    if dims:
        v = ArrayVar(name, vtype)
        inst.ns[name] = v
        if init is not None:
            raise RuntimeError("array decl with initializer unsupported")
    else:
        v = Var(name, vtype)
        if init is not None:
            v.value = eval_expr(init, inst)
        inst.ns[name] = v


_FORMAT_RE = re.compile(r"%([0-9.]*)([A-Za-z%])")


def format_display(fmt: str, args: List) -> str:
    out: List[str] = []
    i, ai = 0, 0
    while i < len(fmt):
        c = fmt[i]
        if c == "%":
            m = _FORMAT_RE.match(fmt, i)
            if m:
                spec, conv = m.group(1), m.group(2)
                if conv == "%":
                    out.append("%")
                else:
                    arg = args[ai]
                    ai += 1
                    if conv == "f":
                        prec = int(spec.split(".")[1]) if "." in spec else 6
                        out.append(f"{arg:.{prec}f}")
                    elif conv == "e":
                        prec = int(spec.split(".")[1]) if "." in spec else 6
                        out.append(f"{arg:.{prec}e}")
                    elif conv == "d":
                        out.append(str(int(arg)))
                    elif conv == "s":
                        out.append(str(arg))
                    else:
                        raise RuntimeError(f"unsupported $display conversion %{conv}")
                i = m.end()
                continue
        out.append(c)
        i += 1
    return "".join(out)


def gen_stmt(st, inst: Instance, sim: Sim):
    k = st[0]
    if k == "block":
        for s in st[1]:
            yield from gen_stmt(s, inst, sim)
    elif k == "if":
        if eval_expr(st[1], inst):
            yield from gen_stmt(st[2], inst, sim)
        elif st[3] is not None:
            yield from gen_stmt(st[3], inst, sim)
    elif k == "for":
        exec_assign(st[1], inst, sim)           # init
        while eval_expr(st[2], inst):           # cond
            yield from gen_stmt(st[4], inst, sim)  # body
            exec_assign(st[3], inst, sim)       # update
    elif k == "while":
        while eval_expr(st[1], inst):
            yield from gen_stmt(st[2], inst, sim)
    elif k == "repeat":
        n = int(eval_expr(st[1], inst))
        for _ in range(n):
            yield from gen_stmt(st[2], inst, sim)
    elif k == "delay":
        yield ("delay", float(eval_expr(st[1], inst)))
    elif k == "delay_stmt":
        yield ("delay", float(eval_expr(st[1], inst)))
        yield from gen_stmt(st[2], inst, sim)
    elif k == "display":
        fmt = st[1]
        args = [eval_expr(a, inst) for a in st[2]]
        line = format_display(fmt, args)
        sim.stdout.append(line)
        print(line)
    elif k == "finish":
        yield ("finish",)
    elif k == "assign":
        exec_assign(st, inst, sim)
    elif k == "nop":
        pass
    else:
        raise RuntimeError(f"unknown statement {k}")


def exec_assign(a, inst: Instance, sim: Sim):
    """Execute an ('assign', target, is_nba, rhs) node."""
    if a[0] != "assign":
        raise RuntimeError(f"expected assignment node, got {a}")
    value = eval_expr(a[3], inst)
    if a[2]:  # non-blocking
        sim.nba.append((resolve_target(a[1], inst), value, inst))
    else:
        assign_to(a[1], value, inst)


def _run_thread(g, sim: Sim) -> Optional[float]:
    """Run a generator until it suspends or completes. Returns wake time."""
    try:
        ev = next(g)
    except StopIteration:
        return None
    if ev[0] == "delay":
        return sim.time + ev[1]
    if ev[0] == "finish":
        sim.finished = True
        return None
    # unreachable for single-event yields
    return None


def _run_always(g, sim: Sim):
    """Run an always-body generator to completion (no delays allowed)."""
    while True:
        try:
            ev = next(g)
        except StopIteration:
            return
        if ev[0] == "finish":
            sim.finished = True
            return
        if ev[0] == "delay":
            raise RuntimeError("delay inside always @(posedge) not supported")


def simulate(src: str, top: str, max_time: float = 1e7) -> List[str]:
    """Parse ``src`` (all modules) and simulate the testbench ``top``."""
    toks = tokenize(src)
    parser = Parser(toks)
    modules = parser.parse_program()

    sim = Sim()
    sim.modules = modules
    root = Instance(top, modules[top], None)
    elaborate(root, sim)

    active: List[Tuple[float, object]] = [
        (0.0, gen_stmt(block, inst, sim)) for inst, block in sim.initials
    ]

    # snapshot of clk values for posedge detection (only ports named clk)
    def clk_vals() -> Dict[Instance, int]:
        out = {}
        for inst, _sig, _block in sim.always:
            out[inst] = inst.ns[_sig].value
        return out

    sim.time = 0.0
    while active and not sim.finished and sim.time <= max_time:
        now = sim.time
        run_now = [(t, g) for t, g in active if t <= now + 1e-12]
        active = [(t, g) for t, g in active if t > now + 1e-12]
        prev = clk_vals()
        for _t, g in run_now:
            wake = _run_thread(g, sim)
            if wake is not None:
                active.append((wake, g))
        if sim.finished:
            break
        # posedge detection -> run always blocks
        for inst, sig, block in sim.always:
            if prev.get(inst) == 0 and inst.ns[sig].value == 1:
                _run_always(gen_stmt(block, inst, sim), sim)
        # apply non-blocking updates
        for target, value, inst in sim.nba:
            assign_to(target, value, inst)
        sim.nba.clear()
        # next event time
        next_t = sim.time + max_time
        for t, _g in active:
            if t < next_t:
                next_t = t
        if next_t > sim.time:
            sim.time = next_t
    return sim.stdout


def simulate_files(paths: Sequence[str], top: str,
                   max_time: float = 1e7) -> List[str]:
    chunks = []
    for p in paths:
        with open(p) as f:
            chunks.append(f.read())
    return simulate("\n".join(chunks), top, max_time=max_time)


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: vsim.py <top_module> <file.v> [file2.v ...]")
        return 2
    top = sys.argv[1]
    out = simulate_files(sys.argv[2:], top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())