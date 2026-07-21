# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import inspect
import re

import pytest

import cuda.tile as ct
import cuda.tile._bytecode as bc
from cuda.tile._bytecode.basic import StringTable
from cuda.tile._bytecode.debug_info import DebugAttrTable
from cuda.tile._cext import CallingConvention
from cuda.tile._compile import compile_tile
from cuda.tile._exception import FunctionDesc, Loc
from cuda.tile._ir.control_flow_ops import Return
from cuda.tile._ir2bytecode import DebugAttrMap, create_synthetic_linkage_name
from cuda.tile.compilation import ArrayConstraint, KernelSignature

_SPEC_ID = re.compile(r"^s\d+$")


def _array(dtype=ct.int32):
    """Default 1-D ArrayConstraint used by the test kernels."""
    return ArrayConstraint(dtype=dtype, ndim=1, index_dtype=ct.int32,
                           stride_lower_bound_incl=0, alias_groups=(),
                           may_alias_internally=False)


def _compile(fn, constraints=None, symbol="kern"):
    """Compile `fn` through compile_tile and return the final IR block."""
    sig = KernelSignature(constraints or [_array()],
                          CallingConvention.cutile_python_v1(), symbol=symbol)
    return compile_tile(fn, [sig], return_final_ir=True,
                        return_cubin=False).final_ir[0]


def _all_descs(body):
    """Distinct FunctionDescs reachable from any op.loc + its call_site chain."""
    seen, ids = [], set()
    for op in body.traverse():
        loc = op.loc
        while loc is not None:
            fd = loc.function
            if fd is not None and id(fd) not in ids:
                ids.add(id(fd))
                seen.append(fd)
            loc = loc.call_site
    return seen


def _line_col(fn, snippet):
    """1-based (line, column) of `snippet` in `fn`'s source."""
    for i, line in enumerate(inspect.getsource(fn).splitlines()):
        col = line.find(snippet)
        if col >= 0:
            return fn.__code__.co_firstlineno + i, col + 1
    raise ValueError(snippet)


def test_lines_and_columns_are_1_based_for_every_callable_kind():
    # Top-level function, nested function, lambda, and closure-with-capture:
    # each FunctionDesc must record the source position of its *own* definition.
    def kernel(x):
        def inner(p): return p + 1
        f = lambda q: q + 2  # noqa: E731
        n = 3
        def add_n(p): return p + n
        t = ct.load(x, (0,), (1,))
        ct.store(x, (1,), inner(t) + f(t) + add_n(t))

    by_name = {d.name: d for d in _all_descs(_compile(kernel))}
    for name in ("kernel", "inner", "add_n"):
        assert (by_name[name].line, by_name[name].column) == _line_col(kernel, f"def {name}")
    [lam] = [d for d in by_name.values() if d.name is None]
    assert (lam.line, lam.column) == _line_col(kernel, "lambda ")

    # Only the kernel entry is flagged; it skips concretization.
    assert by_name["kernel"].is_entry
    assert by_name["kernel"].specialization_id is None
    for fd in (by_name["inner"], by_name["add_n"], lam):
        assert not fd.is_entry
        assert fd.specialization_id is not None and _SPEC_ID.match(fd.specialization_id)


def test_synthetic_linkage_name_format_and_assertion():
    assert (create_synthetic_linkage_name(FunctionDesc("helper", "/a/some_file.py", 42, 7, "abc"))
            == "helper@some_file:42:7_abc")
    assert (create_synthetic_linkage_name(FunctionDesc(None, "/a/other.py", 11, 3, "xy"))
            == "lambda@other:11:3_xy")
    # Non-identifier chars in the basename are scrubbed to '_'.
    assert (create_synthetic_linkage_name(FunctionDesc("f", "/p/weird-file.name.py", 1, 0, "00"))
            == "f@weird_file_name:1:0_00")
    # Refuses any desc that wasn't concretized by hir2ir.
    with pytest.raises(AssertionError, match="specialization_id"):
        create_synthetic_linkage_name(FunctionDesc("helper", "x.py", 1, 0))


def test_get_subprogram_enforces_id_invariant():
    # The bytecode-emission funnel checks: is_entry iff specialization_id is
    # None. Both directions must trip the assertion.
    m = DebugAttrMap(DebugAttrTable(StringTable()), entry_symbol="kern", anonymize=False)

    # Non-entry without an id -- a hir2ir-side concretization bug.
    bad_helper = FunctionDesc("h", "x.py", 1, 1)
    with pytest.raises(AssertionError, match="FunctionDesc invariant"):
        m.get_subprogram(bad_helper)

    # Entry with an id -- a misuse of is_entry / specialization_id.
    bad_entry = FunctionDesc("k", "x.py", 1, 1, specialization_id="abc", is_entry=True)
    with pytest.raises(AssertionError, match="FunctionDesc invariant"):
        m.get_subprogram(bad_entry)

    # Well-formed entry and helper both go through.
    m.get_subprogram(FunctionDesc("k", "x.py", 1, 1, is_entry=True))
    m.get_subprogram(FunctionDesc("h", "x.py", 1, 1, specialization_id="abc"))


def helper(a): return a + 1
def outer(x): return helper((lambda y: y * 2)(x))


def test_compile_emits_unique_linkages_with_correct_call_site_chain():
    # Repeated calls, type polymorphism, nested helpers, and a lambda coexist;
    # every emitted linkage is unique and the entry keeps its visible symbol.
    def kernel(x, y):
        f = lambda q: q + 1  # noqa: E731
        t_i = ct.load(x, (0,), (1,))
        t_f = ct.load(y, (0,), (1,))
        a = helper(t_i)                                   # repeated...
        b = helper(t_i)                                   # ...must split
        c = helper(t_f)                                   # different dtype
        d = outer(t_i)                                    # nested inlining
        ct.store(x, (1,), a + b + d + f(t_i))
        ct.store(y, (1,), c)

    body = _compile(kernel, [_array(ct.int32), _array(ct.float32)], symbol="kern_v1")
    m = DebugAttrMap(DebugAttrTable(StringTable()), entry_symbol="kern_v1",
                     anonymize=False)
    descs = _all_descs(body)
    linkages = {id(d): m._linkage_for(d) for d in descs}

    [entry] = [d for d in descs if d.is_entry]                       # exactly one
    assert linkages[id(entry)] == "kern_v1"
    for d in descs:
        if not d.is_entry:
            assert linkages[id(d)] == create_synthetic_linkage_name(d)
    assert len(set(linkages.values())) == len(linkages)              # all unique

    helpers = [d for d in descs if d.name == "helper"]
    assert len(helpers) >= 3                                          # no dedup
    assert len({d.specialization_id for d in helpers}) == len(helpers)

    # At least one helper op was inlined via outer (call_site chain).
    assert any(op.loc.call_site is not None
               and op.loc.call_site.function is not None
               and op.loc.call_site.function.name == "outer"
               for op in body.traverse()
               if op.loc.function is not None and op.loc.function.name == "helper")


def test_op_locs_resolve_to_correct_inlining_frame():
    # The DWARF property end-users depend on: an op produced from a line inside
    # an inlined helper must carry that helper-internal source position on
    # `op.loc`, and its `call_site` chain must walk back through the actual
    # caller lines — kernel call -> helper body line, not the other way around.
    def add_marker(a):
        return a + 12345                # ← unique constant identifies the op

    def kernel(x):
        t = ct.load(x, (0,), (1,))
        result = add_marker(t)          # the call site whose line we expect
        ct.store(x, (1,), result)

    body = _compile(kernel)
    body_line, _ = _line_col(add_marker, "return a + 12345")
    call_line, _ = _line_col(kernel, "add_marker(t)")
    kernel_line = kernel.__code__.co_firstlineno

    inside_helper = [op for op in body.traverse()
                     if op.loc.function is not None and op.loc.function.name == "add_marker"]
    assert inside_helper, "expected at least one op attributed to add_marker"

    for op in inside_helper:
        # The op's own loc points at the helper's body line, not the call site.
        assert op.loc.line == body_line, (op.loc.line, body_line)
        # The call_site frame is the kernel's `add_marker(t)` line.
        assert op.loc.call_site is not None
        assert op.loc.call_site.line == call_line
        assert op.loc.call_site.function is not None
        assert op.loc.call_site.function.name == "kernel"
        # And there's nothing above the kernel — it's the outermost frame.
        assert op.loc.call_site.call_site is None
        # Function descs at the two frames are distinct: helper is concretized
        # (has a specialization_id), entry is flagged via is_entry.
        assert op.loc.function is not op.loc.call_site.function
        assert op.loc.function.specialization_id is not None
        assert not op.loc.function.is_entry
        assert op.loc.call_site.function.is_entry

    # And the kernel-entry desc has its own source line, not the helper's.
    entry = body.loc.function
    assert entry is not None
    assert entry.line == kernel_line and entry.line != body_line


def test_dummy_return_sets_unknown_loc():
    def kernel(x):
        t = ct.load(x, (0,), (1,))
        ct.store(x, (1,), t)

    body = _compile(kernel)
    return_ops = [op for op in body.traverse() if isinstance(op, Return)]
    assert len(return_ops) == 1, "expected one Return op"
    assert return_ops[0].loc.is_unknown()


def test_explicit_return_keeps_loc():
    def kernel(x):
        t = ct.load(x, (0,), (1,))
        ct.store(x, (1,), t)
        return None

    body = _compile(kernel)
    return_line, _ = _line_col(kernel, "return None")
    return_ops = [op for op in body.traverse() if isinstance(op, Return)]
    assert len(return_ops) == 1, "expected one Return op"
    return_op = return_ops[0]
    assert return_op, "expected a Return op"
    assert return_op.loc.line == return_line, (return_op.loc.line, return_line)
    assert return_op.loc.function is not None
    assert return_op.loc.function.name == "kernel"


def test_linkage_names_appear_in_emitted_bytecode():
    # End-to-end smoke check: every linkage name we compute should also be
    # serialized into the bytecode's string table. This catches regressions
    # where DebugAttrMap and the bytecode writer disagree about what should be
    # emitted (e.g. a refactor accidentally bypassing the writer path).
    def kernel(x):
        t = ct.load(x, (0,), (1,))
        ct.store(x, (1,), helper(t) + outer(t))

    sig = KernelSignature([_array()], CallingConvention.cutile_python_v1(),
                          symbol="bc_test_kernel")
    result = compile_tile(kernel, [sig], return_final_ir=True,
                          return_bytecode=True, return_cubin=False, sm_arch="sm_90")
    [body] = result.final_ir
    bytecode = bytes(result.bytecode)

    # The entry symbol must appear as-is in the bytecode.
    assert b"bc_test_kernel" in bytecode

    # Every non-entry FunctionDesc in the IR must have its synthetic linkage
    # name present in the bytecode's string table.
    for fd in _all_descs(body):
        if fd.is_entry:
            continue
        expected = create_synthetic_linkage_name(fd).encode()
        assert expected in bytecode, f"missing linkage {expected!r}"

    # And we mustn't have leaked an unconcretized form of any helper. Every
    # "<name>@<stem>:<line>:<col>" occurrence must carry an "_s<N>" suffix.
    SYNTH_RE = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*@[A-Za-z0-9_]+:\d+:\d+(_s\d+)?")
    for m in SYNTH_RE.finditer(bytecode):
        assert (
            m.group(1) is not None
        ), f"unconcretized linkage leaked into bytecode: {m.group(0)!r}"


def test_simple_function_desc_specialization_ids_are_deterministic_across_compiles():
    # Sequential specialization ids make the emitted bytecode byte-identical across
    # compiles of the same kernel — required for the compiler cache to hit.
    # This is just a sanity check that the specialization ids are deterministic
    # and is not a comprehensive test of the bytecode determinism.
    def kernel(x):
        t = ct.load(x, (0,), (1,))
        ct.store(x, (1,), helper(t) + outer(t))

    def compile_once():
        sig = KernelSignature(
            [_array()], CallingConvention.cutile_python_v1(), symbol="det_kernel"
        )
        return bytes(
            compile_tile(
                kernel, [sig], return_bytecode=True, return_cubin=False, sm_arch="sm_90"
            ).bytecode
        )

    assert compile_once() == compile_once()


def test_unknown_loc_maps_to_missing_debug_attr():
    m = DebugAttrMap(DebugAttrTable(StringTable()), entry_symbol="kern", anonymize=False)
    assert m.get_debugattr(Loc.unknown()) == bc.MISSING_DEBUG_ATTR_ID
