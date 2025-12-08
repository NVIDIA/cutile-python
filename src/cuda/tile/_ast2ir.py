# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import operator
from enum import Enum, auto
from typing import List, Sequence, Set, Optional, Union, Mapping, Any, Dict, Type, Callable

import cuda.tile._stub as ct
from cuda.tile import _datatype as datatype
from cuda.tile._exception import TileSyntaxError
from cuda.tile._ir import ir, ops
from cuda.tile._ir.typing_support import get_constant_value


def get_function_ir(pyfunc: Callable,
                    ir_ctx: ir.IRContext,
                    call_site: Optional[ir.Loc]) -> ir.Function:
    # Get the original function from the decorated function if it exists.
    pyfunc = getattr(pyfunc, "__wrapped__", pyfunc)

    source_lines, first_line = inspect.getsourcelines(pyfunc)
    # The source code of our function could be inside a class, an if-else block etc.
    # This means it can have extra indentation on the left. If we try to give it
    # to ast.parse() as is, we will get a parse error. The common workaround
    # suggested on the web is to filter the source through textwrap.dedent() to remove
    # a common amount of indentation. This is not correct though, because lines that
    # only contain spaces and comments, as well as continuation lines, are not required to
    # be indented. For example, this code is valid:
    #
    #     class A:
    #         def foo(self):
    #              return (100 +
    #     200)
    #
    # The "textwrap.dedent" method would fail to remove the extra indent because the
    # continuation line "200)" is not indented.
    #
    # To handle this properly, we resort to a hack: add one level of indentation to our
    # function and wrap it inside an "if True:" block.
    header_line = "if True:\n "
    indented_source = header_line + " ".join(source_lines)
    mod = ast.parse(indented_source)
    assert len(mod.body) == 1
    assert isinstance(mod.body[0], ast.If)
    assert len(mod.body[0].body) == 1
    func_def = mod.body[0].body[0]
    assert isinstance(func_def, ast.FunctionDef)

    func_globals = dict(pyfunc.__builtins__)
    # Only include user-defined variables from function globals, not builtins or imported modules
    # This prevents malicious __globals__ from overwriting built-in functions like __import__
    safe_globals = {k: v for k, v in pyfunc.__globals__.items()
                    if not (k.startswith('__') and k.endswith('__')) or k == '__name__'}
    func_globals.update(safe_globals)
    # Add closure variables (from freevars)
    if pyfunc.__closure__:
        for name, cell in zip(pyfunc.__code__.co_freevars, pyfunc.__closure__):
            func_globals[name] = cell.cell_contents
    ctx = _Context(inspect.getfile(pyfunc), first_line, func_globals, call_site, pyfunc, ir_ctx)
    assert isinstance(func_def, ast.FunctionDef)
    func = _ast2ir(func_def, ctx)
    return func


# Translate the 1-based line number of the chunk we passed to the AST parser
# to the original 1-based line number in the file.
def _get_source_line_no(first_line_no: int, ast_line_no: int):
    # Why -2?
    #    -1 because both first_line_no and ast_line_no are 1-based;
    #    another -1 to account for the "if True" line that we inserted.
    return first_line_no + ast_line_no - 2


class LoopKind(Enum):
    FOR = auto()
    WHILE = auto()


class _Context:
    def __init__(self, filename: str, first_line: int, globals: Mapping[str, Any],
                 call_site: Optional[ir.Loc], function: Callable,
                 ir_ctx: ir.IRContext):
        self.filename = filename
        self.first_line = first_line
        self.globals = globals  # raw environment from user and builtins
        self.entry_point = call_site is None
        self.call_site = call_site
        self.function = function
        self.parent_loops: List[LoopKind] = []
        self.ir_ctx = ir_ctx

    def get_loc(self, node: ast.AST) -> ir.Loc:
        line_no = _get_source_line_no(self.first_line, node.lineno)
        last_line_no = _get_source_line_no(self.first_line, node.end_lineno)
        # Subtract 1 from the column offset to correct for an extra level
        # of indentation we inserted for the dummy "if True" block.
        return ir.Loc(line_no, node.col_offset - 1, self.filename,
                      last_line_no, node.end_col_offset - 1, self.function,
                      self.call_site)

    def syntax_error(self, loc: Union[ast.AST, ir.Loc], message: str) -> TileSyntaxError:
        if not isinstance(loc, ir.Loc):
            message = f"{message} {type(loc)}"
            loc = self.get_loc(loc)
        return TileSyntaxError(message, loc)

    def unsupported_syntax(self, loc: Union[ast.AST, ir.Loc]) -> TileSyntaxError:
        return self.syntax_error(loc, "Unsupported syntax")


def _register(mapping, klazz):
    def decorate(f):
        mapping[klazz] = f
        return f
    return decorate


# ================================
# Expressions
# ================================
_expr_handlers: Dict[Type[ast.AST], Callable] = {}


@_register(_expr_handlers, ast.Call)
def _call_expr(call: ast.Call, block: ir.Block, ctx: _Context) -> ir.Var:
    callee = _expr(call.func, block, ctx)
    args = tuple(_expr(a, block, ctx) for a in call.args)
    kwargs = tuple((a.arg, _expr(a.value, block, ctx)) for a in call.keywords)
    res = block.make_temp_var(ctx.get_loc(call))
    ops.call(callee, args, kwargs, block, ctx.get_loc(call), res)
    return res


@_register(_expr_handlers, ast.Name)
def _name_expr(name: ast.Name, block: ir.Block, ctx: Any) -> ir.Var:
    if not isinstance(name.ctx, ast.Load):
        raise ctx.unsupported_syntax(name)
    res = block.make_temp_var(ctx.get_loc(name))
    ops.load(name.id, block, ctx.get_loc(name), res)
    return res


_unary_map = {ast.Invert: operator.invert, ast.Not: operator.not_,
              ast.UAdd: operator.pos, ast.USub: operator.neg}


@_register(_expr_handlers, ast.UnaryOp)
def _unary_op(unary: ast.UnaryOp, block: ir.Block, ctx: _Context) -> ir.Var:
    loc = ctx.get_loc(unary)
    op_func = _unary_map.get(type(unary.op))
    if op_func is None:
        raise ctx.unsupported_syntax(loc)

    operand = _expr(unary.operand, block, ctx)

    func_var = block.make_temp_var(loc)
    ops.const(op_func, block, loc, func_var)
    res = block.make_temp_var(loc)
    ops.call(func_var, (operand,), (), block, loc, res)
    return res


_binop_map = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv, ast.Div: operator.truediv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.BitOr: operator.or_, ast.BitXor: operator.xor, ast.BitAnd: operator.and_,
    ast.LShift: operator.lshift, ast.RShift: operator.rshift,
    ast.MatMult: operator.matmul,
}


@_register(_expr_handlers, ast.BinOp)
def _binop_expr(binop: ast.BinOp, block: ir.Block, ctx: _Context) -> ir.Var:
    loc = ctx.get_loc(binop)
    op_func = _binop_map.get(type(binop.op))
    if op_func is None:
        raise ctx.unsupported_syntax(loc)
    res = block.make_temp_var(ctx.get_loc(binop))
    lhs = _expr(binop.left, block, ctx)
    rhs = _expr(binop.right, block, ctx)
    func_var = block.make_temp_var(loc)
    ops.const(op_func, block, loc, func_var)
    ops.call(func_var, (lhs, rhs), (), block, loc, res)
    return res


_cmp_map = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Is: operator.is_
}


@_register(_expr_handlers, ast.Compare)
def _compare_expr(cmp: ast.Compare, block: ir.Block, ctx: _Context) -> ir.Var:
    """
    cond = left $op0 comparator0 $op1 comparator1 $op2 comparator2
    -->
    c0 = left $op0 comparator0
    c = if c0:
            c1 = comparator0 $op1 comparator1
            c12 = if c1:
                    c2 = comparator1 $op2 comparator2
                    yield c2
                else:
                    yield c1 # False
            yield c12
        else:
            yield c0 # False
    """
    loc = ctx.get_loc(cmp)

    op_func0 = _cmp_map.get(type(cmp.ops[0]))
    if op_func0 is None:
        raise ctx.unsupported_syntax(loc)
    lhs = _expr(cmp.left, block, ctx)
    rhs = _expr(cmp.comparators[0], block, ctx)
    cond0 = block.make_temp_var(loc)

    func0_var = block.make_temp_var(loc)
    ops.const(op_func0, block, loc, func0_var)
    ops.call(func0_var, (lhs, rhs), (), block, loc, cond0)

    if len(cmp.ops) == 1:
        return cond0

    then_block = ir.Block(block.ctx)
    cmp.left = cmp.comparators[0]
    cmp.comparators = cmp.comparators[1:]
    cmp.ops = cmp.ops[1:]
    cond_right = _expr(cmp, then_block, ctx)
    then_block.append(ops.EndBranch(cond0.loc, outputs=(cond_right,)))

    else_block = ir.Block(block.ctx)
    else_block.append(ops.EndBranch(cond0.loc, outputs=(cond0,)))
    result = block.make_temp_var(loc=loc)
    block.append(ops.IfElse(cond0, then_block, else_block, cond0.loc,
                            results=ops.IfElseResults((result.name,), (result,))))
    return result


@_register(_expr_handlers, ast.Attribute)
def _attribute_expr(attr: ast.Attribute, block: ir.Block, ctx: _Context) -> ir.Var:
    value = _expr(attr.value, block, ctx)

    loc = ctx.get_loc(attr)

    func_var = block.make_temp_var(loc)
    ops.const(getattr, block, loc, func_var)

    name_var = block.make_temp_var(loc)
    ops.const(attr.attr, block, loc, name_var)

    res = block.make_temp_var(loc)
    ops.call(func_var, (value, name_var), (), block, loc, res)
    return res


@_register(_expr_handlers, ast.Constant)
def _constant_expr(node: ast.Constant, block: ir.Block, ctx: Any) -> ir.Var:
    """Handle constant expressions"""
    loc = ctx.get_loc(node)
    res = block.make_temp_var(loc)
    ops.const(node.value, block, loc, res)
    return res


@_register(_expr_handlers, ast.Tuple)
def _tuple_expr(tup: ast.Tuple, block: ir.Block, ctx: _Context) -> ir.Var:
    items = tuple(_expr(x, block, ctx) for x in tup.elts)
    loc = ctx.get_loc(tup)

    func_var = block.make_temp_var(loc)
    ops.const(ct._build_tuple, block, loc, func_var)

    res = block.make_temp_var(loc)
    ops.call(func_var, items, (), block, loc, res)
    return res


@_register(_expr_handlers, ast.Subscript)
def _subscript_expr(subscript: ast.Subscript, block: ir.Block, ctx: _Context) -> ir.Var:
    value = _expr(subscript.value, block, ctx)
    loc = ctx.get_loc(subscript)

    index = _expr(subscript.slice, block, ctx)
    res = block.make_temp_var(loc)

    func_var = block.make_temp_var(loc)
    ops.const(operator.getitem, block, loc, func_var)
    ops.call(func_var, (value, index), (), block, loc, res)
    return res


@_register(_expr_handlers, ast.Slice)
def _slice_stmt(slice_: ast.Slice, block: ir.Block, ctx: _Context) -> ir.Var:
    loc = ctx.get_loc(slice_)

    def get_var(x: ast.AST | None):
        if x is None:
            ret = block.make_temp_var(loc)
            ops.const(None, block, loc, ret)
            return ret
        return _expr(x, block, ctx)

    lower, upper, step = map(get_var, (slice_.lower, slice_.upper, slice_.step))

    func_var = block.make_temp_var(loc)
    ops.const(slice, block, loc, func_var)

    res = block.make_temp_var(loc)
    ops.call(func_var, (lower, upper, step), (), block, loc, res)
    return res


def _unsupported_expr(expr: ast.AST, block: ir.Block, ctx: _Context):
    raise ctx.unsupported_syntax(expr)


def _expr(expr: ast.AST, block: ir.Block, ctx: Any) -> ir.Var:
    """Dispatch expression node to appropriate handler"""
    handler = _expr_handlers.get(type(expr), _unsupported_expr)
    return handler(expr, block, ctx)


# ================================
# Statements
# ================================
_stmt_handlers: Dict[Type[ast.AST], Callable] = {}


@_register(_stmt_handlers, ast.Assign)
def _assign_stmt(assign: ast.Assign, block: ir.Block, ctx: Any) -> None:
    """Handle assignment statements"""
    loc = ctx.get_loc(assign)
    value = _expr(assign.value, block, ctx)

    for target in reversed(assign.targets):
        if isinstance(target, ast.Name):
            target_id = target.id
            ops.store(target_id, value, block, loc)
        elif isinstance(target, ast.Tuple):
            for i, el in enumerate(target.elts):
                if not isinstance(el, ast.Name):
                    raise ctx.unsupported_syntax(el)
                getitem_var = block.make_temp_var(loc)
                key = block.make_temp_var(block)
                ops.const(i, block, loc, key)

                func_var = block.make_temp_var(loc)
                ops.const(operator.getitem, block, loc, func_var)
                ops.call(func_var, (value, key), (), block, loc, getitem_var)

                ops.store(el.id, getitem_var, block, getitem_var.loc)
        else:
            raise ctx.unsupported_syntax(target)


@_register(_stmt_handlers, ast.AugAssign)
def _aug_assign_stmt(aug: ast.AugAssign, block: ir.Block, ctx: _Context):
    if not isinstance(aug.target, ast.Name):
        raise ctx.unsupported_syntax(aug.target)
    loc = ctx.get_loc(aug)
    op_func = _binop_map.get(type(aug.op))
    if op_func is None:
        raise ctx.unsupported_syntax(loc)
    lhs = block.make_temp_var(ctx.get_loc(aug.target))
    ops.load(aug.target.id, block, ctx.get_loc(aug.target), lhs)
    func_var = block.make_temp_var(loc)
    ops.const(op_func, block, loc, func_var)
    rhs = _expr(aug.value, block, ctx)
    res = block.make_temp_var(loc)
    ops.call(func_var, (lhs, rhs), (), block, loc, res)
    ops.store(aug.target.id, res, block, loc)


@_register(_stmt_handlers, ast.Expr)
def _expr_stmt(expr: ast.Expr, block: ir.Block, ctx: _Context):
    _expr(expr.value, block, ctx)


def _append_loop(op: ops.Loop, block: ir.Block, ctx: _Context):
    block.append(op)
    if not ctx.entry_point:
        # In order to propagate an early return, insert the following:
        #    if $returning:
        #        break
        flag = block.make_temp_var(op.loc)
        ops.load("$returning", block, op.loc, flag)
        then_block = ir.Block(block.ctx)
        then_block.append(ops.Break(op.loc))
        else_block = ir.Block(block.ctx)
        else_block.append(ops.EndBranch(op.loc))
        block.append(ops.IfElse(flag, then_block, else_block, op.loc))


@_register(_stmt_handlers, ast.For)
def _for_stmt(stmt: ast.For, block: ir.Block, ctx: _Context):
    if len(stmt.orelse) > 0:
        raise ctx.syntax_error(stmt.orelse[0], "'for-else' is not supported")

    iterable = _expr(stmt.iter, block, ctx)
    if not isinstance(stmt.target, ast.Name):
        raise ctx.unsupported_syntax(stmt.target)
    loc = ctx.get_loc(stmt)

    ctx.parent_loops.append(LoopKind.FOR)
    body_block = _block(stmt.body, ctx)
    body_block.append(ops.Continue(loc))
    ctx.parent_loops.pop()

    induction_var = ir.Var(stmt.target.id, ctx.get_loc(stmt.target), block.ctx)
    for_loop = ops.ForLoopInfo(induction_var, iterable)
    # Early return is not supported in a for loop, so we need to append the loop op directly.
    block.append(ops.Loop(body_block, loc, for_loop=for_loop))


def _cast_cond_to_bool(cond: ir.Var, loc: ir.Loc, block: ir.Block) -> ir.Var:
    bool_type_var = block.make_temp_var(loc)
    ops.const(datatype.bool_, block, loc, res=bool_type_var)
    cond_var = block.make_temp_var(loc)
    ops.call(bool_type_var, (cond,), (), block, loc, cond_var)
    return cond_var


@_register(_stmt_handlers, ast.While)
def _while_stmt(stmt: ast.While, block: ir.Block, ctx: _Context):
    if len(stmt.orelse) > 0:
        raise ctx.syntax_error(stmt.orelse[0], "'while-else' is not supported")

    loc = ctx.get_loc(stmt)
    body_block = ir.Block(block.ctx)

    # Add "if cond: pass; else: break"
    cond = _expr(stmt.test, body_block, ctx)
    cond_var = _cast_cond_to_bool(cond, loc, body_block)
    then_block = ir.Block(block.ctx)
    then_block.append(ops.EndBranch(cond.loc))

    else_block = ir.Block(block.ctx)
    else_block.append(ops.Break(cond.loc))

    body_block.append(ops.IfElse(cond_var, then_block, else_block, cond.loc))

    ctx.parent_loops.append(LoopKind.WHILE)
    for stmt in stmt.body:
        _stmt(stmt, body_block, ctx)
    body_block.append(ops.Continue(loc))
    ctx.parent_loops.pop()

    _append_loop(ops.Loop(body_block, loc), block, ctx)


@_register(_expr_handlers, ast.BoolOp)
def _boolop_expr(boolop: ast.BoolOp, block: ir.Block, ctx: _Context) -> ir.Var:
    loc = ctx.get_loc(boolop)
    assert len(boolop.values) >= 2
    cond0 = _expr(boolop.values[0], block, ctx)
    cond0 = _cast_cond_to_bool(cond0, loc, block)
    result = block.make_temp_var(loc=loc)

    if isinstance(boolop.op, ast.And):
        """
        cond = cond0() and cond1():
        -->
        c0 = cond0()
        c = if c0:
            c1 = cond1()
            yield c1
        else:
            yield c0 # False
        """
        then_block = ir.Block(block.ctx)
        if len(boolop.values) > 2:
            # Consecutive operations with the same operator, such as a or b or c,
            # are collapsed into one node with several values.
            boolop.values = boolop.values[1:]
            cond1 = _expr(boolop, then_block, ctx)
        else:
            cond1 = _expr(boolop.values[1], then_block, ctx)
        cond1 = _cast_cond_to_bool(cond1, loc, then_block)
        then_block.append(ops.EndBranch(cond0.loc, outputs=(cond1,)))

        else_block = ir.Block(block.ctx)
        else_block.append(ops.EndBranch(cond0.loc, outputs=(cond0,)))
        block.append(ops.IfElse(cond0, then_block, else_block, loc,
                                results=ops.IfElseResults((result.name,), (result,))))

    elif isinstance(boolop.op, ast.Or):
        """
        cond = cond0() or cond1():
        -->
        c0 = cond0()
        c = if c0:
            yield c0
        else:
            c1 = cond1()
            yield c1
        """
        then_block = ir.Block(block.ctx)
        then_block.append(ops.EndBranch(cond0.loc, outputs=(cond0,)))

        else_block = ir.Block(block.ctx)
        if len(boolop.values) > 2:
            boolop.values = boolop.values[1:]
            cond1 = _expr(boolop, else_block, ctx)
        else:
            cond1 = _expr(boolop.values[1], else_block, ctx)
        cond1 = _cast_cond_to_bool(cond1, loc, else_block)
        else_block.append(ops.EndBranch(cond0.loc, outputs=(cond1,)))

        block.append(ops.IfElse(cond0, then_block, else_block, loc,
                                results=ops.IfElseResults((result.name,), (result,))))

    else:
        raise ctx.unsupported_syntax(boolop)

    return result


@_register(_expr_handlers, ast.IfExp)
def _ifexp_expr(ifexp: ast.IfExp, block: ir.Block, ctx: _Context) -> ir.Var:
    loc = ctx.get_loc(ifexp)
    cond = _expr(ifexp.test, block, ctx)
    cond = _cast_cond_to_bool(cond, loc, block)
    result = block.make_temp_var(loc=loc)

    then_block = ir.Block(block.ctx)
    body_val = _expr(ifexp.body, then_block, ctx)
    then_block.append(ops.EndBranch(loc, outputs=(body_val,)))
    else_block = ir.Block(block.ctx)
    orelse_val = _expr(ifexp.orelse, else_block, ctx)
    else_block.append(ops.EndBranch(loc, outputs=(orelse_val,)))

    block.append(ops.IfElse(cond, then_block, else_block, loc,
                            results=ops.IfElseResults((result.name,), (result,))))
    return result


@_register(_stmt_handlers, ast.If)
def _if_stmt(stmt: ast.If, block: ir.Block, ctx: _Context) -> None:
    loc = ctx.get_loc(stmt)
    cond = _expr(stmt.test, block, ctx)
    cond = _cast_cond_to_bool(cond, loc, block)
    then_block = _block(stmt.body, ctx)
    then_block.append(ops.EndBranch(loc))

    else_block = _block(stmt.orelse, ctx)
    else_block.append(ops.EndBranch(loc))
    block.append(ops.IfElse(cond, then_block, else_block, loc))


@_register(_stmt_handlers, ast.Continue)
def _continue_stmt(stmt: ast.Continue, block: ir.Block, ctx: _Context) -> None:
    block.append(ops.Continue(ctx.get_loc(stmt)))


@_register(_stmt_handlers, ast.Break)
def _break_stmt(stmt: ast.Break, block: ir.Block, ctx: _Context) -> None:
    if ctx.parent_loops and ctx.parent_loops[-1] is LoopKind.FOR:
        raise ctx.syntax_error(stmt, "Break in a for loop is not supported")

    block.append(ops.Break(ctx.get_loc(stmt)))


@_register(_stmt_handlers, ast.Return)
def _return_stmt(stmt: ast.Return, block: ir.Block, ctx: _Context) -> None:
    if ctx.parent_loops and ctx.parent_loops[-1] is LoopKind.FOR:
        raise ctx.syntax_error(stmt, "Returning from a for loop is not supported")

    loc = ctx.get_loc(stmt)
    if stmt.value is None:
        return_val = block.make_temp_var(loc)
        ops.const(None, block, loc, return_val)
    else:
        return_val = _expr(stmt.value, block, ctx)

    if ctx.entry_point:
        block.append(ops.Return(return_val, loc))
    else:
        block.append(ops.Store("$retval", return_val, loc))
        true_var = block.make_temp_var(loc)
        ops.const(True, block, loc, true_var)
        block.append(ops.Store("$returning", true_var, loc))
        block.append(ops.Break(loc))


@_register(_stmt_handlers, ast.Pass)
def _pass_stmt(stmt: ast.Pass, block: ir.Block, ctx: _Context) -> None:
    pass


def _unsupported_stmt(stmt: ast.AST, block: ir.Block, ctx: _Context) -> None:
    raise ctx.unsupported_syntax(stmt)


def _stmt(stmt: ast.AST, block: ir.Block, ctx: Any) -> None:
    handler = _stmt_handlers.get(type(stmt), _unsupported_stmt)
    handler(stmt, block, ctx)


def _block(stmts: Sequence[ast.AST], ctx: _Context) -> ir.Block:
    block = ir.Block(ctx.ir_ctx)
    for stmt in stmts:
        _stmt(stmt, block, ctx)
    return block


class _VersionMap:
    def __init__(self, ir_ctx: ir.IRContext, parent: Optional["_VersionMap"] = None):
        self._ir_ctx = ir_ctx
        self._map = dict()
        self._parent = parent

    def redefine(self, name: str, loc: ir.Loc) -> ir.Var:
        var = self._ir_ctx.make_var(name, loc)
        self._map[name] = var
        return var

    def __getitem__(self, name: str):
        var = self._lookup(name)
        if var is None:
            raise TileSyntaxError(f"Uninitialized variable {name} used")
        return var

    def get(self, name: str, loc: ir.Loc):
        var = self._lookup(name)
        if var is None:
            return self._ir_ctx.make_var(name, loc, undefined=True)
        else:
            return var

    def _lookup(self, name: str) -> Optional[ir.Var]:
        seen = set()
        current = self
        while current is not None:
            var = current._map.get(name)
            if var is not None:
                return var
            # Sanity check, should not reach here.
            if id(current) in seen:
                raise RuntimeError("Cycle detected in VersionMap chain")
            seen.add(id(current))
            current = current._parent
        return None

    def branch(self) -> "_VersionMap":
        return _VersionMap(self._ir_ctx, self)


def _eliminate_load_store_for_op(
    op: ir.Operation,
    ctx: _Context,
    new_operations: List[ir.Operation],
    version_map: _VersionMap,
    all_locals: Set[str],
    stored_vars_by_block: Dict[ir.Block, Set[str]],
    innermost_loop_vars: Optional[ops.CarriedVariables],
    ifelse_results: Optional[ops.IfElseResults],
    is_for_loop_body: bool
) -> None:
    old_op_is_removed = True
    if isinstance(op, ops.Load):
        name = op.var_name
        if name in all_locals:
            rhs = version_map[name]
            if innermost_loop_vars:
                for i, body_var in enumerate(innermost_loop_vars.body):
                    if rhs.name == body_var.name:
                        if innermost_loop_vars.initial[i].is_undefined():
                            raise TileSyntaxError(f"Uninitialized variable {name} used")
            func_var = ctx.ir_ctx.make_temp(op.loc)
            new_operations.append(ops.Const(ct._identity, func_var, op.loc))
            new_operations.append(ops.Call(func_var, (rhs,), (), op.result_var, op.loc))
        elif name in ctx.globals:
            const_val = get_constant_value(ctx.globals[name])
            new_operations.append(ops.Const(const_val, op.result_var, op.loc))
        else:
            raise TileSyntaxError(f"Uninitialized variable {name} used")
    elif isinstance(op, ops.Store):
        var = version_map.redefine(op.lhs_var_name, op.loc)
        func_var = ctx.ir_ctx.make_temp(op.loc)
        new_operations.append(ops.Const(ct._identity, func_var, op.loc))
        new_operations.append(ops.Call(func_var, (op.rhs,), (), var, op.loc))
    elif isinstance(op, ops.Loop):
        # Sort the carried variable names to make the order deterministic.
        carried_var_names = sorted(stored_vars_by_block[op.body])
        input_carried_vars = tuple(version_map.get(name, op.loc) for name in carried_var_names)
        body_carried_vars = tuple(version_map.redefine(name, op.loc)
                                  for name in carried_var_names)

        if op.for_loop is None:
            for_loop = None
        else:
            induction_var = version_map.redefine(op.for_loop.induction_var.name,
                                                 op.for_loop.induction_var.loc)
            for_loop = ops.ForLoopInfo(induction_var, op.for_loop.iterable)

        carried_vars = ops.CarriedVariables(tuple(carried_var_names), input_carried_vars,
                                            body_carried_vars)

        _eliminate_load_store_in_block(op.body, ctx, version_map,
                                       all_locals, stored_vars_by_block, carried_vars, None,
                                       for_loop is not None)

        carried_vars.results = tuple(version_map.redefine(name, op.loc)
                                     for name in carried_var_names)
        loop_op = ops.Loop(op.body, op.loc, for_loop, carried_vars)
        new_operations.append(loop_op)
    elif isinstance(op, ops.Continue):
        if innermost_loop_vars is None:
            raise TileSyntaxError("'continue' not within a loop")
        next_vars = tuple(version_map.get(name, op.loc) for name in innermost_loop_vars.names)
        new_operations.append(ops.Continue(op.loc, next_vars, innermost_loop_vars,
                                           is_for_loop_body))
    elif isinstance(op, ops.Break):
        if innermost_loop_vars is None:
            raise TileSyntaxError("'break' not within a loop")
        output_vars = tuple(version_map.get(name, op.loc) for name in innermost_loop_vars.names)
        new_operations.append(ops.Break(op.loc, output_vars, innermost_loop_vars))
    elif isinstance(op, ops.EndBranch):
        assert ifelse_results is not None
        if len(op.outputs):
            # When the end branch has outputs, it means that it already knows what to yield.
            assert len(op.outputs) == len(ifelse_results.names)
            outputs = op.outputs
        else:
            # When the end branch has no outputs, it means that it needs to yield the result
            # from the if-else.
            outputs = tuple(version_map.get(name, op.loc) for name in ifelse_results.names)
        new_operations.append(ops.EndBranch(op.loc, outputs, ifelse_results))
    elif isinstance(op, ops.IfElse):
        # Sort the output variable names to make the order deterministic.
        output_var_names = sorted(tuple(stored_vars_by_block[op.then_block]
                                  | stored_vars_by_block[op.else_block]))
        result_name_to_var = {}
        if op.results is not None:
            output_var_names += op.results.names
            result_name_to_var = {
                name: var
                for name, var in zip(op.results.names, op.results.vars)
            }

        results = ops.IfElseResults(output_var_names)
        for nested_block in op.nested_blocks:
            _eliminate_load_store_in_block(nested_block, ctx,
                                           version_map.branch(), all_locals,
                                           stored_vars_by_block, innermost_loop_vars,
                                           results, is_for_loop_body)
        result_vars = []
        for name in output_var_names:
            # Get the result var from the result name to var map, or create a new one.
            result_var = result_name_to_var.get(name, version_map.redefine(name, op.loc))
            result_vars.append(result_var)
        results.vars = tuple(result_vars)
        new_op = ops.IfElse(
            op.cond, op.then_block, op.else_block, op.loc, results
        )
        new_operations.append(new_op)
    else:
        assert not op.nested_blocks
        old_op_is_removed = False

    if not old_op_is_removed:
        new_operations.append(op)


def _eliminate_load_store_in_block(
    block: ir.Block,
    ctx: _Context,
    version_map: _VersionMap,
    all_locals: Set[str],
    stored_vars_by_block: Dict[ir.Block, Set[str]],
    innermost_loop_vars: Optional[ops.CarriedVariables],
    ifelse_results: Optional[ops.IfElseResults],
    is_for_loop_body: bool = False,
) -> None:
    new_operations = []
    for i, op in enumerate(block.operations):
        with op.loc:
            _eliminate_load_store_for_op(
                op, ctx, new_operations, version_map, all_locals, stored_vars_by_block,
                innermost_loop_vars, ifelse_results, is_for_loop_body)
        if op.is_terminator:
            break
    block.operations = new_operations


def _eliminate_load_store_pass(root_block: ir.Block,
                               all_params: Sequence[ast.arg],
                               ctx: _Context) -> Sequence[ir.Var]:
    stored_vars_by_block = {}
    _get_stored_vars_by_block(root_block, stored_vars_by_block)
    all_locals = stored_vars_by_block[root_block]
    version_map = _VersionMap(root_block.ctx)
    param_vars = []
    for p in all_params:
        var = version_map.redefine(p.arg, ctx.get_loc(p))
        param_vars.append(var)
        all_locals.add(p.arg)

    _eliminate_load_store_in_block(root_block, ctx, version_map,
                                   all_locals, stored_vars_by_block, None, None, False)
    return param_vars


def _get_stored_vars_by_block(
    block: ir.Block,
    stored_vars_by_block: Dict[ir.Block, Set[str]]
) -> None:
    stored_vars = set()
    for op in block.operations:
        if isinstance(op, ops.Store):
            stored_vars.add(op.lhs_var_name)
        elif op.nested_blocks:
            for nested_block in op.nested_blocks:
                _get_stored_vars_by_block(nested_block, stored_vars_by_block)
            if isinstance(op, ops.Loop) and op.for_loop is not None:
                stored_vars_by_block[op.body].add(op.for_loop.induction_var.name)
            for nested_block in op.nested_blocks:
                stored_vars.update(stored_vars_by_block[nested_block])
    stored_vars_by_block[block] = stored_vars


def _get_all_parameters(func_def: ast.FunctionDef, ctx: _Context) -> List[ast.arg]:
    for a in (func_def.args.vararg, func_def.args.kwarg):
        if a is not None:
            raise TileSyntaxError("Variadic parameters in user-defined functions are not supported",
                                  ctx.get_loc(a))
    all_args = []
    for arg in func_def.args.posonlyargs:
        all_args.append(arg)
    for arg in func_def.args.args:
        all_args.append(arg)
    for arg in func_def.args.kwonlyargs:
        all_args.append(arg)
    return all_args


# Get all public functions defined in ct
ct_module = inspect.getmodule(ct)


def _verify_ir_in_block(block: ir.Block, var_ids: Dict[str, int]) -> None:
    for op in block.operations:
        for result_var in op.result_vars:
            var_ids[result_var.name] = id(result_var)

        # Verify that all the variables with the same name are the same object.
        for operand in op.operands.values():
            if isinstance(operand, ir.Var):
                values = tuple([operand])
            else:
                values = operand
            for value in values:
                if value.name in var_ids and var_ids[value.name] != id(value):
                    raise RuntimeError(f"Variable `{operand.name}' is not unique")

        # Verify that there is no ast ops.
        if isinstance(op, ir.AstOperation):
            raise RuntimeError(f"Ast operation {op.op} found")

        for nested_block in op.nested_blocks:
            _verify_ir_in_block(nested_block, var_ids)


def _ast2ir(func_def: ast.FunctionDef, ctx: _Context) -> ir.Function:
    all_params = _get_all_parameters(func_def, ctx)
    func_loc = ctx.get_loc(func_def)
    root_block = _block(func_def.body, ctx)

    if ctx.entry_point:
        # Append a return op to the root block if it doesn't have one.
        if not root_block.operations or not isinstance(root_block.operations[-1], ops.Return):
            return_val = root_block.make_temp_var(func_loc)
            ops.const(None, root_block, func_loc, return_val)
            root_block.append(ops.Return(return_val, func_loc))
        retval = None
    else:
        # To enable early returns in a helper function, wrap the body in a "Loop" op.
        # Thus, we can use "break" to implement the return statement.
        none_val = root_block.make_temp_var(func_loc)
        ops.const(None, root_block, func_loc, none_val)
        root_block.append(ops.Store("$retval", none_val, func_loc))
        root_block.append(ops.Break(func_loc))

        new_root_block = ir.Block(ctx.ir_ctx)
        false_val = new_root_block.make_temp_var(func_loc)
        ops.const(False, new_root_block, func_loc, false_val)
        new_root_block.append(ops.Store("$returning", false_val, func_loc))
        new_root_block.append(ops.Loop(root_block, func_loc))
        retval = new_root_block.make_temp_var(func_loc)
        new_root_block.append(ops.Load("$retval", retval, func_loc))
        root_block = new_root_block

    param_vars = _eliminate_load_store_pass(root_block, all_params, ctx)
    _verify_ir_in_block(root_block, {})
    return ir.Function(func_def.name, root_block, tuple(param_vars), retval, func_loc)
