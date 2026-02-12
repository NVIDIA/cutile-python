# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import itertools
import operator
from contextlib import contextmanager
from enum import Enum, auto
from functools import lru_cache
from typing import List, Sequence, Optional, Any, Dict, Type, Callable, OrderedDict, Mapping

from cuda.tile import _datatype as datatype
from cuda.tile._exception import TileSyntaxError, Loc, FunctionDesc
from cuda.tile._ir.hir import make_value, ResolvedName, UNKNOWN_NAME
from cuda.tile._ir import hir
from cuda.tile._ir.type import ClosureDefaultPlaceholder
from cuda.tile._passes.ast_util import ast_get_all_local_names
from cuda.tile._stub import static_eval, static_assert


@lru_cache
def get_function_hir(pyfunc: Callable, entry_point: bool) -> hir.Function:
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
    _fix_line_numbers(func_def, first_line)

    func_globals = dict(pyfunc.__builtins__)
    func_globals.update(pyfunc.__globals__)
    # Add closure variables (from freevars)
    if pyfunc.__closure__:
        for name, cell in zip(pyfunc.__code__.co_freevars, pyfunc.__closure__):
            func_globals[name] = cell.cell_contents

    filename = inspect.getfile(pyfunc)
    desc = FunctionDesc(func_def.name, filename, first_line)
    local_names, _, _ = ast_get_all_local_names(func_def)
    ctx = _Context(filename, first_line, desc, func_globals, local_names, entry_point)
    signature = inspect.signature(pyfunc)
    ret = _get_function_hir_inner(func_def, signature, ctx)

    resolved_names = {name: ResolvedName(-1, i) for i, name in enumerate(ret.frozen_global_names)}
    _finalize_func(ret, resolved_names, 0, ())
    return ret


# Translate the 1-based line numbers of the chunk we passed to the AST parser
# to the original 1-based line numbers in the file.
def _fix_line_numbers(tree: ast.AST, first_line: int):
    for node in ast.walk(tree):
        if hasattr(node, "lineno"):
            # Why -2?
            #    -1 because both first_line and node.lineno are 1-based;
            #    another -1 to account for the "if True" line that we inserted.
            node.lineno += first_line - 2
            node.end_lineno += first_line - 2

            # Subtract 1 from the column offset to correct for an extra level
            # of indentation we inserted for the dummy "if True" block.
            node.col_offset -= 1
            node.end_col_offset -= 1


def _finalize_func(func: hir.Function, resolved_names: dict[str, ResolvedName], depth: int,
                   enclosing_functions: tuple[hir.Function, ...]):
    resolved_names = dict(resolved_names)
    for i, name in enumerate(func.local_names):
        resolved_names[name] = ResolvedName(depth, i)

    all_used_names = set(func.loaded_names + func.local_names)
    new_enclosing_functions = enclosing_functions + (func,)
    for nested_func in func.nested_functions:
        _finalize_func(nested_func, resolved_names, depth + 1, new_enclosing_functions)
        for name, rn in nested_func.used_names.items():
            if rn.depth <= depth:
                all_used_names.add(name)

    captures_by_depth = tuple([] for _ in range(depth))
    for name in sorted(all_used_names):
        rn = resolved_names.get(name, UNKNOWN_NAME)
        func.used_names[name] = rn
        if 0 <= rn.depth < depth:
            captures_by_depth[rn.depth].append(rn.index)
    func.captures_by_depth = tuple(tuple(lst) for lst in captures_by_depth)
    func.enclosing_funcs = enclosing_functions


def _get_function_hir_inner(func_def: ast.FunctionDef | ast.Lambda, signature: inspect.Signature,
                            ctx: "_Context") -> hir.Function:
    assert isinstance(func_def, ast.FunctionDef | ast.Lambda)
    body = _ast2hir(func_def, ctx)
    all_ast_args = _get_all_parameters(func_def, ctx)
    param_names = tuple(p.arg for p in all_ast_args)
    body.stored_names.update(param_names)
    local_names = tuple(sorted(body.stored_names))
    frozen_global_names = tuple(sorted(ctx.frozen_globals.keys()))
    frozen_global_values = tuple(ctx.frozen_globals[name] for name in frozen_global_names)
    return hir.Function(
        desc=ctx.function_desc,
        body=body,
        signature=signature,
        local_names=local_names,
        param_local_indices=tuple(local_names.index(name) for name in param_names),
        param_locs=tuple(ctx.get_loc(p) for p in all_ast_args),
        frozen_global_names=frozen_global_names,
        frozen_global_values=frozen_global_values,
        value_id_upper_bound=next(ctx.value_id_sequence),
        nested_functions=tuple(ctx.nested_functions),
        loaded_names=tuple(sorted(ctx.loaded_names)),
        used_names=OrderedDict(),  # to be filled later
        captures_by_depth=(),  # to be filled later
        enclosing_funcs=(),  # to be filled later
        code_object=None,  # to be maybe filled later
    )


class LoopKind(Enum):
    FOR = auto()
    WHILE = auto()


class _Context:
    def __init__(self, filename: str, first_line: int, function_desc: FunctionDesc,
                 frozen_globals: Mapping[str, Any], local_names: set[str], entry_point: bool):
        self.filename = filename
        self.first_line = first_line
        self.function_desc = function_desc
        self.frozen_globals = frozen_globals
        self.local_names = local_names  # includes captures from parent scopes
        self.entry_point = entry_point
        self.parent_loops: List[LoopKind] = []
        self.current_loc = Loc.unknown()
        self.current_block: Optional[hir.Block] = None
        self.value_id_sequence = itertools.count()
        self.block_id_sequence = itertools.count()
        self.nested_functions = []
        self.loaded_names = set()

    def make_value(self) -> hir.Value:
        return make_value(next(self.value_id_sequence))

    @contextmanager
    def change_loc(self, loc: ast.AST | Loc):
        old = self.current_loc
        self.current_loc = loc if isinstance(loc, Loc) else self.get_loc(loc)
        try:
            yield
        finally:
            self.current_loc = old

    @contextmanager
    def new_block(self, params: Sequence[hir.Value] = ()):
        block_id = next(self.block_id_sequence)
        new_block = hir.Block(block_id, tuple(params), calls=[], have_result=False, result=None,
                              jump=None, jump_loc=Loc.unknown(),
                              stored_names=set(), loc=self.current_loc)
        old = self.current_block
        self.current_block = new_block
        try:
            yield self.current_block
        finally:
            self.current_block = old

        if old is not None:
            old.stored_names.update(new_block.stored_names)

    def call(self, callee, args, kwargs=()) -> hir.Value:
        res = self.make_value()
        self.current_block.calls.append(hir.Call(res, callee, args, kwargs, self.current_loc))
        return res

    def call_void(self, callee, args, kwargs=()) -> None:
        self.current_block.calls.append(hir.Call(None, callee, args, kwargs, self.current_loc))

    def set_block_jump(self, jump: hir.Jump):
        assert self.current_block.jump is None
        self.current_block.jump = jump
        self.current_block.jump_loc = self.current_loc

    def set_block_jump_with_result(self, jump: hir.Jump, result: hir.Operand):
        self.set_block_jump(jump)
        self.current_block.result = result
        self.current_block.have_result = True

    def store(self, var_name: str, value: hir.Operand):
        self.call_void(hir.store_var, (var_name, value))
        self.current_block.stored_names.add(var_name)

    def load(self, var_name: str) -> hir.Value:
        self.loaded_names.add(var_name)
        return self.call(hir.load_var, (var_name,))

    def get_loc(self, node: ast.AST) -> Loc:
        return Loc(node.lineno, node.col_offset, self.filename,
                   node.end_lineno, node.end_col_offset, self.function_desc)

    def syntax_error(self, message: str, loc=None) -> TileSyntaxError:
        if loc is None:
            loc = self.current_loc
        elif not isinstance(loc, Loc):
            loc = self.get_loc(loc)
        return TileSyntaxError(message, loc)

    def unsupported_syntax(self, loc=None) -> TileSyntaxError:
        return self.syntax_error("Unsupported syntax", loc=loc)


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
def _call_expr(call: ast.Call, ctx: _Context) -> hir.Value:
    kwd_func = _parse_keyword_like_func(call.func,
                                        (static_eval, static_assert),
                                        ("static_eval", "static_assert"),
                                        ctx)
    if kwd_func is not None:
        if kwd_func == "static_eval":
            if len(call.args) != 1 or len(call.keywords) != 0:
                raise ctx.syntax_error("static_eval() expects a single expression")
            return _call_static_eval(call.args[0], hir.StaticEvalKind.STATIC_EVAL, ctx)
        elif kwd_func == "static_assert":
            if len(call.args) not in (1, 2) or len(call.keywords) != 0:
                raise ctx.syntax_error("static_assert(cond, msg=None, /)"
                                       " expects one or two positional arguments")
            with ctx.new_block() as message_block:
                if len(call.args) > 1:
                    message = _call_static_eval(call.args[1],
                                                hir.StaticEvalKind.STATIC_ASSERT_MESSAGE, ctx)
                else:
                    message = ""
                ctx.set_block_jump_with_result(hir.Jump.END_BRANCH, message)
            condition = _call_static_eval(call.args[0],
                                          hir.StaticEvalKind.STATIC_ASSERT_CONDITION, ctx)
            return ctx.call(hir.do_static_assert, (condition, message_block))
        else:
            assert False
    else:
        callee = _expr(call.func, ctx)
        args = tuple(_expr(a, ctx) for a in call.args)
        kwargs = tuple((a.arg, _expr(a.value, ctx)) for a in call.keywords)
        return ctx.call(callee, args, kwargs)


def _call_static_eval(expr: ast.expr, kind: hir.StaticEvalKind, ctx: _Context) -> hir.Value:
    # Wrap the `expr` as `lambda: expr`
    inner_lambda_ast = _wrap_in_lambda(expr, ())

    # Wrap with another lambda that takes all local names as arguments:
    #     `lambda local1, local2, ...: lambda: expr`
    local_names = sorted(ctx.local_names)
    outer_lambda_ast = _wrap_in_lambda(inner_lambda_ast, local_names)

    # Compile and eval the AST to get an instance of the outer lambda function
    outer_lambda = _eval_ast_expr(outer_lambda_ast, ctx)

    # Call the outer lambda to create an instance of the inner lambda
    inner_lambda = outer_lambda(*tuple(None for _ in local_names))

    # Make sure the function doesn't store any locals, e.g. using the walrus operator
    if len(inner_lambda.__code__.co_varnames) > 0:
        name = inner_lambda.__code__.co_varnames[0]
        raise TileSyntaxError(f"static_eval() expression attempted"
                              f" to modify a local variable '{name}'")

    # Look at the inner lambda's freevars to determine which locals are being used
    used_locals = sorted(inner_lambda.__code__.co_freevars)

    # Compile a new lambda function of the form
    #    lambda p1, p2, ...: expr
    # Where p1, p2, ... are the names of used local variables.
    final_lambda_ast = _wrap_in_lambda(expr, used_locals)
    final_lambda = _eval_ast_expr(final_lambda_ast, ctx)

    loaded_locals = tuple(ctx.load(local_name) for local_name in used_locals)
    return ctx.call(hir.do_static_eval,
                    (hir.StaticEvalExpression(final_lambda, kind), *loaded_locals))


def _wrap_in_call(lamb: ast.Lambda) -> ast.Call:
    return ast.Call(func=lamb, args=[], keywords=[], lineno=lamb.lineno, end_lineno=lamb.end_lineno,
                    col_offset=lamb.col_offset, end_col_offset=lamb.end_col_offset)


def _wrap_in_lambda(expr: ast.expr, param_names: Sequence[str]) -> ast.Lambda:
    locals_as_ast_args = [ast.arg(arg=name, lineno=0, col_offset=0) for name in param_names]
    outer_lambda_args = ast.arguments(posonlyargs=locals_as_ast_args, args=[], vararg=None,
                                      kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    return ast.Lambda(args=outer_lambda_args, body=expr,
                      lineno=expr.lineno, end_lineno=expr.end_lineno,
                      col_offset=expr.col_offset, end_col_offset=expr.end_col_offset)


def _eval_ast_expr(expr: ast.expr, ctx: _Context):
    ast_to_eval = ast.Expression(body=expr)
    try:
        code = compile(ast_to_eval, ctx.filename, "eval")
    except (SyntaxError, ValueError) as e:
        # TODO: get location info from SyntaxError
        raise TileSyntaxError(str(e))

    return eval(code, dict(ctx.frozen_globals), {})


def _parse_keyword_like_func(expr: ast.expr,
                             kwd_funcs: tuple[Callable, ...],
                             kwd_func_names: tuple[str, ...],
                             ctx: _Context) -> str | None:
    if isinstance(expr, ast.Name):
        if (expr.id not in ctx.local_names
                and ctx.frozen_globals.get(expr.id) in kwd_funcs):
            idx = kwd_funcs.index(ctx.frozen_globals.get(expr.id))
            return kwd_func_names[idx]
    elif isinstance(expr, ast.Attribute):
        if expr.attr in kwd_func_names and _is_cuda_tile_module(expr.value, ctx):
            return expr.attr
    return None


def _is_cuda_tile_module(value: ast.expr, ctx: _Context) -> bool:
    if isinstance(value, ast.Name):
        if value.id in ctx.local_names:
            return False
        import cuda.tile
        return ctx.frozen_globals.get(value.id) is cuda.tile
    elif isinstance(value, ast.Attribute):
        return value.attr == "tile" and _is_cuda_module(value.value, ctx)
    else:
        return False


def _is_cuda_module(value: ast.expr, ctx: _Context) -> bool:
    if not isinstance(value, ast.Name):
        return False
    import cuda
    return ctx.frozen_globals.get(value.id) is cuda


@_register(_expr_handlers, ast.Name)
def _name_expr(name: ast.Name, ctx: Any) -> hir.Value:
    if not isinstance(name.ctx, ast.Load):
        raise ctx.unsupported_syntax()
    return ctx.load(name.id)


_unary_map = {ast.Invert: operator.invert, ast.Not: operator.not_,
              ast.UAdd: operator.pos, ast.USub: operator.neg}


@_register(_expr_handlers, ast.UnaryOp)
def _unary_op(unary: ast.UnaryOp, ctx: _Context) -> hir.Value:
    op_func = _unary_map.get(type(unary.op))
    if op_func is None:
        raise ctx.unsupported_syntax()

    operand = _expr(unary.operand, ctx)
    return ctx.call(op_func, (operand,))


_binop_map = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv, ast.Div: operator.truediv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.BitOr: operator.or_, ast.BitXor: operator.xor, ast.BitAnd: operator.and_,
    ast.LShift: operator.lshift, ast.RShift: operator.rshift,
    ast.MatMult: operator.matmul,
}


@_register(_expr_handlers, ast.BinOp)
def _binop_expr(binop: ast.BinOp, ctx: _Context) -> hir.Value:
    op_func = _binop_map.get(type(binop.op))
    if op_func is None:
        raise ctx.unsupported_syntax()
    lhs = _expr(binop.left, ctx)
    rhs = _expr(binop.right, ctx)
    return ctx.call(op_func, (lhs, rhs))


_cmp_map = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Is: operator.is_, ast.IsNot: operator.is_not,
}


@_register(_expr_handlers, ast.Compare)
def _compare_expr(cmp: ast.Compare, ctx: _Context) -> hir.Value:
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
    op_func0 = _cmp_map.get(type(cmp.ops[0]))
    if op_func0 is None:
        raise ctx.unsupported_syntax()
    lhs = _expr(cmp.left, ctx)
    rhs = _expr(cmp.comparators[0], ctx)

    cond0 = ctx.call(op_func0, (lhs, rhs))
    if len(cmp.ops) == 1:
        return cond0

    with ctx.new_block() as then_block:
        cmp.left = cmp.comparators[0]
        cmp.comparators = cmp.comparators[1:]
        cmp.ops = cmp.ops[1:]
        cond_right = _expr(cmp, ctx)
        ctx.set_block_jump_with_result(hir.Jump.END_BRANCH, cond_right)

    with ctx.new_block() as else_block:
        ctx.set_block_jump_with_result(hir.Jump.END_BRANCH, cond0)

    return ctx.call(hir.if_else, (cond0, then_block, else_block))


@_register(_expr_handlers, ast.Attribute)
def _attribute_expr(attr: ast.Attribute, ctx: _Context) -> hir.Value:
    value = _expr(attr.value, ctx)
    return ctx.call(getattr, (value, attr.attr))


@_register(_expr_handlers, ast.Constant)
def _constant_expr(node: ast.Constant, ctx: Any) -> Any:
    # We could just return node.value directly here, but we wrap the constant
    # in a `identity` call in order to preserve location info.
    return ctx.call(hir.identity, (node.value,))


@_register(_expr_handlers, ast.Tuple)
def _tuple_expr(tup: ast.Tuple, ctx: _Context) -> hir.Value:
    items = tuple(_expr(x, ctx) for x in tup.elts)
    return ctx.call(hir.build_tuple, items)


@_register(_expr_handlers, ast.Subscript)
def _subscript_expr(subscript: ast.Subscript, ctx: _Context) -> hir.Value:
    value = _expr(subscript.value, ctx)
    index = _expr(subscript.slice, ctx)
    return ctx.call(operator.getitem, (value, index))


@_register(_expr_handlers, ast.Slice)
def _slice_expr(slice_: ast.Slice, ctx: _Context) -> hir.Value:
    def get_var(x: ast.AST | None):
        return None if x is None else _expr(x, ctx)
    lower, upper, step = map(get_var, (slice_.lower, slice_.upper, slice_.step))
    return ctx.call(slice, (lower, upper, step))


@_register(_expr_handlers, ast.Lambda)
def _lambda_expr(lamb: ast.Lambda, ctx: _Context) -> hir.Value:
    return _make_closure(lamb, ctx)


def _unsupported_expr(expr: ast.AST, ctx: _Context):
    raise ctx.unsupported_syntax()


def _expr(expr: ast.AST, ctx: _Context) -> hir.Operand:
    """Dispatch expression node to appropriate handler"""
    handler = _expr_handlers.get(type(expr), _unsupported_expr)
    with ctx.change_loc(expr):
        return handler(expr, ctx)


# ================================
# Statements
# ================================
_stmt_handlers: Dict[Type[ast.AST], Callable] = {}


@_register(_stmt_handlers, ast.Assign)
def _assign_stmt(assign: ast.Assign, ctx: _Context) -> None:
    value = _expr(assign.value, ctx)
    for target in reversed(assign.targets):
        _do_assign(value, target, ctx)


@_register(_stmt_handlers, ast.AnnAssign)
def _ann_assign_stmt(ann_assign: ast.AnnAssign, ctx: _Context) -> None:
    if ann_assign.value is not None:
        value = _expr(ann_assign.value, ctx)
        _do_assign(value, ann_assign.target, ctx)


def _do_assign(value: hir.Operand, target, ctx: _Context):
    with ctx.change_loc(target):
        if isinstance(target, ast.Name):
            ctx.store(target.id, value)
        elif isinstance(target, ast.Tuple | ast.List):
            value = ctx.call(hir.unpack, (value, len(target.elts)))
            for i, el in enumerate(target.elts):
                item_var = ctx.call(operator.getitem, (value, i), )
                _do_assign(item_var, el, ctx)
        else:
            raise ctx.unsupported_syntax()


@_register(_stmt_handlers, ast.AugAssign)
def _aug_assign_stmt(aug: ast.AugAssign, ctx: _Context):
    if not isinstance(aug.target, ast.Name):
        raise ctx.unsupported_syntax(aug.target)
    op_func = _binop_map.get(type(aug.op))
    if op_func is None:
        raise ctx.unsupported_syntax()
    lhs = ctx.load(aug.target.id)
    rhs = _expr(aug.value, ctx)
    res = ctx.call(op_func, (lhs, rhs))
    ctx.store(aug.target.id, res)


@_register(_stmt_handlers, ast.Expr)
def _expr_stmt(expr: ast.Expr, ctx: _Context):
    _expr(expr.value, ctx)


def _propagate_return(ctx: _Context):
    if ctx.entry_point:
        return
    # In order to propagate an early return, insert the following:
    #    if $returning:
    #        break
    flag = ctx.load("$returning")
    with ctx.new_block() as then_block:
        ctx.set_block_jump(hir.Jump.BREAK)
    with ctx.new_block() as else_block:
        ctx.set_block_jump(hir.Jump.END_BRANCH)
    ctx.call_void(hir.if_else, (flag, then_block, else_block))


@_register(_stmt_handlers, ast.For)
def _for_stmt(stmt: ast.For, ctx: _Context):
    if len(stmt.orelse) > 0:
        raise ctx.syntax_error("'for-else' is not supported", loc=stmt.orelse[0])

    iterable = _expr(stmt.iter, ctx)
    if not isinstance(stmt.target, ast.Name):
        raise ctx.unsupported_syntax(stmt.target)

    ctx.parent_loops.append(LoopKind.FOR)
    induction_var = ctx.make_value()
    with ctx.new_block(params=(induction_var,)) as body_block:
        with ctx.change_loc(stmt.target):
            ctx.store(stmt.target.id, induction_var)
        _stmt_list(stmt.body, ctx)
        if body_block.jump is None:
            ctx.set_block_jump(hir.Jump.CONTINUE)
    ctx.parent_loops.pop()

    ctx.call_void(hir.loop, (body_block, iterable))


def _bool_expr(expr: ast.AST, ctx: _Context) -> hir.Value:
    val = _expr(expr, ctx)
    with ctx.change_loc(expr):
        return ctx.call(datatype.bool_, (val,))


@_register(_stmt_handlers, ast.While)
def _while_stmt(stmt: ast.While, ctx: _Context):
    if len(stmt.orelse) > 0:
        raise ctx.syntax_error("'while-else' is not supported", loc=stmt.orelse[0])

    with ctx.new_block() as body_block:
        # Add "if cond: pass; else: break"
        cond = _bool_expr(stmt.test, ctx)

        with ctx.new_block() as then_block:
            ctx.set_block_jump(hir.Jump.END_BRANCH)

        with ctx.new_block() as else_block:
            ctx.set_block_jump(hir.Jump.BREAK)

        ctx.call_void(hir.if_else, (cond, then_block, else_block))

        ctx.parent_loops.append(LoopKind.WHILE)
        _stmt_list(stmt.body, ctx)
        if body_block.jump is None:
            ctx.set_block_jump(hir.Jump.CONTINUE)
        ctx.parent_loops.pop()

    ctx.call_void(hir.loop, (body_block, None))
    _propagate_return(ctx)


@_register(_expr_handlers, ast.BoolOp)
def _boolop_expr(boolop: ast.BoolOp, ctx: _Context) -> hir.Value:
    assert len(boolop.values) >= 2
    cond0 = _bool_expr(boolop.values[0], ctx)

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
        with ctx.new_block() as then_block:
            if len(boolop.values) > 2:
                # Consecutive operations with the same operator, such as a or b or c,
                # are collapsed into one node with several values.
                boolop.values = boolop.values[1:]
                cond1 = _bool_expr(boolop, ctx)
            else:
                cond1 = _bool_expr(boolop.values[1], ctx)
            ctx.set_block_jump_with_result(hir.Jump.END_BRANCH, cond1)

        with ctx.new_block() as else_block:
            ctx.set_block_jump_with_result(hir.Jump.END_BRANCH, cond0)

        return ctx.call(hir.if_else, (cond0, then_block, else_block))
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
        with ctx.new_block() as then_block:
            ctx.set_block_jump_with_result(hir.Jump.END_BRANCH, cond0)

        with ctx.new_block() as else_block:
            if len(boolop.values) > 2:
                boolop.values = boolop.values[1:]
                cond1 = _bool_expr(boolop, ctx)
            else:
                cond1 = _bool_expr(boolop.values[1], ctx)
            ctx.set_block_jump_with_result(hir.Jump.END_BRANCH, cond1)

        return ctx.call(hir.if_else, (cond0, then_block, else_block))
    else:
        raise ctx.unsupported_syntax()


@_register(_expr_handlers, ast.IfExp)
def _ifexp_expr(ifexp: ast.IfExp, ctx: _Context) -> hir.Value:
    cond = _bool_expr(ifexp.test, ctx)

    with ctx.new_block() as then_block:
        then_val = _expr(ifexp.body, ctx)
        ctx.set_block_jump_with_result(hir.Jump.END_BRANCH, then_val)

    with ctx.new_block() as else_block:
        else_val = _expr(ifexp.orelse, ctx)
        ctx.set_block_jump_with_result(hir.Jump.END_BRANCH, else_val)

    return ctx.call(hir.if_else, (cond, then_block, else_block))


@_register(_stmt_handlers, ast.If)
def _if_stmt(stmt: ast.If, ctx: _Context) -> None:
    cond = _bool_expr(stmt.test, ctx)

    with ctx.new_block() as then_block:
        _stmt_list(stmt.body, ctx)
        if then_block.jump is None:
            ctx.set_block_jump(hir.Jump.END_BRANCH)

    with ctx.new_block() as else_block:
        _stmt_list(stmt.orelse, ctx)
        if else_block.jump is None:
            ctx.set_block_jump(hir.Jump.END_BRANCH)

    ctx.call_void(hir.if_else, (cond, then_block, else_block))


@_register(_stmt_handlers, ast.Continue)
def _continue_stmt(stmt: ast.Continue, ctx: _Context) -> None:
    ctx.set_block_jump(hir.Jump.CONTINUE)


@_register(_stmt_handlers, ast.Break)
def _break_stmt(stmt: ast.Break, ctx: _Context) -> None:
    if ctx.parent_loops and ctx.parent_loops[-1] is LoopKind.FOR:
        raise ctx.syntax_error("Break in a for loop is not supported")
    ctx.set_block_jump(hir.Jump.BREAK)


@_register(_stmt_handlers, ast.Return)
def _return_stmt(stmt: ast.Return, ctx: _Context) -> None:
    if ctx.parent_loops and ctx.parent_loops[-1] is LoopKind.FOR:
        raise ctx.syntax_error("Returning from a for loop is not supported")

    return_val = None if stmt.value is None else _expr(stmt.value, ctx)
    if ctx.entry_point:
        ctx.set_block_jump_with_result(hir.Jump.RETURN, return_val)
    else:
        ctx.store("$retval", return_val)
        ctx.store("$returning", True)
        ctx.set_block_jump(hir.Jump.BREAK)


@_register(_stmt_handlers, ast.Pass)
def _pass_stmt(stmt: ast.Pass, ctx: _Context) -> None:
    pass


def _make_closure(node: ast.FunctionDef | ast.Lambda, ctx: _Context) -> hir.Value:
    signature, default_exprs = _signature_from_ast_arguments(node.args)
    default_values = tuple(_expr(x, ctx) for x in default_exprs)
    line_no = node.lineno
    name = None if isinstance(node, ast.Lambda) else node.name
    desc = FunctionDesc(name, ctx.filename, line_no)
    new_locals, new_globals, _ = ast_get_all_local_names(node)
    local_names = (ctx.local_names - new_globals) | new_locals
    new_ctx = _Context(ctx.filename, ctx.first_line, desc, ctx.frozen_globals,
                       local_names, entry_point=False)

    func_hir = _get_function_hir_inner(node, signature, new_ctx)
    func_hir.code_object = (_compile_lambda(node, ctx.filename) if isinstance(node, ast.Lambda)
                            else _compile_nested_function_def(node, ctx.filename)).__code__

    ctx.nested_functions.append(func_hir)
    return ctx.call(hir.make_closure, (func_hir, *default_values))


def _compile_lambda(node: ast.Lambda, filename: str):
    node = ast.Lambda(args=_flatten_parameters(node.args), body=node.body,
                      lineno=node.lineno, col_offset=node.col_offset,
                      end_lineno=node.end_lineno, end_col_offset=node.end_col_offset)
    container = ast.Expression(node)
    code = compile(container, filename, "eval")
    return eval(code, {}, {})


def _compile_nested_function_def(node: ast.FunctionDef, filename: str):
    node = ast.FunctionDef(name=node.name, args=_flatten_parameters(node.args),
                           body=node.body, decorator_list=[],
                           returns=None, type_comment=None,
                           lineno=node.lineno, col_offset=node.col_offset,
                           end_lineno=node.end_lineno, end_col_offset=node.end_col_offset)
    container = ast.Module(body=[node], type_ignores=[])
    code = compile(container, filename, "exec")
    locals_dict = {}
    exec(code, {}, locals_dict)
    return locals_dict[node.name]


def _flatten_parameters(args: ast.arguments) -> ast.arguments:
    all_args = args.posonlyargs + args.args
    if args.vararg is not None:
        all_args.append(args.vararg)
    all_args += args.kwonlyargs
    if args.kwarg is not None:
        all_args.append(args.kwarg)
    return ast.arguments(posonlyargs=[], args=all_args, vararg=None, kwonlyargs=[], kw_defaults=[],
                         kwarg=None, defaults=[])


@_register(_stmt_handlers, ast.FunctionDef)
def _function_def_stmt(stmt: ast.FunctionDef, ctx: _Context) -> None:
    if len(stmt.decorator_list) > 0:
        raise ctx.syntax_error("Decorators on nested functions are not supported")

    closure = _make_closure(stmt, ctx)
    ctx.store(stmt.name, closure)


def _signature_from_ast_arguments(aa: ast.arguments) \
        -> tuple[inspect.Signature, list[ast.expr]]:
    def make_default_placeholder(default_expr: ast.expr) -> ClosureDefaultPlaceholder:
        ret = ClosureDefaultPlaceholder(len(all_default_exprs))
        all_default_exprs.append(default_expr)
        return ret

    all_default_exprs: list[ast.expr] = []
    all_params = []
    for p in aa.posonlyargs:
        all_params.append((p, inspect.Parameter.POSITIONAL_ONLY))
    for p in aa.args:
        all_params.append((p, inspect.Parameter.POSITIONAL_OR_KEYWORD))

    # Defaults for POSITIONAL_ONLY & POSITIONAL_OR_KEYWORD
    num_pos_params = len(all_params)
    defaults: list[Any] = [inspect.Parameter.empty] * (num_pos_params - len(aa.defaults))
    for def_expr in aa.defaults:
        defaults.append(make_default_placeholder(def_expr))

    # *args
    if aa.vararg is not None:
        all_params.append((aa.vararg, inspect.Parameter.VAR_POSITIONAL))
        defaults.append(inspect.Parameter.empty)

    # Keyword-only parameters
    for p, def_expr in zip(aa.kwonlyargs, aa.kw_defaults, strict=True):
        all_params.append((p, inspect.Parameter.KEYWORD_ONLY))
        defaults.append(inspect.Parameter.empty if def_expr is None
                        else make_default_placeholder(def_expr))

    # **kwargs
    if aa.kwarg is not None:
        all_params.append((aa.kwarg, inspect.Parameter.VAR_KEYWORD))
        defaults.append(inspect.Parameter.empty)

    parameters = tuple(inspect.Parameter(p.arg, kind, default=default)
                       for (p, kind), default in zip(all_params, defaults, strict=True))
    return inspect.Signature(parameters), all_default_exprs


def _unsupported_stmt(stmt: ast.AST, ctx: _Context) -> None:
    raise ctx.unsupported_syntax()


def _stmt(stmt: ast.AST, ctx: _Context) -> None:
    handler = _stmt_handlers.get(type(stmt), _unsupported_stmt)
    with ctx.change_loc(stmt):
        handler(stmt, ctx)


def _stmt_list(statements: Sequence[ast.stmt], ctx: _Context):
    statements = iter(statements)
    for stmt in statements:
        _stmt(stmt, ctx)
        if ctx.current_block.jump is not None:
            break

    # Process "dead" statements, i.e. the ones after a jump ("continue"/"break"/"return").
    # We still need to look at them in order to figure out the set of local variables.
    # So create a throwaway block to store these into.
    with ctx.new_block():
        for stmt in statements:
            _stmt(stmt, ctx)


def _get_all_parameters(func_def: ast.FunctionDef | ast.Lambda, ctx: _Context) -> List[ast.arg]:
    for a in (func_def.args.vararg, func_def.args.kwarg):
        if a is not None:
            raise ctx.syntax_error(
                "Variadic parameters in user-defined functions are not supported", a)
    all_args = []
    for arg in func_def.args.posonlyargs:
        all_args.append(arg)
    for arg in func_def.args.args:
        all_args.append(arg)
    for arg in func_def.args.kwonlyargs:
        all_args.append(arg)
    return all_args


def _ast2hir(func_def: ast.FunctionDef | ast.Lambda, ctx: _Context) -> hir.Block:
    with ctx.change_loc(func_def), ctx.new_block() as root_block:
        if ctx.entry_point:
            assert isinstance(func_def, ast.FunctionDef)
            _stmt_list(func_def.body, ctx)
            # Add a Return jump to the root block if it doesn't have one
            if root_block.jump is None:
                ctx.set_block_jump(hir.Jump.RETURN)
        elif isinstance(func_def, ast.FunctionDef):
            # To enable early returns in a helper function, wrap the body in a loop.
            # Thus, we can use "break" to implement the return statement.
            ctx.store("$returning", False)
            with ctx.new_block() as body_block:
                _stmt_list(func_def.body, ctx)
                if body_block.jump is None:
                    ctx.store("$retval", None)
                    ctx.set_block_jump(hir.Jump.BREAK)

            ctx.call_void(hir.loop, (body_block, None))
            root_block.result = ctx.load("$retval")
            root_block.have_result = True
        else:
            assert isinstance(func_def, ast.Lambda)
            root_block.result = _expr(func_def.body, ctx)
            root_block.have_result = True

    return root_block
