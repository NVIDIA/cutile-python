#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import sys
import pathlib
import warnings
from typing import Dict, NamedTuple, Any
from functools import lru_cache


ROOT = pathlib.Path(__file__).resolve().parents[2]
KERNELS_DIR = ROOT / "test" / "kernels"
SAMPLES_DIR = ROOT / "samples"
SAMPLES_TEMPLATES_DIR = SAMPLES_DIR / "templates"


if sys.version_info >= (3, 12, 0, 0, 0):
    TypeAlias = ast.TypeAlias
else:
    class TypeAlias:
        pass


def _used_names_in_function(fn: ast.FunctionDef) -> set[str]:
    """Collect names read in body, annotations, decorators, defaults;
       also include attribute bases like `ct` in `ct.load`."""
    used: set[str] = set()
    locals_: set[str] = {a.arg for a in (fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs)}
    if fn.args.vararg:
        locals_.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        locals_.add(fn.args.kwarg.arg)

    class V(ast.NodeVisitor):
        def visit_Name(self, n: ast.Name):
            if isinstance(n.ctx, ast.Load):
                used.add(n.id)
            elif isinstance(n.ctx, ast.Store):
                locals_.add(n.id)

        def visit_Attribute(self, n: ast.Attribute):
            # capture the base name in ct.foo
            if isinstance(n.value, ast.Name):
                used.add(n.value.id)
            self.generic_visit(n)

        def visit_arg(self, n: ast.arg):
            if n.annotation:
                self.visit(n.annotation)

        def visit_FunctionDef(self, n: ast.FunctionDef):
            for d in n.decorator_list:
                self.visit(d)
            for a in n.args.posonlyargs + n.args.args + n.args.kwonlyargs:
                self.visit(a)
            if n.returns:
                self.visit(n.returns)
            for s in n.body:
                self.visit(s)

    V().visit(fn)
    used.discard(fn.name)
    used.difference_update(locals_)
    return used


class NodeRef(NamedTuple):
    path: pathlib.Path | None
    node: Any
    seq: int  # first-seen order


_seq = 0


def _next_seq():
    global _seq
    _seq += 1
    return _seq


def node_ref_key(ref: NodeRef) -> tuple[str, int, int]:
    # If the path is the same, keep the lineno order.
    # Otherwise, keep the reversed seq order.
    if ref.path is None:
        return (ref.node.lineno, -ref.seq)
    return (str(ref.path) if ref.path else "", ref.node.lineno, -ref.seq)


def get_helper_nodes(
    module_file: pathlib.Path,
    node: ast.FunctionDef,
    all_func_nodes: Dict[str, ast.FunctionDef],
    all_alias_nodes: Dict[str, TypeAlias | ast.Assign | ast.AnnAssign],
    all_import_nodes: Dict[str, ast.ImportFrom | ast.Import],
    func_nodes_to_add: Dict[str, NodeRef],
    alias_nodes_to_add: Dict[str, NodeRef],
    import_nodes_to_add: Dict[str, NodeRef],
) -> None:
    used_names = _used_names_in_function(node)
    for name in used_names:
        if name in all_func_nodes:
            func_nodes_to_add[name] = NodeRef(module_file, all_func_nodes[name], _next_seq())
            get_helper_nodes(
                module_file, all_func_nodes[name],
                all_func_nodes, all_alias_nodes, all_import_nodes,
                func_nodes_to_add, alias_nodes_to_add, import_nodes_to_add
            )
        elif name in all_alias_nodes:
            alias_nodes_to_add[name] = NodeRef(module_file, all_alias_nodes[name], _next_seq())
        elif name in all_import_nodes:
            imp = all_import_nodes[name]
            if isinstance(imp, ast.ImportFrom):
                # If `name` came from `from MOD import name` and MOD is local, open MOD and pull it.
                imported = add_nodes_from_import_node(
                    imp, func_nodes_to_add, alias_nodes_to_add, import_nodes_to_add
                )
                if not imported:
                    import_nodes_to_add[name] = NodeRef(module_file, imp, _next_seq())
            else:
                import_nodes_to_add[name] = NodeRef(module_file, imp, _next_seq())
        elif name not in dir(__builtins__):
            warnings.warn(f"Unknown non-builtin name used in {module_file}: {name}, "
                          "it will not be inlined to the sample.")


def get_all_nodes(tree: ast.Module) -> tuple:
    all_func_nodes: Dict[str, ast.FunctionDef] = {}
    all_alias_nodes: Dict[str, TypeAlias | ast.Assign | ast.AnnAssign] = {}
    all_import_nodes: Dict[str, ast.ImportFrom | ast.Import] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            all_func_nodes[node.name] = node
        elif isinstance(node, TypeAlias):
            all_alias_nodes[node.name.id] = node
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            all_alias_nodes[node.targets[0].id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            all_alias_nodes[node.target.id] = node
        elif isinstance(node, ast.Import):
            for a in node.names:
                intro = a.asname or a.name.split(".", 1)[0]
                all_import_nodes[intro] = node
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    continue
                intro = a.asname or a.name
                all_import_nodes[intro] = node
    return all_func_nodes, all_alias_nodes, all_import_nodes


@lru_cache(maxsize=None)
def _read_text(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _split_lines(p: pathlib.Path) -> list[str]:
    return _read_text(p).splitlines()


@lru_cache(maxsize=None)
def _parse_ast(p: pathlib.Path) -> ast.Module:
    return ast.parse(_read_text(p))


def find_nodes_to_add(module_file: pathlib.Path, import_names: list[str],
                      func_nodes_to_add: Dict[str, NodeRef],
                      alias_nodes_to_add: Dict[str, NodeRef],
                      import_nodes_to_add: Dict[str, NodeRef]) -> None:
    tree = _parse_ast(module_file)
    all_func_nodes, all_alias_nodes, all_import_nodes = get_all_nodes(tree)
    for name in import_names:
        node = all_func_nodes[name]
        func_nodes_to_add[node.name] = NodeRef(module_file, node, _next_seq())
        get_helper_nodes(
            module_file, node,
            all_func_nodes, all_alias_nodes, all_import_nodes,
            func_nodes_to_add, alias_nodes_to_add, import_nodes_to_add
        )


def _resolve_local_module_file(module: str) -> pathlib.Path | None:
    if not module:
        return None
    parts = module.split(".")
    top_module = parts[0]
    base = KERNELS_DIR.parent if top_module == "kernels" else ROOT
    rel = pathlib.Path(*parts).with_suffix(".py")
    module_file = base / rel
    if module_file.exists():
        return module_file
    return None


def add_nodes_from_import_node(import_node: ast.ImportFrom,
                               func_nodes_to_add: Dict[str, NodeRef],
                               alias_nodes_to_add: Dict[str, NodeRef],
                               import_nodes_to_add: Dict[str, NodeRef]) -> bool:
    module_file = _resolve_local_module_file(import_node.module or "")
    if module_file is None:
        return False

    find_nodes_to_add(
        module_file, [name.name for name in import_node.names],
        func_nodes_to_add=func_nodes_to_add,
        alias_nodes_to_add=alias_nodes_to_add,
        import_nodes_to_add=import_nodes_to_add,
    )
    return True


def get_kernels_and_helpers_content(import_node: ast.ImportFrom, dst_tree: ast.Module) -> list[str]:
    func_nodes_to_add: Dict[str, NodeRef] = {}
    alias_nodes_to_add: Dict[str, NodeRef] = {}
    import_nodes_to_add: Dict[str, NodeRef] = {}
    imported = add_nodes_from_import_node(
        import_node, func_nodes_to_add, alias_nodes_to_add, import_nodes_to_add
    )
    assert imported, f"Failed to import nodes from {import_node.module}"
    _, dst_alias_nodes, dst_import_nodes = get_all_nodes(dst_tree)
    # Dedup alias and import nodes
    alias_nodes_to_add = {
        k: v for k, v in alias_nodes_to_add.items() if k not in dst_alias_nodes
    }
    import_nodes_to_add = {
        k: v for k, v in import_nodes_to_add.items() if k not in dst_import_nodes
    }

    res = []
    if import_nodes_to_add:
        for ref in sorted(import_nodes_to_add.values(), key=node_ref_key):
            code_lines = _split_lines(ref.path)
            res.extend(code_lines[ref.node.lineno-1:ref.node.end_lineno])
        res.extend(["", ""])
    if alias_nodes_to_add:
        for ref in sorted(alias_nodes_to_add.values(), key=node_ref_key):
            code_lines = _split_lines(ref.path)
            res.extend(code_lines[ref.node.lineno-1:ref.node.end_lineno])
        res.extend(["", ""])
    # Add codes for function nodes
    # Reverse order of the function being called to make the output deterministic.
    for i, ref in enumerate(sorted(func_nodes_to_add.values(), key=node_ref_key)):
        code_lines = _split_lines(ref.path)
        if i > 0:
            res.extend(["", ""])
        if len(ref.node.decorator_list) == 1:
            # Kernel function
            res.extend(code_lines[ref.node.decorator_list[0].lineno-1:ref.node.end_lineno])
        else:
            # Helper function
            res.extend(code_lines[ref.node.lineno-1:ref.node.end_lineno])
    return res


def _extend_with_empty_lines(lines: list[str]) -> None:
    if not lines or lines[-1] != "":
        lines.extend(["", ""])
    elif len(lines) == 1 or lines[-2] != "":
        lines.append("")


def replace_kernel_content(py: pathlib.Path, prefix: str) -> list[str]:
    """
    Replace the kernel import lines with the content of the imported modules,
    return the new code lines if any replaced, otherwise return the original code lines.
    """
    with open(py, "r") as f:
        code = f.read()
    tree = ast.parse(code)
    code_lines = code.splitlines()

    replace_map = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module.startswith(prefix):
                replace_map[node.lineno] = (
                    node.end_lineno, get_kernels_and_helpers_content(node, tree)
                )
    new_code_lines = []
    i = 0
    while i < len(code_lines):
        line_no = i + 1
        if line_no in replace_map:
            _extend_with_empty_lines(new_code_lines)
            new_code_lines.extend(replace_map[line_no][1])
            i = replace_map[line_no][0]
        else:
            new_code_lines.append(code_lines[i])
            i += 1
    return new_code_lines


def _check_or_update_file(path: pathlib.Path, expected_text: str, check: bool) -> bool:
    relative_path = path.relative_to(ROOT)
    if check:
        if not path.exists():
            print(f"[inline_samples] File {relative_path} does not exist")
            return True
        current_text = path.read_text(encoding="utf-8")
        if current_text != expected_text:
            print(f"[inline_samples] File {relative_path} is out of date")
            return True
    else:
        # Write mode: update the file.
        path.write_text(expected_text, encoding="utf-8")
        print(f"Updated {relative_path}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        default="test.kernels",
        type=str,
        help="Module prefix to inline from (default: test.kernels)"
    )
    parser.add_argument("--template-dir", type=pathlib.Path, default=SAMPLES_TEMPLATES_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write any files; instead, check that generated samples and "
            "copied files are up to date. Exit with non-zero status if changes "
            "would be made."
        ),
    )

    args = parser.parse_args()

    changes_detected = False
    for py in args.template_dir.rglob("*.py"):
        if py.name == "__init__.py":
            continue
        replaced_code_lines = replace_kernel_content(py, args.prefix)
        expected_text = "\n".join(replaced_code_lines) + "\n"
        sample_path = SAMPLES_DIR / py.relative_to(SAMPLES_TEMPLATES_DIR)
        if _check_or_update_file(sample_path, expected_text, args.check):
            changes_detected = True

    # Copy the autotuner.py to the samples directory
    autotuner_path = ROOT / "test" / "autotuner" / "autotuner.py"
    samples_autotuner_path = SAMPLES_DIR / "utils" / "autotuner.py"
    expected_autotuner_text = autotuner_path.read_text(encoding="utf-8")
    if _check_or_update_file(samples_autotuner_path, expected_autotuner_text, args.check):
        changes_detected = True

    if args.check:
        if changes_detected:
            print(
                "[inline_samples] Some files are out of date. "
                "Run without --check to regenerate them."
            )
            return 1
        else:
            print("[inline_samples] All files are up to date.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
