"""Static sweep for the D019 finding-1 class: a function that loads a global name
the module never binds and builtins does not provide is a guaranteed ``NameError``.

``tests/test_correctness.py`` covers one critical function by walking bytecode; this
sweep covers every module in ``src/`` at once, because the failure only surfaces when
the affected stage actually runs — hours into a GPU chain.
"""

from __future__ import annotations

import builtins
import symtable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Implicit bindings the import system provides; symtable does not model them.
IMPLICIT = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
            "__loader__", "__builtins__"}


def _module_bound_names(table: symtable.SymbolTable) -> set[str]:
    return {s.get_name() for s in table.get_symbols()
            if s.is_assigned() or s.is_imported() or s.is_namespace()}


def _undefined_global_loads(table: symtable.SymbolTable, bound: set[str]) -> list[str]:
    """True global loads — referenced, not local, not free — must resolve at module
    scope or in builtins."""
    found = []
    for sym in table.get_symbols():
        if sym.is_referenced() and not sym.is_local() and not sym.is_free():
            name = sym.get_name()
            if name not in bound and not hasattr(builtins, name):
                found.append(name)
    for child in table.get_children():
        found.extend(_undefined_global_loads(child, bound))
    return found


def test_no_module_loads_an_undefined_global():
    modules = sorted(SRC.rglob("*.py"))
    assert modules, "src/ has modules to sweep"
    failures = {}
    for path in modules:
        table = symtable.symtable(path.read_text(), str(path), "exec")
        bound = _module_bound_names(table) | IMPLICIT
        names = _undefined_global_loads(table, bound)
        if names:
            failures[str(path.relative_to(ROOT))] = sorted(set(names))
    assert not failures, f"undefined global loads: {failures}"
