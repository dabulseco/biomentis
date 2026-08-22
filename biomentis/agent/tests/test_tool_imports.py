"""Tests for the tool-import repair applied before every executed block.

The agent writes its own imports from the function dictionary in the system
prompt, and with 200+ tools over 22 sibling modules it misfiles them — the
reported failure was

    Error: cannot import name 'query_pubmed' from 'biomentis.tool.database'

`fix_tool_imports` re-points such a statement at the module that actually
defines the function. These tests pin down:

  1. Tool names are unique across the modules — the premise that makes a
     name-only lookup unambiguous
  2. A misfiled import is re-pointed at the owning module
  3. A correct import is left alone
  4. One statement naming tools from several modules splits per module
  5. `as` aliases and indentation survive the rewrite
  6. Unregistered names (private helpers) stay on the module the agent named,
     so a real typo still raises
  7. Stale `biomni.tool.*` paths are normalized to `biomentis.tool.*`
  8. Code with no tool imports comes back byte-identical
  9. The repaired import actually executes through `run_python_repl`

Run with:
    python -m biomentis.agent.tests.test_tool_imports
"""

from __future__ import annotations

import sys

from biomentis.tool.tool_imports import fix_tool_imports, tool_name_to_module
from biomentis.utils import read_module2api

# --- Tiny test harness (matches test_tab_focus.py) -----------------------

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


# --- Tests ---------------------------------------------------------------


def test_tool_names_are_unique_across_modules() -> None:
    print("\n[1] tool names are unique across modules")
    seen: dict[str, str] = {}
    collisions: list[str] = []
    for module_name, apis in read_module2api().items():
        for api in apis:
            name = api["name"]
            if name in seen and seen[name] != module_name:
                collisions.append(f"{name}: {seen[name]} vs {module_name}")
            seen[name] = module_name
    _check(
        "no tool name is defined in two modules",
        not collisions,
        f"collisions={collisions[:5]}",
    )


def test_misfiled_import_is_repointed() -> None:
    print("\n[2] a misfiled import is re-pointed")
    fixed = fix_tool_imports("from biomentis.tool.database import query_pubmed\n")
    _check(
        "query_pubmed is imported from literature",
        fixed.strip() == "from biomentis.tool.literature import query_pubmed",
        repr(fixed),
    )


def test_correct_import_is_untouched() -> None:
    print("\n[3] a correct import is untouched")
    src = "from biomentis.tool.database import query_kegg, query_uniprot\n"
    _check("unchanged", fix_tool_imports(src) == src, repr(fix_tool_imports(src)))


def test_mixed_import_splits_per_module() -> None:
    print("\n[4] a mixed import splits per module")
    fixed = fix_tool_imports("from biomentis.tool.database import query_pubmed, query_kegg\n")
    lines = [line for line in fixed.splitlines() if line.strip()]
    _check(
        "one statement per owning module",
        lines
        == [
            "from biomentis.tool.literature import query_pubmed",
            "from biomentis.tool.database import query_kegg",
        ],
        repr(lines),
    )


def test_alias_and_indentation_survive() -> None:
    print("\n[5] aliases and indentation survive")
    fixed = fix_tool_imports("def f():\n    from biomentis.tool.database import query_pubmed as pm\n")
    _check(
        "alias kept and indent preserved",
        "    from biomentis.tool.literature import query_pubmed as pm" in fixed,
        repr(fixed),
    )


def test_unknown_name_is_left_alone() -> None:
    print("\n[6] unregistered names stay put")
    src = "from biomentis.tool.database import _query_ncbi_database\n"
    _check("private helper untouched", fix_tool_imports(src) == src, repr(fix_tool_imports(src)))

    typo = "from biomentis.tool.database import query_pubmedd\n"
    _check("a typo is not silently rehomed", fix_tool_imports(typo) == typo, repr(fix_tool_imports(typo)))


def test_stale_package_name_is_normalized() -> None:
    print("\n[7] biomni.tool.* is normalized")
    fixed = fix_tool_imports("from biomni.tool.database import query_kegg\n")
    _check(
        "renamed to biomentis",
        fixed.strip() == "from biomentis.tool.database import query_kegg",
        repr(fixed),
    )


def test_unrelated_code_is_identical() -> None:
    print("\n[8] unrelated code is byte-identical")
    src = "import pandas as pd\nfrom collections import Counter\nprint('hi')\n"
    _check("unchanged", fix_tool_imports(src) == src, repr(fix_tool_imports(src)))


def test_repl_executes_a_misfiled_import() -> None:
    print("\n[9] run_python_repl executes a misfiled import")
    from biomentis.tool.support_tools import run_python_repl

    out = run_python_repl(
        "from biomentis.tool.database import query_pubmed\nprint('imported', query_pubmed.__module__)"
    )
    _check(
        "no ImportError, and the real function is bound",
        "Error:" not in out and "biomentis.tool.literature" in out,
        repr(out),
    )


def test_mapping_covers_the_registry() -> None:
    print("\n[10] the name→module map covers every registered tool")
    mapping = tool_name_to_module()
    total = sum(len(apis) for apis in read_module2api().values())
    _check("every tool is mapped", len(mapping) == total, f"{len(mapping)} mapped vs {total} tools")


def run_all() -> int:
    print("=" * 60)
    print("Tool import repair tests")
    print("=" * 60)
    test_tool_names_are_unique_across_modules()
    test_misfiled_import_is_repointed()
    test_correct_import_is_untouched()
    test_mixed_import_splits_per_module()
    test_alias_and_indentation_survive()
    test_unknown_name_is_left_alone()
    test_stale_package_name_is_normalized()
    test_unrelated_code_is_identical()
    test_repl_executes_a_misfiled_import()
    test_mapping_covers_the_registry()

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
