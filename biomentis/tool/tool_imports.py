"""Repair the import statements the agent writes for Biomentis tools.

The agent composes its own ``from ... import ...`` lines from the function
dictionary in the system prompt. With 200+ tools spread across 22 sibling
modules it regularly attributes a function to the wrong one: `query_pubmed`
lives in `biomentis.tool.literature`, but reads like a database tool, so the
model asks `biomentis.tool.database` for it and the block dies on

    Error: cannot import name 'query_pubmed' from 'biomentis.tool.database'

Tool names are unique across the 22 modules, so the owning module is always
recoverable from the name alone. `fix_tool_imports` rewrites such an import to
point at the module that actually defines the function; `run_python_repl` runs
every block through it before executing.

Only `from <pkg>.<module> import <names>` statements are touched. A name that
isn't a registered tool (a private helper, say) is left on whatever module the
agent named, so a genuinely missing attribute still raises as before.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Both spellings show up: the package was renamed from `biomni` to
# `biomentis`, and the old name still leaks into generated code.
_TOOL_PKGS = ("biomentis.tool", "biomni.tool")

_IMPORT_RE = re.compile(
    r"^(?P<indent>[ \t]*)from[ \t]+(?P<mod>(?:biomentis|biomni)\.tool\.[A-Za-z_][\w.]*)"
    r"[ \t]+import[ \t]+(?P<targets>\([^()]*\)|[^\n#]+)",
    re.MULTILINE,
)


@lru_cache(maxsize=1)
def tool_name_to_module() -> dict[str, str]:
    """Map every registered tool name to the module that defines it."""
    try:
        from biomentis.utils import read_module2api

        module2api = read_module2api()
    except Exception:  # a missing description module must not break execution
        return {}

    mapping: dict[str, str] = {}
    for module_name, apis in module2api.items():
        for api in apis:
            name = api.get("name") if isinstance(api, dict) else None
            if name:
                # First writer wins; names are unique across the tool modules.
                mapping.setdefault(name, module_name)
    return mapping


def _normalize_pkg(module: str) -> str:
    """Rewrite a stale `biomni.tool.*` path to its `biomentis.tool.*` twin."""
    if module.startswith("biomni.tool."):
        return "biomentis.tool." + module[len("biomni.tool.") :]
    return module


def _split_targets(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1]
    return [t.strip() for t in raw.split(",") if t.strip()]


def _imported_name(target: str) -> str:
    """The name being imported, ignoring any `as` alias."""
    return re.split(r"[ \t]+as[ \t]+", target, maxsplit=1)[0].strip()


def fix_tool_imports(code: str) -> str:
    """Point each tool import at the module that actually defines the tool.

    An import naming several tools is split into one statement per owning
    module. Statements that are already correct come back untouched.
    """
    if not any(pkg in code for pkg in _TOOL_PKGS):
        return code

    mapping = tool_name_to_module()

    def _rewrite(match: re.Match[str]) -> str:
        declared = _normalize_pkg(match.group("mod"))
        targets = match.group("targets")

        if "*" in targets:
            return f"{match.group('indent')}from {declared} import {targets.strip()}"

        groups: dict[str, list[str]] = {}
        for target in _split_targets(targets):
            owner = mapping.get(_imported_name(target), declared)
            groups.setdefault(owner, []).append(target)

        if not groups:
            return match.group(0)

        indent = match.group("indent")
        return "\n".join(f"{indent}from {mod} import {', '.join(names)}" for mod, names in groups.items())

    return _IMPORT_RE.sub(_rewrite, code)
