"""Structure files (PDB/SDF/SMILES) in the knowledge base.

Only descriptive metadata is indexed -- never atom coordinates -- so the tutor
can cite a structure without the KB filling with ATOM records.

Run with:
    python -m biomentis.agent.tests.test_kb_structures
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


def _copy(name: str, root: str) -> str:
    dst = os.path.join(root, name)
    shutil.copy(os.path.join(FIX, name), dst)
    return dst


def test_kb_indexes_metadata_only():
    from biomentis.agent.tutor.kb import KnowledgeBase

    root = tempfile.mkdtemp()
    kb = KnowledgeBase("kbstruct", path=root)
    n = kb.add_files([_copy(f, root) for f in ("mini_xray.pdb", "two_mols.sdf", "aspirin.smi")])
    _check("all three indexed", n == 3, str(n))
    docs = kb.retrieve("crystal structure resolution organism", k=4)
    text = "\n".join(d.page_content for d in docs)
    _check("metadata retrievable", "Resolution" in text, text[:200])
    _check("no coordinates indexed", not re.search(r"^(ATOM|HETATM)\s", text, re.M))
    _check("sources named", {"mini_xray.pdb", "two_mols.sdf", "aspirin.smi"} <= set(kb.stats().source_names))


def test_uploader_accepts_structure_types():
    src = open(os.path.join(os.path.dirname(__file__), "..", "..", "ui_tutor.py")).read()
    _check(
        "uploader lists pdb/sdf/smi",
        '"txt", "pdb", "sdf", "smi"]' in src,
    )
    _check("indexed sources have View button", 'key=f"kbview-{name}"' in src)


def run_all() -> int:
    for test in (test_kb_indexes_metadata_only, test_uploader_accepts_structure_types):
        try:
            test()
        except Exception as e:
            _check(f"{test.__name__} ran", False, repr(e))
    print(f"\nResults: {len(_PASSED)} passed, {len(_FAILED)} failed")
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
