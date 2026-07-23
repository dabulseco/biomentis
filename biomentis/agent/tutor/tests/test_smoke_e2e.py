"""End-to-end smoke test for the Biomentis-Tutor pipeline.

Exercises:
  1. KnowledgeBase.add_files() with a real text file
  2. KnowledgeBase.retrieve() returning relevant chunks
  3. InstructionGenerator generating a card (uses a real LLM if available)
  4. TutorChat.ask() returning an answer + classification
  5. SessionLogger persisting step + qa records
  6. Log exports (JSON, Steps CSV, Q&A CSV) all succeeding

Usage:
    python -m biomentis.agent.tutor.tests.test_smoke_e2e
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path


# --- Tiny test harness (no pytest) ---------------------------------------

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


# --- A nanobody-shaped course note ---------------------------------------

_COURSE_NOTE = """\
CHIKV E2 Epitope Mapping and Nanobody Discovery — Study Notes
==============================================================

Section 1: Why E2?
-----------------
The E2 envelope glycoprotein of Chikungunya virus (CHIKV) is the primary
target of neutralizing antibodies. Its ectodomain (residues 1–361) is
exposed on the virion surface and is the most accessible region for both
conventional antibodies and camelid-derived nanobodies (VHH domains,
~15 kDa). E2 is a class II fusion protein that exists as a heterodimer
with E1 on the viral surface.

Section 2: Epitope conservation
-------------------------------
For a broadly neutralizing nanobody, the target epitope should be
conserved across genotypes. CHIKV has three major genotypes (West
African, Asian, and East/Central/South African — ECSA). A multiple
sequence alignment (MSA) of E2 from representative isolates from each
genotype lets you identify columns that are invariant or near-invariant.
Residues with >95% identity across all genotypes are good candidates.

Section 3: Tools and databases
------------------------------
- UniProt: canonical sequence (accession Q5WBD5 for CHIKV strain S27).
- RCSB PDB: experimentally determined E2 structures (e.g. 6NK5, 6NK6).
- PyMOL / ChimeraX: structure visualization.
- BLAST: finding homologs in PDB to seed homology-based docking.
- ClusPro / HADDOCK: protein-protein docking servers.

Section 4: Nanobody selection workflow
--------------------------------------
1. Identify conserved epitope on E2 (MSA + SASA filter).
2. Find or design 3–5 CDR sets (H1, H2, H3) that bind the epitope.
3. Predict 3D models of the nanobody (e.g. via AlphaFold-Multimer or
   nanobody-specific tools like NanoNet).
4. Dock the nanobody to E2 (ClusPro, HADDOCK, or AutoDock CrankPep).
5. Filter docking poses by binding energy and interface contacts.
6. Pick top 3 candidates for downstream experimental validation.

Section 5: What "good" looks like
---------------------------------
- BLAST E-value < 1e-10 against PDB hits indicates strong homology.
- A conserved epitope typically has 5–15 surface residues within a 6 Å
  patch, with side-chain solvent-accessible surface area > 40 Å².
- Nanobody–E2 binding free energy (ΔG) below -8 kcal/mol is typically
  considered "strong" by docking scoring functions.
"""


# --- Real LLM call (Ollama) -----------------------------------------------


def _try_get_llm():
    """Return a working LangChain chat model, or None if unavailable."""
    try:
        from biomentis.llm import get_llm
    except Exception:
        return None

    # Try Ollama first (local). If `ollama serve` is up and the model is
    # pulled, this returns a real LangChain chat model.
    try:
        llm = get_llm(source="Ollama")
        if llm is not None:
            # Probe with a tiny prompt — if it fails, fall back to stub.
            try:
                from langchain_core.messages import HumanMessage
                llm.invoke([HumanMessage(content="ping")])
                return llm
            except Exception:
                pass
    except Exception:
        pass
    return None


# --- Test stubs (used when no LLM is reachable) --------------------------


class _StubLLM:
    """A minimal LangChain-shaped stub for offline smoke tests."""

    def invoke(self, messages, *args, **kwargs):
        from langchain_core.messages import AIMessage

        # Find the system + human content
        sys_text = ""
        user_text = ""
        for m in messages:
            t = getattr(m, "content", str(m))
            if "tutor" in t.lower() or "return a single json" in t.lower():
                sys_text = t
            else:
                user_text = t

        # Look at both texts — the stub has to disambiguate between the
        # three prompts the tutor sends: instruction card, classify_qa,
        # and tutor chat. They're differentiated by signals in the
        # system prompt AND the user prompt.
        combined = (sys_text + "\n" + user_text).lower()
        user_lower = user_text.lower()

        if (
            "rubric:" in user_lower
            and "question:" in user_lower
            and "answer:" in user_lower
        ):
            # classify_qa prompt (Rubric.classify_qa)
            return AIMessage(content=json.dumps({
                "bloom_level": "Apply",
                "dok_level": 2,
                "rubric_hit": ["GEN2"],
                "confidence": 0.78,
            }))
        if "question:" in user_lower and "kb_snippets" in user_lower:
            # TutorChat prompt
            return AIMessage(content=json.dumps({
                "answer": "BLAST finds homologous protein sequences in a database. The E-value tells you how likely the match is by chance; below 1e-10 is strong.",
                "follow_up": "Why is a low E-value more useful than a high percent identity?",
                "citations": [
                    {"source": "course_notes.txt", "page": None, "snippet": "BLAST E-value < 1e-10 against PDB hits indicates strong homology."},
                ],
            }))
        if "event_type" in user_lower and "look_for" in sys_text.lower():
            # Instruction card prompt
            return AIMessage(content=json.dumps({
                "what": "Running a BLAST search to find homologs of CHIKV E2 in the PDB.",
                "why": "Homologs in the PDB are required as templates for homology-based docking of the nanobody candidates.",
                "prerequisites": ["BLAST E-value interpretation", "PDB file parsing"],
                "look_for": ["A non-empty DataFrame with hit_id, e_value, identity columns"],
                "citations": [
                    {"source": "course_notes.txt", "page": None, "snippet": "BLAST E-value < 1e-10 against PDB hits indicates strong homology."}
                ],
                "bloom_target": "Apply",
                "dok_target": 2,
            }))
        # Default — emit valid chat JSON so the test can at least exercise the parser.
        return AIMessage(content=json.dumps({
            "answer": "(stub answer)",
            "follow_up": "",
            "citations": [],
        }))


# --- The actual smoke test -----------------------------------------------


def run_smoke_test() -> int:
    print("=" * 60)
    print("Biomentis-Tutor end-to-end smoke test")
    print("=" * 60)

    # 1. Set up an isolated temp dir for the KB + log.
    tmp = tempfile.mkdtemp(prefix="biomni_tutor_smoke_")
    raw_dir = Path(tmp) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    note_path = raw_dir / "course_notes.txt"
    note_path.write_text(_COURSE_NOTE)

    session_id = "smoke-e2e"
    path = tmp

    print(f"\n[setup] session_id={session_id} path={path}")
    print(f"[setup] wrote {len(_COURSE_NOTE)} bytes of course notes to {note_path}")

    # 2. Build the engine.
    from biomentis.agent.tutor import TutorEngine

    real_llm = _try_get_llm()
    if real_llm is not None:
        print("[setup] LLM: real Ollama model")
        llm = real_llm
    else:
        print("[setup] LLM: stub (no Ollama reachable — using offline stub)")
        llm = _StubLLM()

    try:
        engine = TutorEngine(session_id=session_id, llm=llm, path=path)
    except Exception as e:
        traceback.print_exc()
        _check("TutorEngine construction", False, str(e))
        return _finish()

    _check("TutorEngine construction", True)

    # 3. Ingest the KB.
    try:
        chunks = engine.kb.add_files([str(note_path)])
    except Exception as e:
        traceback.print_exc()
        _check("KnowledgeBase.add_files()", False, str(e))
        return _finish()

    _check("KnowledgeBase.add_files()", chunks > 0, f"chunks added = {chunks}")

    stats = engine.kb.stats()
    _check("KnowledgeBase.stats() shows ≥1 source", stats.sources >= 1, f"stats={stats}")

    # 4. Retrieve.
    try:
        docs = engine.kb.retrieve("BLAST E-value cutoff for homology", k=3)
    except Exception as e:
        traceback.print_exc()
        _check("KnowledgeBase.retrieve()", False, str(e))
        return _finish()

    _check("KnowledgeBase.retrieve() returns hits", len(docs) > 0, f"hits={len(docs)}")

    # 5. Generate an instruction card.
    from biomentis.ui_core import UIEvent
    ev = UIEvent(
        type="code",
        content="run_blast('CHIKV E2', db='pdb')",
        title="Run BLAST against PDB",
    )

    try:
        card = engine.instruction_gen.generate(ev, task="Find 3 candidate nanobodies against CHIKV E2.")
    except Exception as e:
        traceback.print_exc()
        _check("InstructionGenerator.generate()", False, str(e))
        return _finish()

    _check("InstructionGenerator.generate() returns card", card is not None)
    _check("Card has 'what' field", bool(getattr(card, "what", "")), f"what='{card.what}'")
    _check("Card has 'why' field", bool(getattr(card, "why", "")), f"why='{card.why}'")
    _check("Card has 'bloom_target'", bool(getattr(card, "bloom_target", "")), f"bloom='{card.bloom_target}'")
    _check("Card has 'dok_target'", getattr(card, "dok_target", 0) > 0, f"dok={card.dok_target}")
    # Log the step manually (this is what the wrapper does).
    try:
        engine.logger.log({
            "kind": "step",
            "event_type": ev.type,
            "step_id": 1,
            "title": ev.title,
            "bloom_target": card.bloom_target,
            "dok_target": card.dok_target,
            "instruction": {
                "what": card.what,
                "why": card.why,
                "prerequisites": card.prerequisites,
                "look_for": card.look_for,
            },
            "kb_citations": card.citations,
            "generation_failed": False,
        })
        _check("SessionLogger.log(step) succeeds", True)
    except Exception as e:
        _check("SessionLogger.log(step) succeeds", False, str(e))

    # 6. Tutor chat.
    try:
        turn = engine.chat.ask(
            question="Why does the E2 ectodomain make a good nanobody target?",
            context="step 1: BLAST search",
            task="Find 3 candidate nanobodies against CHIKV E2.",
        )
    except Exception as e:
        traceback.print_exc()
        _check("TutorChat.ask()", False, str(e))
        return _finish()

    _check("TutorChat.ask() returns a turn", turn is not None)
    _check("Turn has content", bool(turn.content), f"content='{turn.content[:80]}...'")
    _check(
        "Citation source is whitelisted to KB",
        all(
            c.get("source") == "course_notes.txt"
            for c in (turn.citations or [])
        ),
        f"citations={turn.citations}",
    )
    _check("Turn classified with Bloom level", bool(turn.bloom_level), f"bloom={turn.bloom_level}")
    _check("Turn classified with DOK level", int(turn.dok_level) > 0, f"dok={turn.dok_level}")
    if not turn.failed:
        _check("Turn has confidence > 0 when not failed", turn.confidence > 0, f"conf={turn.confidence}")

    # 7. Log inspection — JSONL has both records.
    log_path = Path(engine.logger.path)
    if log_path.exists():
        with log_path.open() as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    else:
        lines = []
    _check("JSONL log exists", log_path.exists(), f"path={log_path}")
    _check("JSONL has ≥2 records (1 step + 1 qa)", len(lines) >= 2, f"lines={len(lines)}")

    parsed = []
    for ln in lines:
        try:
            parsed.append(json.loads(ln))
        except Exception as e:
            _check("JSONL lines are valid JSON", False, f"{e}: {ln[:100]}")
    if len(parsed) == len(lines):
        _check("All JSONL lines are valid JSON", True)

    kinds = {r.get("kind") for r in parsed}
    _check("JSONL contains 'step' record", "step" in kinds, f"kinds={kinds}")
    _check("JSONL contains 'qa' record", "qa" in kinds, f"kinds={kinds}")

    # 8. Exports.
    try:
        export_str = engine.logger.export_json()
        export_ok = isinstance(export_str, str) and len(export_str) > 0
    except Exception as e:
        export_ok = False
        _check("export_json() runs", False, str(e))
    _check("export_json() returns a non-empty string", export_ok)

    # Persist the export next to the JSONL for inspection.
    try:
        export_path = Path(tmp) / "export.json"
        export_path.write_text(export_str)
        _check("Export written to export.json", export_path.exists() and export_path.stat().st_size > 0)
    except Exception as e:
        _check("Export written to export.json", False, str(e))

    try:
        steps_csv, qa_csv = engine.logger.export_csv()
        _check(
            "export_csv() returns 2 strings",
            isinstance(steps_csv, str) and isinstance(qa_csv, str),
        )
        _check("Steps CSV has step header", "bloom_target" in steps_csv, f"steps_csv head: {steps_csv[:120]}")
        _check("Q&A CSV has qa header", "bloom_level" in qa_csv, f"qa_csv head: {qa_csv[:120]}")
        _check("Steps CSV has at least 1 step row", steps_csv.count("\n") >= 2)
        _check("Q&A CSV has at least 1 qa row", qa_csv.count("\n") >= 2)
    except Exception as e:
        _check("export_csv() runs", False, str(e))

    # 9. TutorEngine + rubric swap roundtrip.
    from biomentis.agent.tutor.rubric import Rubric
    new_rubric = Rubric.default()
    new_rubric.objectives[0].description = "Modified description for smoke test"
    engine.set_rubric(new_rubric)
    _check("TutorEngine.set_rubric() works", engine.rubric is new_rubric)

    # 10. TutorEngine disable/enable.
    engine.enable()
    _check("TutorEngine.enable() flips enabled", engine.is_enabled() is True)
    engine.disable()
    _check("TutorEngine.disable() flips enabled", engine.is_enabled() is False)

    return _finish(tmp=tmp)


def _finish(tmp: str | None = None) -> int:
    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    if _FAILED:
        print("\nFailures:")
        for name, detail in _FAILED:
            print(f"  - {name}: {detail}")
    if tmp:
        # Leave the tmp dir on disk for inspection (a teacher can find it).
        print(f"\nSession artifacts left at: {tmp}")
    return 0 if not _FAILED else 1


if __name__ == "__main__":
    sys.exit(run_smoke_test())
