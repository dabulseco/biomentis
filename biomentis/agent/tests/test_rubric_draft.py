"""Tests for the per-run rubric: drafted, converted or uploaded, then frozen.

What these tests pin down:

  (a) No upload: the rubric is drafted with DRAFT_SYSTEM_PROMPT ("agent-drafted").
  (b) A canonical upload is used as is, with no model call ("uploaded").
  (c) A prose upload is converted with CONVERT_SYSTEM_PROMPT ("converted").
  (d) Replies wrapped in fences or thinking text still parse; an unreadable
      reply is asked for once more, and a second one raises.
  (e) In the agent graph the rubric is drafted once, is the same object after
      a critique round, reaches the gate, is never sent to the model, and is
      saved to the output folder (re-parses equal).
  (f) A rubric failure does not cost the run; a preset rubric skips the
      model; the upload is used by the agent; the gate off drafts nothing.
  (g) The UI shows one frozen-rubric event, keeps it in the transcript and
      the export, and the run-quality summary names the rubric source.

Run with:
    python -m biomentis.agent.tests.test_rubric_draft
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["BIOMENTIS_ARTIFACT_DIR"] = tempfile.mkdtemp(prefix="biomentis_rubric_draft_")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


BP_PROMPT = (Path(__file__).parent / "fixtures" / "robustness" / "bp_prompt.txt").read_text(encoding="utf-8")

DRAFTED = (
    "## Deliverables\n"
    "- [D1] Learning objectives for each phase\n"
    "- [D2] A detailed protocol for each phase\n\n"
    "## Constraints\n"
    "- [C1] Standard teaching-lab equipment only\n\n"
    "## Core lessons\n"
    "- [L1] Boiling-point elevation depends on particle count, not identity\n"
)
UPLOAD_CANON = (
    "# Rubric\n\n"
    "## Deliverables\n"
    "- [D1] A sample data table\n\n"
    "## Constraints\n"
    "- [C1] Flag any step needing a fume hood\n\n"
    "## Core lessons\n"
    "- [L1] Correct the boiling point to 760 mmHg\n"
)
UPLOAD_PROSE = "Students must give a data table, and the answer should explain the 760 mmHg correction."


class _FakeLLM:
    """Answers rubric requests with `replies` (in order) and quantity requests with none."""

    def __init__(self, replies: list[str] | None = None):
        self.replies = list(replies if replies is not None else [DRAFTED])
        self.rubric_requests: list[list] = []
        self.quantity_calls = 0

    def invoke(self, messages):
        from biomentis.gate.rubric import CONVERT_SYSTEM_PROMPT, DRAFT_SYSTEM_PROMPT

        system = messages[0].content if messages and isinstance(messages[0], SystemMessage) else ""
        if system in (DRAFT_SYSTEM_PROMPT, CONVERT_SYSTEM_PROMPT):
            self.rubric_requests.append(list(messages))
            if not self.replies:
                raise RuntimeError("called more times than scripted")
            return AIMessage(content=self.replies.pop(0))
        self.quantity_calls += 1
        return AIMessage(content=json.dumps({"quantities": []}))


def test_resolve() -> None:
    print("\n(a–c) resolve_rubric: drafted, uploaded, converted")
    from biomentis.gate.rubric import (
        CONVERT_SYSTEM_PROMPT,
        DRAFT_SYSTEM_PROMPT,
        prompt_sha,
        resolve_rubric,
    )

    llm = _FakeLLM()
    r = resolve_rubric(BP_PROMPT, None, llm)
    _check("no upload: agent-drafted", r.source == "agent-drafted", r.source)
    _check("one model call", len(llm.rubric_requests) == 1, str(len(llm.rubric_requests)))
    req = llm.rubric_requests[0] if llm.rubric_requests else []
    _check("drafted with DRAFT_SYSTEM_PROMPT", bool(req) and req[0].content == DRAFT_SYSTEM_PROMPT, "")
    _check("the prompt is sent", bool(req) and BP_PROMPT.strip() in req[-1].content, "")
    _check("the prompt sha is set", r.prompt_sha == prompt_sha(BP_PROMPT), r.prompt_sha)
    _check("the items are parsed", [i.id for i in r.items] == ["D1", "D2", "C1", "L1"], str(r.items))
    for word in ("deliverable", "constraint", "lesson", "3–6"):
        _check(f"the draft prompt asks for {word}", word in DRAFT_SYSTEM_PROMPT.lower(), "")

    llm = _FakeLLM([])
    r = resolve_rubric(BP_PROMPT, UPLOAD_CANON, llm)
    _check("canonical upload: uploaded", r.source == "uploaded", r.source)
    _check("no model call for it", llm.rubric_requests == [], str(len(llm.rubric_requests)))
    _check("uploaded: prompt sha set", r.prompt_sha == prompt_sha(BP_PROMPT), r.prompt_sha)

    llm = _FakeLLM()
    r = resolve_rubric(BP_PROMPT, UPLOAD_PROSE, llm)
    _check("prose upload: converted", r.source == "converted", r.source)
    req = llm.rubric_requests[0] if llm.rubric_requests else []
    _check("converted with CONVERT_SYSTEM_PROMPT", bool(req) and req[0].content == CONVERT_SYSTEM_PROMPT, "")
    _check("the upload and the prompt are both sent",
           bool(req) and UPLOAD_PROSE in req[-1].content and BP_PROMPT.strip() in req[-1].content, "")
    _check("converted: prompt sha set", r.prompt_sha == prompt_sha(BP_PROMPT), r.prompt_sha)

    llm = _FakeLLM()
    r = resolve_rubric(BP_PROMPT, "   \n", llm)
    _check("a blank upload counts as none", r.source == "agent-drafted", r.source)


def test_replies() -> None:
    print("\n(d) fences, thinking text, one retry, then an error")
    from biomentis.gate.rubric import draft_rubric

    wrapped = "<think>Let me list the items.</think>\nHere it is:\n```markdown\n# Rubric\n\n" + DRAFTED + "```\n"
    r = draft_rubric(BP_PROMPT, _FakeLLM([wrapped]))
    _check("a fenced reply with thinking parses", [i.id for i in r.items] == ["D1", "D2", "C1", "L1"], str(r))

    llm = _FakeLLM(["I think the rubric is about boiling.", DRAFTED])
    r = draft_rubric(BP_PROMPT, llm)
    _check("an unreadable reply is asked once more", len(llm.rubric_requests) == 2 and r.source == "agent-drafted",
           str(len(llm.rubric_requests)))
    retry = llm.rubric_requests[1] if len(llm.rubric_requests) == 2 else []
    _check("the retry shows the bad reply, then a correction",
           len(retry) == 4 and isinstance(retry[2], AIMessage) and isinstance(retry[3], HumanMessage), str(retry))

    llm = _FakeLLM(["nope", "still nope"])
    try:
        draft_rubric(BP_PROMPT, llm)
        raised = ""
    except ValueError as exc:
        raised = str(exc)
    _check("two unreadable replies raise a clear error", "rubric" in raised.lower(), raised)
    _check("no third call", len(llm.rubric_requests) == 2, str(len(llm.rubric_requests)))


def _agent(gate_llm, critic_replies=None, planned=1, gate=True, output_dir=None):
    from biomentis.agent.tests.test_gate_graph import _EVIDENCE_STEPS, _RIGHT, _WRONG, _Scripted
    from biomentis.agent.tests.test_gate_graph import _agent as gate_agent

    llm = _Scripted(_EVIDENCE_STEPS + ([_WRONG, _RIGHT] if planned else [_RIGHT]))
    agent = gate_agent(llm, _Scripted(critic_replies or ["Check the sum."]), planned=planned, gate=gate)
    agent.preset_rubric = None  # the graph-test helper presets one; these tests draft
    agent.gate_llm = gate_llm
    agent.output_dir = output_dir
    return agent, llm


def test_graph() -> None:
    print("\n(e) the rubric in the agent graph")
    import biomentis.agent.a1 as a1_module
    from biomentis.gate.rubric import format_rubric_markdown, parse_rubric_markdown

    out = tempfile.mkdtemp(prefix="biomentis_rubric_out_")
    gate_llm = _FakeLLM()
    agent, llm = _agent(gate_llm, output_dir=out)
    agent.run_rubric = "left over from an earlier run"
    seen: dict = {"at_critique": [], "at_gate": []}

    original_feedback = agent._creative_llm.invoke

    def critic_invoke(messages):
        seen["at_critique"].append(agent.run_rubric)
        return original_feedback(messages)

    agent._creative_llm.invoke = critic_invoke
    original_gate = a1_module.run_gate

    def spy_gate(agent_, messages, round_number, rubric=None):
        seen["at_gate"].append(rubric)
        return original_gate(agent_, messages, round_number, rubric=rubric)

    a1_module.run_gate = spy_gate
    try:
        _, final = agent.go("What is 24+3?")
    finally:
        a1_module.run_gate = original_gate
    rubric = agent.run_rubric
    _check("a rubric was drafted", rubric is not None and rubric.source == "agent-drafted", repr(rubric))
    _check("drafted once", len(gate_llm.rubric_requests) == 1, str(len(gate_llm.rubric_requests)))
    _check("a critique round ran", len(seen["at_critique"]) == 1 and len(agent.run_gate_results) == 2,
           f"{len(seen['at_critique'])} critiques, {len(agent.run_gate_results)} gates")
    _check("the same object before and after the critique round",
           bool(seen["at_critique"]) and seen["at_critique"][0] is rubric, "")
    _check("every gate got it", len(seen["at_gate"]) == 2 and all(r is rubric for r in seen["at_gate"]),
           str(seen["at_gate"]))
    leaked = [m for req in llm.requests for m in req
              if isinstance(m, HumanMessage) and "[D1]" in str(m.content)]
    _check("the rubric is never sent to the model", leaked == [], str(leaked[:1]))
    _check("the answer is kept", "24+3 = 27" in final, final)

    path = Path(out) / f"rubric-{rubric.prompt_sha[:8]}.md" if rubric else None
    _check("saved to the output folder", path is not None and path.exists(), str(list(Path(out).iterdir())))
    if path is not None and path.exists():
        text = path.read_text(encoding="utf-8")
        again = parse_rubric_markdown(text, source=rubric.source, prompt_sha=rubric.prompt_sha)
        _check("the saved file re-parses equal", again == rubric, text)
        _check("the saved file is the canonical text", text == format_rubric_markdown(rubric), text)

    from biomentis.agent.tests.test_gate_graph import _EVIDENCE_STEPS, _RIGHT

    first = rubric
    gate_llm.replies = [DRAFTED]
    llm.replies = _EVIDENCE_STEPS + [_RIGHT, _RIGHT]
    agent._creative_llm.replies = ["Looks fine."]
    agent.go("What is 24+3? Again.")
    _check("a new run drafts a new rubric", agent.run_rubric is not first and len(gate_llm.rubric_requests) == 2, "")


def test_failure_preset_upload_off() -> None:
    print("\n(f) failure, preset, upload, gate off")
    from biomentis.gate.rubric import Rubric, RubricItem

    agent, _ = _agent(_FakeLLM(["junk", "more junk"]), planned=0)
    agent.run_rubric = "stale"
    _, final = agent.go("What is 24+3?")
    _check("a failed rubric does not cost the run", "24+3 = 27" in final, final)
    _check("no rubric for that run", agent.run_rubric is None, repr(agent.run_rubric))
    _check("the run still has a status", agent.final_status == "⚠️ Not fully checked (reviewer off)",
           agent.final_status)

    preset = Rubric(items=(RubricItem("D1", "deliverable", "The total"),), source="uploaded", prompt_sha="abc12345")
    gate_llm = _FakeLLM([])
    agent, _ = _agent(gate_llm, planned=0)
    agent.preset_rubric = preset
    agent.go("What is 24+3?")
    _check("a preset rubric is used as is", agent.run_rubric is preset, repr(agent.run_rubric))
    _check("with no model call", gate_llm.rubric_requests == [], str(len(gate_llm.rubric_requests)))
    _check("the preset is a setting: kept after the run", agent.preset_rubric is preset, "")

    gate_llm = _FakeLLM([])
    agent, _ = _agent(gate_llm, planned=0)
    agent.set_rubric_upload(UPLOAD_CANON)
    agent.go("What is 24+3?")
    _check("the agent uses the upload", agent.run_rubric is not None and agent.run_rubric.source == "uploaded",
           repr(agent.run_rubric))
    _check("the upload is a setting: kept after the run", agent.rubric_upload == UPLOAD_CANON, "")
    agent.set_rubric_upload(None)
    _check("set_rubric_upload(None) clears it", agent.rubric_upload is None, repr(agent.rubric_upload))

    gate_llm = _FakeLLM([])
    agent, _ = _agent(gate_llm, planned=0, gate=False)
    agent.go("What is 24+3?")
    _check("gate off: no draft_rubric node", "draft_rubric" not in agent.app.get_graph().nodes, "")
    _check("gate off: no rubric drafted", gate_llm.rubric_requests == [] and agent.run_rubric is None,
           repr(agent.run_rubric))


def test_ui() -> None:
    print("\n(g) the rubric in the UI, transcript, export and run-quality summary")
    from biomentis.gate.rubric import format_rubric_markdown
    from biomentis.ui_core import (
        gate_summary,
        stream_agent_events,
        transcript_entry_for_event,
        ui_event_to_export_entry,
    )

    agent, _ = _agent(_FakeLLM(), planned=0)
    events = list(stream_agent_events(agent, "What is 24+3?", [], [], 71))
    rubric_events = [e for e in events if e.type == "rubric"]
    _check("one rubric event", len(rubric_events) == 1, str([e.type for e in events]))
    if rubric_events:
        e = rubric_events[0]
        _check("in the main channel, titled frozen",
               e.channel == "main" and e.title == "📋 Rubric for this run (frozen)", f"{e.channel} {e.title}")
        _check("its content is the canonical markdown", e.content == format_rubric_markdown(agent.run_rubric),
               e.content)
        entry = transcript_entry_for_event(e)
        _check("a main-panel rubric transcript entry",
               entry is not None and entry["panel"] == "main" and entry.get("kind") == "rubric"
               and entry["content"] == e.content and entry["title"] == e.title, str(entry))
        export = ui_event_to_export_entry(e)
        _check("kept in the export", export.content == e.content and export.title == e.title, str(export))
    quality = [e for e in events if e.title == "📊 Run quality"]
    _check("summary: Rubric: agent-drafted",
           bool(quality) and "**Rubric:** agent-drafted" in quality[0].content,
           quality[0].content if quality else "no summary")
    agent.run_rubric = None
    summary = gate_summary(agent)
    _check("no rubric: none", summary is not None and summary["rubric"] == "none", str(summary))

    agent, _ = _agent(_FakeLLM(["junk", "junk"]), planned=0)
    events = list(stream_agent_events(agent, "What is 24+3?", [], [], 72))
    _check("a failed rubric: no rubric event", not any(e.type == "rubric" for e in events), "")
    warnings = [e for e in events if e.type == "status" and e.title != "📊 Run quality"
                and "rubric" in (e.content or "").lower()]
    _check("a failed rubric: one warning line", len(warnings) == 1, str([e.content for e in warnings]))


def test_sidebar() -> None:
    print("\n(h) sidebar: the rubric upload")
    src = (Path(__file__).resolve().parents[3] / "streamlit_app.py").read_text(encoding="utf-8")
    flat = " ".join(src.split())
    _check("the uploader, verbatim",
           'st.file_uploader( "Rubric (optional)", type=["md", "txt", "yaml", "yml"], '
           'key="biomentis_rubric_upload",' in flat, "")
    _check("near Self-critique rounds",
           0 < src.index('"Rubric (optional)"') - src.index('"Self-critique rounds"') < 800, "")
    _check("the upload reaches the agent, decoded as UTF-8 with replacement",
           'agent.set_rubric_upload(' in flat and 'decode("utf-8", errors="replace")' in flat
           and "else None" in flat, "")

    import shutil

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[3] / "streamlit_app.py"), default_timeout=300)
    at.run()
    uploaders = [u for u in at.get("file_uploader") if u.key == "biomentis_rubric_upload"]
    _check("the app shows the uploader", len(uploaders) == 1 and not at.exception, str(at.exception))
    agent = at.session_state["biomni_agent"]
    _check("no file: no upload", agent.rubric_upload is None, repr(agent.rubric_upload))
    if uploaders:
        uploaders[0].upload("rubric.md", UPLOAD_CANON.encode("utf-8") + b"\xff", "text/markdown")
        at.run()
        _check("the uploaded text reaches the agent, bad bytes replaced",
               agent.rubric_upload == UPLOAD_CANON + "\ufffd", repr(agent.rubric_upload))
    tutor = at.session_state["biomni_tutor"]
    shutil.rmtree(tutor.kb.path, ignore_errors=True)
    shutil.rmtree(tutor.logger.dir, ignore_errors=True)


def main() -> int:
    test_resolve()
    test_replies()
    test_graph()
    test_failure_preset_upload_off()
    test_ui()
    test_sidebar()
    print(f"\n{len(_PASSED)} passed, {len(_FAILED)} failed")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
