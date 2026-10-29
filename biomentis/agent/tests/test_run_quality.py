"""Tests for what a run actually did: the evidence check and the run-quality summary.

Two runs of one CHO-workflow prompt on deepseek-v4.1-flash:cloud (2026-10-28)
differed by 19 code steps and 3 figures against 6 steps and none. The thin one
answered without a calculation or a figure, and nothing said so.

What these tests pin down:

  1.  run_evidence counts code steps, calculations, literature tools, protocol
      lookups, figures, failed and truncated outputs, and the loop's own
      corrections and critique rounds — only for the latest run
  2.  missing_evidence names what a research answer lacks, and the nudge asks
      for exactly those things
  3.  The summary line reports the counts and warns about what is missing
  4.  With the check on, an answer with no evidence goes back, at most twice,
      then is accepted; the draft is not left in the history as an answer
  5.  An answer from a run with the evidence is accepted at once, and with
      the check off nothing changes
  6.  The critic asks for every number to be checked against code
  7.  The UI shows the summary under the answer, in the main panel and its
      export, and shows the evidence request as a status line
  8.  Printed tables are found, named by title and column header, and a
      corrected rerun replaces the earlier version
  9.  A table counts as plotted when a figure's title or axis labels match it
  10. Figures the answer cites are matched to drawn figures by number
  11. The check asks once for unplotted tables and cited-but-missing figures,
      naming them, then accepts the answer
  12. The summary warns about figures drawn but not saved, and lists tables
      without a figure
  13. The figure summary warns about a missing title and lists a diagram's
      labels; the plot guidance maps data shapes to charts

Run with:
    python -m biomentis.agent.tests.test_run_quality
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

# --- Tiny test harness (matches test_smoke_e2e.py) -----------------------

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


def _run_messages() -> list:
    from biomentis.utils import PARSE_ERROR_NUDGE

    return [
        HumanMessage(content="An earlier question."),
        AIMessage(content="<execute>print(query_pubmed('old'))</execute>"),
        AIMessage(content="<observation>old paper</observation>"),
        AIMessage(content="<solution>old answer</solution>"),
        HumanMessage(content="Design a CHO workflow."),
        AIMessage(content="<execute>print(query_pubmed('CHO fed-batch'))</execute>"),
        AIMessage(content="<observation>Title: A benchmark fed-batch process</observation>"),
        AIMessage(content="<execute>import matplotlib.pyplot as plt\nplt.plot([1, 2]); plt.show()</execute>"),
        AIMessage(content="<observation>Plot captured.\nFigure 'VCD': x = day</observation>"),
        HumanMessage(content=PARSE_ERROR_NUDGE),
        AIMessage(content="<execute>print(search_protocols('CHO'))</execute>"),
        AIMessage(content="<observation>Error: Protocols.io access token is not configured.</observation>"),
        AIMessage(content="<execute>print(read_local_protocol('x.txt'))</execute>"),
        AIMessage(content="<observation>The output is too long to be added to context. Here are the first 10K characters...\nPROTOCOL</observation>"),
        AIMessage(content="<solution>Draft</solution>"),
        HumanMessage(content="Wait... this is not enough to solve the task. Here are some feedbacks for improvement:\nmore"),
    ]


def test_run_evidence() -> None:
    print("\n[1] counting what the latest run did")
    from biomentis.run_quality import current_run, run_evidence

    msgs = _run_messages()
    _check("the run starts at the user's last prompt", current_run(msgs)[0].content == "Design a CHO workflow.",
           current_run(msgs)[0].content)
    ev = run_evidence(msgs)
    _check("code steps", ev.code_steps == 4, str(ev.code_steps))
    _check("only this run's literature", ev.literature_tools == {"query_pubmed"}, str(ev.literature_tools))
    _check("a plotting step is a calculation", ev.calculation_steps == 1, str(ev.calculation_steps))
    _check("figures", ev.figures == 1, str(ev.figures))
    _check("a failed protocols.io call isn't a lookup", ev.protocol_steps == 1, str(ev.protocol_steps))
    _check("failed steps", ev.failed_steps == 1, str(ev.failed_steps))
    _check("truncated outputs", ev.truncated_outputs == 1, str(ev.truncated_outputs))
    _check("retries", ev.retries == 1, str(ev.retries))
    _check("critique rounds", ev.critique_rounds == 1, str(ev.critique_rounds))

    claude = [
        HumanMessage(content="Q"),
        AIMessage(content="<execute>x = 6 * 7\nprint(x)</execute>"),
        HumanMessage(content="<observation>42</observation>"),
    ]
    ev = run_evidence(claude)
    _check("observations sent as user turns still count", ev.calculation_steps == 1, str(ev))
    _check("and don't start a new run", run_evidence(claude).code_steps == 1, str(ev))


def test_missing_evidence_and_nudge() -> None:
    print("\n[2] what's missing, and the request for it")
    from biomentis.run_quality import EVIDENCE_NUDGE_PREFIX, RunEvidence, evidence_nudge, missing_evidence

    _check("an empty run lacks everything",
           missing_evidence(RunEvidence()) == ["literature", "calculation", "figure"], str(missing_evidence(RunEvidence())))
    full = RunEvidence(literature_tools={"query_pubmed"}, calculation_steps=2, figures=1)
    _check("a full run lacks nothing", missing_evidence(full) == [])
    nudge = evidence_nudge(["figure"])
    _check("the nudge is recognizable", nudge.startswith(EVIDENCE_NUDGE_PREFIX))
    _check("it asks only for what's missing", "matplotlib" in nudge and "query_pubmed" not in nudge, nudge)
    _check("and for the full answer again", "<solution></solution>" in nudge and "<execute>" in nudge, nudge)


def test_summary_line() -> None:
    print("\n[3] the run-quality summary")
    from biomentis.run_quality import RunEvidence, format_quality_summary

    full = format_quality_summary(RunEvidence(code_steps=19, calculation_steps=7, literature_tools={"query_pubmed"},
                                              protocol_steps=4, figures=3, critique_rounds=1, truncated_outputs=3))
    _check("counts", "19 code steps · 7 calculations · 1 literature source · 4 protocol lookups · 3 figures · 1 critique round" in full, full)
    _check("issues", "3 outputs cut at 10,000 characters" in full, full)
    _check("plural of reply", "12 replies without a tag" in format_quality_summary(RunEvidence(retries=12)),
           format_quality_summary(RunEvidence(retries=12)))
    _check("nothing missing", "Missing" not in full, full)
    thin = format_quality_summary(RunEvidence(code_steps=6, protocol_steps=5, calculation_steps=1))
    _check("a thin run says what it lacks", "no literature source" in thin and "no figure" in thin
           and "no calculations" not in thin, thin)


class _ScriptedLLM:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.requests: list = []

    def invoke(self, messages):
        self.requests.append(list(messages))
        if not self.replies:
            raise RuntimeError("called more times than scripted")
        return AIMessage(content=self.replies.pop(0))


def _agent(llm, evidence_gate: bool):
    from biomentis.agent.a1 import A1

    agent = A1(path="./data", use_tool_retriever=False, expected_data_lake_files=[], trace=False)
    agent.llm = llm
    agent.configure(evidence_gate=evidence_gate)
    return agent


def test_gate_sends_answer_back() -> None:
    print("\n[4] an answer with no evidence goes back, at most twice")
    from biomentis.run_quality import EVIDENCE_NUDGE_PREFIX

    llm = _ScriptedLLM(["<solution>Draft A</solution>", "<solution>Draft B</solution>", "<solution>Draft C</solution>"])
    log, final = _agent(llm, evidence_gate=True).go("Design a CHO workflow.")
    _check("asked twice, then accepted", len(llm.requests) == 3, str(len(llm.requests)))
    _check("the request names what's missing",
           len(llm.requests) > 1 and llm.requests[1][-1].content.startswith(EVIDENCE_NUDGE_PREFIX),
           repr(llm.requests[1][-1:])[:200] if len(llm.requests) > 1 else "")
    _check("the final answer is the last one", "<solution>Draft C</solution>" in final, final)
    _check("drafts aren't kept as answers", not any("<solution>Draft A" in m for m in log), "")


def test_gate_accepts_a_run_with_evidence() -> None:
    print("\n[5] a run with the evidence is accepted at once; the check off changes nothing")
    llm = _ScriptedLLM([
        "<execute>def query_pubmed(q):\n    return 'Title: CHO paper'\nprint(query_pubmed('CHO'))</execute>",
        "<execute>titer = 5.0 * 1000\nprint('Plot captured.', titer)</execute>",
        "<solution>Workflow</solution>",
    ])
    _, final = _agent(llm, evidence_gate=True).go("Design a CHO workflow.")
    _check("no extra calls", len(llm.requests) == 3, str(len(llm.requests)))
    _check("answered", "<solution>Workflow</solution>" in final, final)

    llm = _ScriptedLLM(["<solution>42</solution>"])
    _, final = _agent(llm, evidence_gate=False).go("What is 6 x 7?")
    _check("check off: one call", len(llm.requests) == 1 and "42" in final, str(len(llm.requests)))


def test_critic_checks_numbers() -> None:
    print("\n[6] the critic checks every number against code")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "a1.py")
    src = open(path, encoding="utf-8").read()
    _check("asks whether each number was computed in code", "whether it was computed in" in src)
    _check("asks for units", "whether its units are right" in src)
    _check("asks for unsupported choices", "no source consulted in this run supports" in src)


def _scripted_agent(messages: list):
    history: list = []

    def _stream(inputs, stream_mode=None, config=None):
        history.extend(inputs["messages"])
        for m in messages:
            history.append(m)
            yield {"messages": list(history)}

    return SimpleNamespace(app=SimpleNamespace(stream=_stream), use_tool_retriever=False, path="/tmp", user_task="")


def test_ui_shows_summary() -> None:
    print("\n[7] the summary and the evidence request in the UI")
    from biomentis.run_quality import evidence_nudge
    from biomentis.ui_core import stream_agent_events, transcript_entry_for_event

    events = list(stream_agent_events(_scripted_agent([
        AIMessage(content="Draft without evidence"),
        HumanMessage(content=evidence_nudge(["literature", "figure"])),
        AIMessage(content="<execute>print(query_pubmed('CHO'))</execute>"),
        AIMessage(content="<observation>Title: paper</observation>"),
        AIMessage(content="<solution>Workflow</solution>"),
    ]), "Design a CHO workflow.", [], [], thread_id=1))
    statuses = [e.content for e in events if e.type == "status" and e.channel != "main"]
    _check("the request is a status line naming what's missing",
           any("missing evidence (no literature source, no figure)" in s for s in statuses), str(statuses))
    summary = [e for e in events if e.type == "status" and e.channel == "main"]
    _check("one summary, in the main panel", len(summary) == 1, str([e.content[:60] for e in summary]))
    if summary:
        _check("after the answer", events.index(summary[0]) > max(i for i, e in enumerate(events) if e.type in ("solution", "summary")))
        _check("it reports the run", "1 code step" in summary[0].content and "1 literature source" in summary[0].content,
               summary[0].content)
        entry = transcript_entry_for_event(summary[0])
        _check("its transcript entry is in the main panel, titled",
               entry["panel"] == "main" and entry.get("title") == "📊 Run quality", str(entry))
    inner = transcript_entry_for_event(SimpleNamespace(type="status", channel="inner", content="x", title=None))
    _check("other status lines stay in the inner panel", inner["panel"] == "inner" and "title" not in inner, str(inner))


_TABLE_STEP = """ITEM 2 — WORKED DSP CLEARANCE TRAIN
================================================
                step     HCP ppm  LRV this step   cum LRV
         Harvest/CCF       1e+05             --        --
    Protein A eluate         100           3.00      3.00
         Low-pH pool          30           0.52      3.52
          AEX eluate           1           1.48      5.00
     UF/DF retentate         0.3           0.52      5.52
mass balance OK"""


def test_printed_tables() -> None:
    print("\n[8] finding computed tables")
    from biomentis.run_quality import RunEvidence, _add_table, printed_tables

    tables = printed_tables(_TABLE_STEP)
    _check("one table", len(tables) == 1, str(tables))
    if tables:
        heading, size = tables[0]
        _check("named by its title and column header",
               heading.startswith("ITEM 2 — WORKED DSP CLEARANCE TRAIN [step HCP ppm"), heading)
        _check("'--' placeholders don't split it (5 rows x 3 values)", size == 15, str(size))
    _check("three numeric lines are not a table", printed_tables("a 1 2\nb 3 4\nc 5 6") == [], "")
    _check("a rule line is not a row", printed_tables("x 1 2\n" * 2 + "=====\n" + "y 1 2\n" * 2) == [], "")
    listing = "total 48\n" + "".join(
        f"-rw-r--r--@ 1 fcs3  staff  1103{i}4 Oct 29 05:4{i} fig{i}_plot.png\n" for i in range(6)
    )
    _check("an ls -l listing is not a table", printed_tables(listing) == [], str(printed_tables(listing)))
    ev = RunEvidence()
    _add_table(ev.tables, ("ITEM 2 — DSP CLEARANCE TRAIN", 15))
    _add_table(ev.tables, ("LOT HISTORY [lot ppm]", 40))
    _add_table(ev.tables, ("FIX 2 — DSP CLEARANCE TRAIN, MASS BASIS", 18))
    _check("a corrected rerun replaces the earlier version",
           [h for h, _ in ev.tables] == ["LOT HISTORY [lot ppm]", "FIX 2 — DSP CLEARANCE TRAIN, MASS BASIS"], str(ev.tables))


def _table_run(figure_line: str = "", literature: bool = False) -> list:
    code = "print(query_pubmed('HCP'))" if literature else "print(table)"
    msgs = [
        HumanMessage(content="Plan HCP control."),
        AIMessage(content=f"<execute>{code}</execute>"),
        AIMessage(content=f"<observation>{_TABLE_STEP}</observation>"),
    ]
    if figure_line:
        msgs += [
            AIMessage(content="<execute>import matplotlib.pyplot as plt\nplt.show()</execute>"),
            AIMessage(content=f"<observation>Plot captured.\n{figure_line}</observation>"),
        ]
    return msgs


def test_plotted_matching() -> None:
    print("\n[9] a table is plotted when a figure matches it")
    from biomentis.run_quality import run_evidence, unplotted_tables

    _check("no figure: unplotted", len(unplotted_tables(run_evidence(_table_run()))) == 1, "")
    by_title = "Figure 'Fig 1 — DSP clearance train': x = step [0 to 5], y = HCP (ppm) [0.1 to 1e5]"
    _check("a figure titled for it covers it", unplotted_tables(run_evidence(_table_run(by_title))) == [], "")
    by_axes = "Figure 'Fig 1 — purification': x = step [0 to 5], y = cum LRV [0 to 6]"
    _check("its axis labels count too", unplotted_tables(run_evidence(_table_run(by_axes))) == [], "")
    other = "Figure 'Fig 1 — ELISA calibration': x = concentration (ng/mL) [0 to 100], y = absorbance [0 to 2]"
    _check("an unrelated figure doesn't", len(unplotted_tables(run_evidence(_table_run(other)))) == 1, "")
    _check("literature output is not a computed table",
           unplotted_tables(run_evidence(_table_run(literature=True))) == [], "")


def test_figure_refs() -> None:
    print("\n[10] figures the answer cites")
    from biomentis.run_quality import RunEvidence, unmatched_figure_refs

    ev = RunEvidence(figure_titles=["Figure 'Fig 1A  calibration': x = c", "Figure 'Fig 1B accuracy': x = c"])
    _check("drawn figures match", unmatched_figure_refs("See Figure 1A and Fig. 1B.", ev) == [], "")
    _check("an undrawn one is reported", unmatched_figure_refs("See Figure 1 and Figure 3.", ev) == [3], "")
    ev.figure_titles.append("Figure 'axes 1': x = x [0 to 100], y = y [0 to 100]")
    _check("an untitled figure may be it", unmatched_figure_refs("See Figure 1 and Figure 2.", ev) == [], "")
    _check("references in code are ignored",
           unmatched_figure_refs("```\nplt.title('Figure 9')\n```\nDone.", RunEvidence()) == [], "")


def test_gate_asks_for_named_figures() -> None:
    print("\n[11] the check names unplotted tables and missing figures, once")
    from biomentis.run_quality import FIGURE_REFS_MARK, UNPLOTTED_MARK

    table_code = "print(" + repr(_TABLE_STEP) + ")"
    llm = _ScriptedLLM([
        "<execute>def query_pubmed(q):\n    return 'Title: HCP paper'\nprint(query_pubmed('HCP'))</execute>",
        f"<execute>{table_code}</execute>",
        "<solution>Plan. See Figure 1 and Figure 2.</solution>",
        "<execute>print('Plot captured.')\nprint(\"Figure 'Fig 1 — clearance': x = step, y = HCP ppm\")</execute>",
        "<solution>Plan with Figure 1. See Figure 2.</solution>",
    ])
    _, final = _agent(llm, evidence_gate=True).go("Plan HCP control.")
    asked = llm.requests[3][-1].content if len(llm.requests) > 3 else ""
    _check("the first answer went back", UNPLOTTED_MARK in asked and FIGURE_REFS_MARK in asked, asked[:300])
    _check("naming the table", "ITEM 2 — WORKED DSP CLEARANCE TRAIN" in asked, asked)
    _check("and the missing figures", "Figures 1, 2." in asked, asked)
    _check("each asked once: the second answer is accepted",
           len(llm.requests) == 5 and "Plan with Figure 1" in final, f"{len(llm.requests)} {final[-200:]}")

    llm = _ScriptedLLM(["<solution>42</solution>"])
    _, final = _agent(llm, evidence_gate=False).go("Plan HCP control.")
    _check("check off: nothing asked", len(llm.requests) == 1, str(len(llm.requests)))


def test_summary_figure_lines() -> None:
    print("\n[12] the summary: figures lost, tables not plotted")
    from biomentis.run_quality import format_quality_summary, run_evidence
    from biomentis.ui_core import stream_agent_events

    ev = run_evidence(_table_run())
    text = format_quality_summary(ev, unsaved_figure_steps=5)
    _check("figures lost", "5 steps drew a figure that was not saved" in text, text)
    _check("tables not plotted", "**Not plotted:** 1 computed table without a figure: ITEM 2" in text, text)
    _check("nothing lost, nothing said", "Figures lost" not in format_quality_summary(ev), "")

    events = list(stream_agent_events(_scripted_agent([
        AIMessage(content="<execute>plt.show()</execute>"),
        AIMessage(content="<observation>Plot captured.\nFigure 'Fig 1 — x': x = a, y = b</observation>"),
        AIMessage(content="<solution>Done</solution>"),
    ]), "Plot it.", [], [], thread_id=1))
    summary = [e.content for e in events if e.type == "status" and e.channel == "main"]
    _check("a figure with no saved file is reported",
           bool(summary) and "1 step drew a figure that was not saved" in summary[0], str(summary))


def test_figure_summary_and_guidance() -> None:
    print("\n[13] the figure summary and the plot guidance")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from biomentis.plots import describe_figure, plot_guidance

    fig, ax = plt.subplots()
    ax.text(0.2, 0.8, "Inventory")
    ax.text(0.2, 0.5, "G1 risk\nscoring")
    lines = describe_figure(fig)
    plt.close(fig)
    _check("an untitled figure is flagged", any("WARNING: no title" in ln for ln in lines), str(lines))
    _check("a diagram's labels are listed", any("text labels: Inventory; G1 risk scoring" in ln for ln in lines), str(lines))
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    ax.set_title("Fig 1 — trend")
    lines = describe_figure(fig)
    plt.close(fig)
    _check("a titled figure isn't", not any("no title" in ln for ln in lines), str(lines))
    g = plot_guidance("show")
    _check("plans and workflows get figures too", "plan, workflow, protocol or control strategy" in g, g)
    _check("chart picked from the data's shape", "control chart" in g and "log y-axis" in g and "Gantt" in g, g)
    _check("figures numbered in their titles", "'Fig 2 — HCP clearance by step'" in g, g)


def run_all() -> int:
    print("=" * 60)
    print("Run quality tests")
    print("=" * 60)
    for test in (
        test_run_evidence,
        test_missing_evidence_and_nudge,
        test_summary_line,
        test_gate_sends_answer_back,
        test_gate_accepts_a_run_with_evidence,
        test_critic_checks_numbers,
        test_ui_shows_summary,
        test_printed_tables,
        test_plotted_matching,
        test_figure_refs,
        test_gate_asks_for_named_figures,
        test_summary_figure_lines,
        test_figure_summary_and_guidance,
    ):
        try:
            test()
        except Exception as e:
            _check(f"{test.__name__} ran", False, repr(e))

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 0 if not _FAILED else 1


if __name__ == "__main__":
    sys.exit(run_all())
