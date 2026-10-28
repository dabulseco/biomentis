"""Tests for what the agent learns about its own plots, and the repeat guard.

The agent never sees a figure, only the observation text. Before these
changes that text was "Plot captured.", so a curve stuck at zero (a units bug)
looked like a good plot. gemma4:26b-mlx then re-ran the same block 108 times
in one 32-minute run, and gemma4:31b-mlx picked a material whose curve never
reached the target wavelength (2026-10-26 model comparison).

What these tests pin down:

  1.  describe_figure reports axis labels, ranges, and where each curve meets
      a horizontal reference line, or that it NEVER does
  2.  ...flags curves that are all ~0, constant, or contain NaN/inf
  3.  ...summarizes scatter, bars, images and vertical reference lines, skips
      colorbars, and flags empty axes; single points are listed as marked
      points, not as curves
  4.  The observation carries the summary after plt.show() and for figures
      left open
  5.  The plotting guidance asks for a plot on comparisons, reference lines,
      and checking the summary
  6.  A step that repeats an earlier one (same code, same output) is pointed
      out; the 3rd asks for the answer; the 4th ends the run
  7.  Same code with a different output is not a repeat
  8.  With self-critic on, the stopped run does not go back to the critic
  9.  Each run starts with no steps seen (go, and stream_agent_events)
 10.  A step that raises keeps what it printed, and says which error and which
      line ("Error: GUARD FAILED" alone hid an agent's own self-check
      results, 2026-10-28); a figure still open when it raises is captured

Run with:
    python -m biomentis.agent.tests.test_plot_feedback
"""

from __future__ import annotations

import sys

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


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# --- 1-3: describe_figure -------------------------------------------------


def _quantum_dot_figure():
    import numpy as np

    plt = _plt()
    d = np.linspace(2, 10, 50)
    fig, ax = plt.subplots()
    ax.plot(d, 1240 / (2.26 + 2.5 / d**2), label="ZnTe")
    ax.plot(d, 1240 / (2.7 + 2.0 / d**2), label="ZnSe")
    ax.axhline(530, linestyle="--", label="target")
    ax.set_xlabel("diameter (nm)")
    ax.set_ylabel("emission (nm)")
    ax.set_title("QD emission")
    return fig


def test_crossings() -> None:
    print("\n[1] ranges and reference-line crossings")
    from biomentis.plots import describe_figure

    plt = _plt()
    fig = _quantum_dot_figure()
    text = "\n".join(describe_figure(fig))
    plt.close(fig)
    _check("title and axis labels", "QD emission" in text and "diameter (nm)" in text and "emission (nm)" in text, text)
    znte = next((ln for ln in text.splitlines() if ln.strip().startswith("ZnTe")), "")
    znse = next((ln for ln in text.splitlines() if ln.strip().startswith("ZnSe")), "")
    _check("ZnTe reaches 530 near 5.6 nm", "reaches y=530 at x≈5.6" in znte, znte)
    _check("ZnSe NEVER reaches 530", "NEVER reaches y=530" in znse, znse)
    _check("the curve's range is given", "y 387.5 to 455.9" in znse, znse)
    _check("the reference line is not listed as a curve", not any(ln.strip().startswith("target") for ln in text.splitlines()), text)


def test_warnings() -> None:
    print("\n[2] zero, constant and NaN curves are flagged")
    import numpy as np

    from biomentis.plots import describe_figure

    plt = _plt()
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 20)
    ax.plot(x, x * 1e-17, label="tiny")
    ax.plot(x, np.full_like(x, 3.0), label="flat")
    ax.plot(x, np.where(x > 0.5, np.nan, x), label="gappy")
    lines = describe_figure(fig)
    plt.close(fig)
    by = {ln.strip().split(":")[0]: ln for ln in lines}
    _check("all-zero curve warned", "every value is ~0" in by.get("tiny", ""), str(lines))
    _check("constant curve warned", "constant at 3" in by.get("flat", ""), str(lines))
    _check("NaN values counted", "10 NaN/inf values dropped" in by.get("gappy", ""), str(lines))


def test_other_marks() -> None:
    print("\n[3] scatter, bars, images, vertical lines, colorbars, empty axes")
    from biomentis.plots import describe_figure

    plt = _plt()
    fig, (a1, a2, a3) = plt.subplots(1, 3)
    a1.scatter([1, 2, 3], [4, 5, 6], label="obs")
    a1.plot([1, 2, 3], [1, 2, 3], label="fit")
    a1.axvline(2.5)
    a2.bar(["InP", "ZnTe"], [104.8, 3.99])
    im = a3.imshow([[0, 1], [2, 3]])
    fig.colorbar(im)
    lines = describe_figure(fig)
    plt.close(fig)
    text = "\n".join(lines)
    _check("scatter summarized", "obs (scatter): 3 points" in text, text)
    _check("value at the vertical line", "at x=2.5, y≈2.5" in text, text)
    _check("bars with their labels", "bars: InP=104.8, ZnTe=3.99" in text, text)
    _check("image range", "values 0 to 3" in text, text)
    _check("colorbar axes skipped", text.count("Figure '") == 3, text)

    fig, ax = plt.subplots()
    text = "\n".join(describe_figure(fig))
    plt.close(fig)
    _check("empty axes warned", "axes are empty" in text, text)

    fig, ax = plt.subplots()
    ax.plot([2, 4, 6], [500, 530, 545], label="ZnTe")
    ax.plot([4.0], [530.0], "o")
    ax.scatter([3.5], [530], label="Selected")
    ax.axhline(530)
    lines = describe_figure(fig)
    plt.close(fig)
    text = "\n".join(lines)
    _check("single points listed as marked points", "marked points: line 2 (4, 530), Selected (3.5, 530)" in text, text)
    _check("and not as curves", not any("1 points" in ln for ln in lines), text)


# --- 4: the observation -----------------------------------------------------


def test_observation_carries_summary() -> None:
    print("\n[4] the observation includes the summary")
    from biomentis.tool.support_tools import clear_captured_plots, run_python_repl

    shown = run_python_repl(
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3], [0, 0, 0], label='zero')\n"
        "plt.axhline(1)\nplt.title('shown')\nplt.show()\nprint('after')"
    )
    _check("after plt.show()", "Plot captured." in shown and "Figure 'shown'" in shown, shown)
    _check("the zero curve is flagged to the agent", "every value is ~0" in shown and "NEVER reaches y=1" in shown, shown)
    _check("the code's own output is kept", "after" in shown, shown)

    left_open = run_python_repl(
        "import matplotlib.pyplot as plt\nplt.plot([1, 2], [3, 4], label='open')\nplt.title('left open')\nprint('done')"
    )
    _check("for a figure left open", left_open.startswith("done") and "Figure 'left open'" in left_open, left_open)

    failed = run_python_repl("import matplotlib.pyplot as plt\nplt.plot([1, 2], [3, 4])\nraise ValueError('boom')")
    _check("a failing step reports its error and its figure", "Error: boom" in failed and "Figure '" in failed, failed)

    plain = run_python_repl("print('no plot')")
    _check("no figure, no summary", plain == "no plot\n", repr(plain))
    clear_captured_plots()


# --- 5: guidance --------------------------------------------------------------


def test_guidance() -> None:
    print("\n[5] plotting guidance")
    from biomentis.plots import plot_guidance

    for mode in ("show", "student"):
        text = plot_guidance(mode)
        _check(f"{mode}: plot comparisons unasked", "even if the user did not ask" in text, text)
        _check(f"{mode}: mark targets", "axhline" in text, text)
        _check(f"{mode}: read the summary", "text summary" in text and "never rerun the same code" in text, text)


# --- 6-9: the repeat guard ------------------------------------------------------


class _ScriptLLM:
    """Replies from a script; repeats the last reply forever. Raises past
    `limit` calls so an unbounded loop fails the test instead of hanging."""

    def __init__(self, replies: list[str], limit: int = 20):
        self.replies = list(replies)
        self.calls = 0
        self.limit = limit
        self.seen: list = []

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        self.seen.append(messages)
        if self.calls > self.limit:
            raise RuntimeError(f"still looping after {self.limit} calls")
        reply = self.replies[min(self.calls - 1, len(self.replies) - 1)]
        return AIMessage(content=reply)


def _agent(llm, self_critic: bool = False):
    from biomentis.agent.a1 import A1

    agent = A1(path="./data", use_tool_retriever=False, expected_data_lake_files=[], trace=False)
    agent.llm = llm
    agent._creative_llm = llm  # the self-critic's client
    agent.configure(self_critic=self_critic)
    return agent


_SAME = "Try again.\n<execute>print('same')</execute>"


def test_repeats_end_the_run() -> None:
    print("\n[6] repeats are pointed out, then the run ends")
    from biomentis.agent.a1 import REPEAT_STOP_AT

    llm = _ScriptLLM([_SAME])
    agent = _agent(llm)
    log, final = agent.go("loop please")
    _check("stopped after the 4th repeat", llm.calls == REPEAT_STOP_AT + 1, str(llm.calls))
    _check("the run says why it stopped", "same code with the same result 4 times" in final, final)
    observations = [m for m in log if "<observation>" in m]
    _check("first run has no note", observations and "NOTE" not in observations[0], str(observations[:1]))
    _check("a repeat names the step it repeats", "repeated step 1 exactly" in (observations[1] if len(observations) > 1 else ""), str(observations))
    third = observations[3] if len(observations) > 3 else ""
    _check("the 3rd repeat asks for the answer now", "Give your <solution> now" in third, third)
    _check("the 1st repeat does not", "Give your <solution> now" not in (observations[1] if len(observations) > 1 else ""), str(observations))


def test_changed_output_is_not_a_repeat() -> None:
    print("\n[7] same code, different output, is not a repeat")
    step = "<execute>n = globals().get('n', 0) + 1\nprint(n)</execute>"
    llm = _ScriptLLM([step] * 6 + ["<solution>done</solution>"])
    agent = _agent(llm)
    log, final = agent.go("count")
    _check("ran to its solution", "<solution>done</solution>" in final, final)
    _check("no repeat notes", not any("NOTE: this step repeated" in m for m in log), str(log[-3:]))


def test_self_critic_does_not_restart() -> None:
    print("\n[8] the stopped run skips the self-critic")
    llm = _ScriptLLM([_SAME])
    agent = _agent(llm, self_critic=True)
    _, final = agent.go("loop please")
    _check("ended on the stop message", "Run stopped" in final, final)
    _check("no critic call", llm.calls == 5, str(llm.calls))


def test_each_run_starts_fresh() -> None:
    print("\n[9] each run starts with no steps seen")
    llm = _ScriptLLM([_SAME, _SAME, "<solution>first</solution>", _SAME, "<solution>second</solution>"])
    agent = _agent(llm)
    log, final = agent.go("run one")
    _check("first run had one repeat", sum("NOTE: this step repeated" in m for m in log) == 1, str(log))
    log, final = agent.go("run two")
    _check("go(): second run's first step is not a repeat", not any("NOTE: this step repeated" in m for m in log), str(log))

    from biomentis.ui_core import stream_agent_events

    agent._seen_steps = {("print('same')", "same\n"): 1}
    llm2 = _ScriptLLM([_SAME, "<solution>ui</solution>"])
    agent.llm = llm2
    agent.configure()
    events = list(stream_agent_events(agent, "ui run", [], [], thread_id=99))
    obs = [e.content for e in events if e.type == "observation"]
    _check("stream_agent_events: no stale repeat", obs and not any("NOTE" in o for o in obs), str(obs))


def test_error_keeps_output_and_line() -> None:
    print("\n[10] a step that raises keeps its output, the error type and the line")
    import matplotlib

    matplotlib.use("Agg")
    from biomentis.tool.support_tools import run_python_repl

    out = run_python_repl(
        'ok = {"glucose": False}\nprint("peak VCC 21.0 e6/mL")\nprint("titer 4.2 g/L")\nassert all(ok.values())'
    )
    _check("still starts with Error:", out.startswith("Error: AssertionError"), out)
    _check("names the failing line", "at line 4: assert all(ok.values())" in out, out)
    _check("keeps what was printed", "peak VCC 21.0 e6/mL" in out and "titer 4.2 g/L" in out, out)
    _check("the error comes before the output", out.index("Error:") < out.index("peak VCC"), out)

    out = run_python_repl('print("results")\nraise RuntimeError("GUARD FAILED")')
    _check("a message is kept", out.startswith("Error: RuntimeError: GUARD FAILED"), out)

    out = run_python_repl("import numpy as np\nx = 2\nnp.zeros(3).reshape(2, 2)")
    _check("an error inside a library points at the agent's line",
           "at line 3: np.zeros(3).reshape(2, 2)" in out and "ValueError" in out, out)
    _check("no empty output section", "Output printed before" not in out, out)

    out = run_python_repl("def f(:\n    pass")
    _check("a syntax error names its line", out.startswith("Error: SyntaxError") and "at line 1: def f(:" in out, out)

    out = run_python_repl(
        "import matplotlib.pyplot as plt\nplt.plot([1, 2], [3, 4])\nplt.title('VCD')\nraise ValueError('bad fit')"
    )
    _check("an open figure is still captured", "Error: ValueError: bad fit" in out and "Plot captured." in out, out)


def run_all() -> int:
    print("=" * 60)
    print("Plot feedback + repeat guard tests")
    print("=" * 60)
    for test in (
        test_crossings,
        test_warnings,
        test_other_marks,
        test_observation_carries_summary,
        test_guidance,
        test_repeats_end_the_run,
        test_changed_output_is_not_a_repeat,
        test_self_critic_does_not_restart,
        test_each_run_starts_fresh,
        test_error_keeps_output_and_line,
    ):
        try:
            test()
        except Exception as e:
            import traceback

            traceback.print_exc()
            _check(f"{test.__name__} ran", False, repr(e))

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(run_all())
