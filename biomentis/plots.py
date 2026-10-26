"""Plot display setting shared by the agent, the tutor and the UI.

"show" displays plots the agent or tutor makes; "student" withholds them so
students run the code themselves. Code is always shown.
"""

from __future__ import annotations

import base64
import binascii
import os
import signal
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass

PLOT_MODES = ("show", "student")
PLOT_MODE_LABELS = {"show": "Show plots", "student": "Students run the code"}


def plot_mode_default() -> str:
    """`BIOMENTIS_PLOT_MODE`, stripped and lowercased; unknown or unset is "show"."""
    value = (os.environ.get("BIOMENTIS_PLOT_MODE") or "").strip().lower()
    return value if value in PLOT_MODES else "show"


_PNG_URI_PREFIX = "data:image/png;base64,"


def save_plot_pngs(data_uris: list[str], out_dir: str, start_index: int) -> list[str]:
    """Write each `data:image/png;base64,…` URI to `out_dir/plot-<NNN>.png`.

    Numbering starts at `start_index` and is zero-padded to three digits. A URI
    that is not a decodable PNG data URI is skipped without using a number.
    Returns the absolute paths written, in order.
    """
    paths: list[str] = []
    if not data_uris:
        return paths
    os.makedirs(out_dir, exist_ok=True)
    index = start_index
    for uri in data_uris:
        if not isinstance(uri, str) or not uri.startswith(_PNG_URI_PREFIX):
            continue
        try:
            data = base64.b64decode(uri[len(_PNG_URI_PREFIX) :], validate=True)
        except (binascii.Error, ValueError):
            continue
        path = os.path.abspath(os.path.join(out_dir, f"plot-{index:03d}.png"))
        with open(path, "wb") as handle:
            handle.write(data)
        paths.append(path)
        index += 1
    return paths


SELF_CONTAINED_PLOT_RULE = (
    "write the data inline, import only numpy and matplotlib, and use no variables from "
    "earlier steps, no files, and no biomentis tools, so a student can paste the block "
    "alone into Colab and run it."
)


@dataclass
class PlotResult:
    png_path: str | None = None
    error: str | None = None


_RUNNER = """
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
code, path = sys.argv[1], sys.argv[2]
exec(compile(code, "<plot>", "exec"), {"__name__": "__main__"})
nums = plt.get_fignums()
if nums:
    plt.figure(nums[-1]).savefig(path, dpi=120, bbox_inches="tight")
"""

# What the plot child may see of the parent's environment. Everything else,
# API keys above all, stays out of reach of model-written code.
_ENV_PASSTHROUGH = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError, OSError):
        proc.kill()


def run_plot_code(code: str, out_dir: str, timeout: float = 20.0) -> PlotResult:
    """Run matplotlib `code` in a subprocess and save the last open figure.

    A fresh interpreter in a fresh temp dir shares nothing with the agent's
    REPL, so tutor-written code cannot see or disturb the agent's variables,
    nor read the parent's environment. The parent picks the PNG path and
    accepts only that exact file, whatever the child prints.
    """
    out_dir = os.path.abspath(out_dir)
    png = os.path.join(out_dir, f"chat-plot-{uuid.uuid4().hex[:8]}.png")
    try:
        os.makedirs(out_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="plotrun-") as cwd:
            env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
            env.update(MPLBACKEND="Agg", MPLCONFIGDIR=cwd)
            proc = subprocess.Popen(
                [sys.executable, "-I", "-c", _RUNNER, code, png],
                cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, start_new_session=True,
            )
            try:
                _, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                proc.communicate()
                return PlotResult(error=f"Plot code took longer than {timeout:g} s and was stopped.")
            finally:
                _kill_group(proc)  # no grandchild outlives the run
    except Exception as exc:
        return PlotResult(error=f"{type(exc).__name__}: {exc}")
    if proc.returncode != 0:
        lines = [ln for ln in (stderr or "").strip().splitlines() if ln.strip()]
        return PlotResult(error=lines[-1] if lines else f"Plot code exited with status {proc.returncode}.")
    if os.path.isfile(png):
        return PlotResult(png_path=png)
    return PlotResult(error="The code ran but drew no figure.")


# The agent never sees its figures, only the observation text. Without a
# summary a curve stuck at zero (a units bug) or one that never reaches the
# target looks exactly like a good plot: "Plot captured." Seen with
# gemma4:26b-mlx (123 steps over zero curves) and gemma4:31b-mlx (picked a
# material whose curve never reaches 530 nm), 2026-10-26.
_SUMMARY_MAX_LINES = 24
_SUMMARY_MAX_SERIES = 8


def _num(value: float) -> str:
    return f"{value:.4g}"


def _finite_xy(x, y):
    import numpy as np

    try:
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
    except (TypeError, ValueError):
        return None, None, 0
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok], int(n - ok.sum())


def _crossing(x, y, level: float) -> float | None:
    """First x where the polyline (x, y) reaches `level`, linearly interpolated."""
    for i in range(len(y) - 1):
        d0, d1 = y[i] - level, y[i + 1] - level
        if d0 == 0:
            return float(x[i])
        if d0 * d1 < 0:
            return float(x[i] + (x[i + 1] - x[i]) * d0 / (d0 - d1))
    if len(y) and y[-1] == level:
        return float(x[-1])
    return None


def _series_warning(y, bad: int) -> str:
    import numpy as np

    notes = []
    if bad:
        notes.append(f"{bad} NaN/inf values dropped")
    if len(y) and float(np.max(np.abs(y))) < 1e-9:
        notes.append("WARNING: every value is ~0, check the formula and units")
    elif len(y) > 2 and float(np.ptp(y)) == 0:
        notes.append(f"WARNING: constant at {_num(float(y[0]))}")
    return ("; " + "; ".join(notes)) if notes else ""


def _series_line(label: str, x, y, bad: int, hlines, vlines) -> str:
    import numpy as np

    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    text = (
        f"  {label}: {len(xs)} points, x {_num(xs[0])} to {_num(xs[-1])}, "
        f"y {_num(float(np.min(ys)))} to {_num(float(np.max(ys)))} "
        f"(y={_num(ys[0])} at x={_num(xs[0])}, y={_num(ys[-1])} at x={_num(xs[-1])})"
    )
    text += _series_warning(ys, bad)
    for _, level in hlines:
        hit = _crossing(xs, ys, level)
        text += f"; reaches y={_num(level)} at x≈{_num(hit)}" if hit is not None else f"; NEVER reaches y={_num(level)}"
    for _, at in vlines:
        if xs[0] <= at <= xs[-1]:
            text += f"; at x={_num(at)}, y≈{_num(float(np.interp(at, xs, ys)))}"
        else:
            text += f"; does not extend to x={_num(at)}"
    return text


def describe_figure(fig) -> list[str]:
    """Plain-text summary of a matplotlib figure, for the agent's observation.

    Per axes: title, axis labels and limits. Per curve: its range, and where
    it meets each horizontal/vertical reference line (axhline/axvline), or
    that it never does. Flags curves that are all ~0, constant, or NaN/inf.
    """
    import numpy as np
    from matplotlib.collections import PathCollection

    lines: list[str] = []
    suptitle = fig._suptitle.get_text() if getattr(fig, "_suptitle", None) else ""
    for a_i, ax in enumerate(fig.get_axes(), 1):
        if not ax.get_visible() or getattr(ax, "_colorbar", None) is not None:
            continue  # hidden axes and colorbars
        name = ax.get_title() or suptitle or f"axes {a_i}"
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        lines.append(
            f"Figure '{name}': x = {ax.get_xlabel() or 'x'} [{_num(x0)} to {_num(x1)}], "
            f"y = {ax.get_ylabel() or 'y'} [{_num(y0)} to {_num(y1)}]"
        )
        hlines, vlines, series = [], [], []
        for k, line in enumerate(ax.get_lines(), 1):
            x, y, bad = _finite_xy(line.get_xdata(), line.get_ydata())
            if x is None or not len(x):
                continue
            label = line.get_label()
            label = f"line {k}" if not label or label.startswith("_") else label
            # axhline/axvline (and any two-point flat line) are reference lines;
            # a long flat curve stays a series so its warning shows.
            if len(y) == 2 and y[0] == y[1] and x[0] != x[1]:
                hlines.append((label, float(y[0])))
            elif len(x) == 2 and x[0] == x[1] and y[0] != y[1]:
                vlines.append((label, float(x[0])))
            else:
                series.append((label, x, y, bad))
        for coll in ax.collections:
            if not isinstance(coll, PathCollection):
                continue  # only scatter points are data series
            offsets = np.asarray(coll.get_offsets())
            if offsets.ndim != 2 or not len(offsets):
                continue
            x, y, bad = _finite_xy(offsets[:, 0], offsets[:, 1])
            if x is None or not len(x):
                continue
            label = coll.get_label()
            label = "points" if not label or label.startswith("_") else label
            series.append((f"{label} (scatter)", x, y, bad))
        for container in ax.containers:
            patches = getattr(container, "patches", None) or []
            if not patches:
                continue
            ticks = [t.get_text() for t in ax.get_xticklabels()]
            heights = [p.get_height() for p in patches]
            if len(ticks) == len(heights) and all(ticks):
                pairs = ", ".join(f"{t}={_num(h)}" for t, h in zip(ticks, heights))
            else:
                pairs = ", ".join(_num(h) for h in heights)
            label = container.get_label()
            label = f" {label}" if label and not str(label).startswith("_") else ""
            lines.append(f"  bars{label}: {pairs}"[:400])
        for image in ax.images:
            arr = np.asarray(image.get_array(), dtype=float)
            if arr.size:
                lines.append(f"  image {arr.shape}, values {_num(np.nanmin(arr))} to {_num(np.nanmax(arr))}")
        for label, x, y, bad in series[:_SUMMARY_MAX_SERIES]:
            lines.append(_series_line(label, x, y, bad, hlines, vlines))
        if len(series) > _SUMMARY_MAX_SERIES:
            lines.append(f"  … {len(series) - _SUMMARY_MAX_SERIES} more series")
        if not series and hlines and not ax.containers and not ax.images:
            lines.append("  reference lines only: " + ", ".join(f"y={_num(v)}" for _, v in hlines))
        if not (series or hlines or vlines or ax.containers or ax.images or ax.patches):
            lines.append("  WARNING: these axes are empty, nothing was drawn")
    if len(lines) > _SUMMARY_MAX_LINES:
        lines = lines[:_SUMMARY_MAX_LINES] + ["  … summary truncated"]
    return lines


def plot_guidance(plot_mode: str) -> str:
    """System-prompt text telling the agent how to plot and show molecules.

    Every plotting block must stand on its own so a student can paste it alone
    into Colab: the agent's REPL keeps state between steps, Colab does not.
    """
    lines = [
        "PLOTS AND STRUCTURES:",
        "- Whenever the task compares quantities or asks how one quantity depends on another "
        "(a trend, a curve, a size or dose dependence, a spectrum), make at least one matplotlib "
        "plot before you give your <solution>, even if the user did not ask for one. "
        "Label both axes, with units, give the figure a title, add a legend, and call plt.show().",
        "- Mark any target value the task names (e.g. a wavelength) with plt.axhline or "
        "plt.axvline, so you can see which curves reach it.",
        "- You cannot see the figure. After plt.show() the observation prints a text summary of it: "
        "axis ranges, each curve's range, where each curve meets the reference lines (or NEVER "
        "reaches them), and warnings such as values that are all ~0. Check that summary makes "
        "physical sense before drawing conclusions. If it doesn't, fix the code; never rerun the "
        "same code expecting a different result.",
        "- Every plotting block is self-contained: " + SELF_CONTAINED_PLOT_RULE,
        "- Finish each figure within one step (one <execute> block); figures still open when a "
        "step ends are captured and closed.",
        "- To display a molecule, call show_structure(source) with a PDB ID (e.g. 6NK5), a path "
        "to a .pdb/.sdf/.smi file, or a SMILES string: "
        "`from biomentis.tool.support_tools import show_structure`.",
    ]
    if plot_mode == "student":
        lines.append(
            "- Students will run the plotting code themselves and will not see the figure in the "
            "app, so the code must be complete and commented, explaining what each part draws."
        )
    else:
        lines.append("- The figure will be shown beside the code in the app.")
    return "\n".join(lines)
