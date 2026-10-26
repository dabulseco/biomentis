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


def plot_guidance(plot_mode: str) -> str:
    """System-prompt text telling the agent how to plot and show molecules.

    Every plotting block must stand on its own so a student can paste it alone
    into Colab: the agent's REPL keeps state between steps, Colab does not.
    """
    lines = [
        "PLOTS AND STRUCTURES:",
        "- When explaining a quantitative or visual principle, plot the data with matplotlib. "
        "Label both axes, with units, give the figure a title, and call plt.show().",
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
