"""Plot display setting shared by the agent, the tutor and the UI.

"show" displays plots the agent or tutor makes; "student" withholds them so
students run the code themselves. Code is always shown.
"""

from __future__ import annotations

import base64
import binascii
import os

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


def plot_guidance(plot_mode: str) -> str:
    """System-prompt text telling the agent how to plot and show molecules.

    Every plotting block must stand on its own so a student can paste it alone
    into Colab: the agent's REPL keeps state between steps, Colab does not.
    """
    lines = [
        "PLOTS AND STRUCTURES:",
        "- When explaining a quantitative or visual principle, plot the data with matplotlib. "
        "Label both axes, with units, give the figure a title, and call plt.show().",
        "- Every plotting block is self-contained: write the data inline, import only numpy and "
        "matplotlib, and use no variables from earlier steps, no files, and no biomentis tools, "
        "so a student can paste that block alone into Colab and run it.",
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
