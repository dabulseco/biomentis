"""Plot display setting shared by the agent, the tutor and the UI.

"show" displays plots the agent or tutor makes; "student" withholds them so
students run the code themselves. Code is always shown.
"""

from __future__ import annotations

import os

PLOT_MODES = ("show", "student")
PLOT_MODE_LABELS = {"show": "Show plots", "student": "Students run the code"}


def plot_mode_default() -> str:
    """`BIOMENTIS_PLOT_MODE`, stripped and lowercased; unknown or unset is "show"."""
    value = (os.environ.get("BIOMENTIS_PLOT_MODE") or "").strip().lower()
    return value if value in PLOT_MODES else "show"
