"""A run's figures: kept as drawn, and matched to the figures its answer cites.

Two problems this solves, both seen in the HCP runs of 2026-10-29:

1. The agent saves figures as loose files in the app's working directory,
   shared by every run, and redraws, renames and deletes them as it goes. The
   transcript only pointed at those paths, so a figure shown at step 12 could
   later show a different drawing, or "Image not found" once the agent had
   cleaned up. `snapshot_image` copies each image into the run's own folder
   the moment a step reports it.
2. The answer cites "Fig 2" or `fig2_risk_tiers.png`, but no figure reached
   the answer panel. `cited_figures` finds, for each citation, the latest
   figure the run drew with that number in its title or file name.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import dataclass, field

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp")

# "Fig 2", "Figure 3", "Figs. 2, 3 and 5", "Figures 4–6", "Fig 5a".
_CITATION_RE = re.compile(
    r"\bFig(?:ure)?s?\.?\s*(\d+[a-z]?(?:\s*(?:,|and|&|or|–|—|-|to)\s*\d+[a-z]?)*)",
    re.IGNORECASE,
)
_TITLE_NUMBER_RE = re.compile(r"^\s*Fig(?:ure)?\.?\s*(\d+)(?!\d)", re.IGNORECASE)
_NAME_NUMBER_RE = re.compile(r"^fig(?:ure)?[_\- ]?0*(\d+)(?!\d)", re.IGNORECASE)
_FILE_NAME_RE = re.compile(r"[\w\-.]+\.(?:png|jpe?g|gif|webp|svg)\b", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_MAX_RANGE = 20


@dataclass
class RunFigure:
    """One figure a step drew: where it is now, and what identifies it."""

    path: str
    titles: list[str] = field(default_factory=list)
    name: str = ""  # the file name the agent gave it, if any

    def numbers(self) -> set[int]:
        """Figure numbers from its titles, or else from its file name."""
        found = {int(m.group(1)) for t in self.titles if (m := _TITLE_NUMBER_RE.match(t))}
        if not found and self.name and (m := _NAME_NUMBER_RE.match(self.name)):
            found.add(int(m.group(1)))
        return found

    def caption(self) -> str:
        parts = [" · ".join(dict.fromkeys(self.titles))] if self.titles else []
        if self.name:
            parts.append(self.name)
        return " — ".join(parts)


def snapshot_image(path: str, out_dir: str, index: int) -> str:
    """Copy `path` into `out_dir` as `image-<NNN>-<name>`, returning the copy's path."""
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.abspath(os.path.join(out_dir, f"image-{index:03d}-{os.path.basename(path)}"))
    shutil.copyfile(path, target)
    return target


def file_digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha1(handle.read()).hexdigest()


def cited_numbers(text: str) -> list[int]:
    """Figure numbers the text cites, in order of first mention.

    Code blocks are ignored, and so are file names: "fig2_risk.png" is a
    citation of that file (see `cited_names`), not of "Fig 2".
    """
    numbers: list[int] = []
    for match in _CITATION_RE.finditer(_FILE_NAME_RE.sub(" ", _CODE_BLOCK_RE.sub("", text or ""))):
        group = match.group(1)
        values = [int(v) for v in re.findall(r"\d+", group)]
        # "4–6" and "4 to 6" are ranges; "4, 6" and "4 and 6" are lists.
        if len(values) == 2 and re.search(r"\d\s*(?:–|—|-|to)\s*\d", group) and 0 < values[1] - values[0] <= _MAX_RANGE:
            values = list(range(values[0], values[1] + 1))
        numbers += [v for v in values if v not in numbers]
    return numbers


def cited_names(text: str) -> list[str]:
    """Image file names the text mentions, in order, once each."""
    names: list[str] = []
    for name in _FILE_NAME_RE.findall(_CODE_BLOCK_RE.sub("", text or "")):
        if name not in names:
            names.append(name)
    return names


def cited_figures(answer: str, figures: list[RunFigure]) -> tuple[list[tuple[str, RunFigure]], list[str]]:
    """The figures the answer cites, as (label, figure), and the citations no figure matches.

    A number matches a figure titled "Fig N…" (or, failing a title, a file
    named fig<N>…); a file name matches a figure saved under that name. The
    latest such figure wins: the agent often redraws a figure several times,
    and the answer describes the last version. Numbered figures come first,
    in number order, then figures cited only by file name.
    """
    by_number: dict[int, RunFigure] = {}
    by_name: dict[str, RunFigure] = {}
    for figure in figures:
        for number in figure.numbers():
            by_number[number] = figure
        if figure.name:
            by_name[figure.name] = figure

    chosen: list[tuple[str, RunFigure]] = []
    missing: list[str] = []
    shown: set[str] = set()
    for number in sorted(cited_numbers(answer)):
        figure = by_number.get(number)
        if figure is None:
            missing.append(f"Fig {number}")
        elif figure.path not in shown:
            chosen.append((f"Fig {number}", figure))
            shown.add(figure.path)
    for name in cited_names(answer):
        figure = by_name.get(name)
        if figure is None:
            missing.append(name)
        elif figure.path not in shown:
            chosen.append((name, figure))
            shown.add(figure.path)
    return chosen, missing


def is_image(path: str) -> bool:
    return path.lower().endswith(_IMAGE_EXTENSIONS)
