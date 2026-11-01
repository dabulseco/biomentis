"""Claim checks: "verified" with no source, simulated data not labelled, missing evidence."""

from __future__ import annotations

import re

from biomentis.gate import Finding
from biomentis.run_quality import EVIDENCE_ITEMS, RunEvidence, missing_evidence, unplotted_tables

_VERIFIED_RE = re.compile(r"(?i)\b(verified|confirmed)\b")
# A dotted number; whether it is a version is decided in `_versions`.
_DOTTED_RE = re.compile(r"(?<![\w.])(v)?(\d+(?:\.\d+)+)(?!\d)", re.IGNORECASE)
# Words that make a one-dot number a version when they come at most 3 words before it.
_VERSION_WORDS = {"v", "version", "release"}
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

_RANDOM_RE = re.compile(r"np\.random|\brandom\.|default_rng|\brng\.")
_SAMPLE_HEADING_RE = re.compile(r"(?i)sample data|example data|sample .*table")
_LABEL_RE = re.compile(r"(?i)simulated|illustrative|synthetic|hypothetical|invented|not real")
_LABEL_WINDOW = 300
_MAX_CAPTION_CHARS = 200


def _versions(sentence: str) -> list[str]:
    """The version numbers in `sentence`, without a leading "v".

    A decimal is not a version: "pH 7.4" and "21.25 g" are not. A number is
    one when it has two or more dots (1.2.12), is written v1.2, or has
    "v", "version" or "release" at most 3 words before it.
    """
    found = []
    for m in _DOTTED_RE.finditer(sentence):
        number = m.group(2)
        before = [w.strip("()[],:;\"'").lower() for w in sentence[:m.start()].split()[-3:]]
        if m.group(1) or number.count(".") >= 2 or _VERSION_WORDS & set(before):
            found.append(number)
    return found


def _shown(version: str, run_text: str) -> bool:
    """Whether `version` appears in the run's output as a whole number: 1.2 is not in 21.25 or 1.2.12."""
    return re.search(r"(?<![\d.])" + re.escape(version) + r"(?!\.?\d)", run_text) is not None


def check_verified_claims(answer: str, run_text: str) -> list[Finding]:
    findings: list[Finding] = []
    for sentence in _SENTENCE_SPLIT_RE.split(answer):
        sentence = sentence.strip()
        if not sentence or not _VERIFIED_RE.search(sentence):
            continue
        versions = _versions(sentence)
        if not versions or _URL_RE.search(sentence):
            continue
        if any(_shown(v, run_text) for v in versions):
            continue
        findings.append(Finding(
            check="claim",
            severity="blocking",
            quote=sentence,
            issue=f"Claims a version ({', '.join(versions)}) is verified, but no URL is given and "
                  "no tool output in the run shows it.",
            fix="Look the version up in code and cite the page, or remove the word 'verified' and "
                "say it comes from memory.",
        ))
    return findings


def check_simulated_labels(answer: str, run_code: list[str]) -> list[Finding]:
    if not any(_RANDOM_RE.search(code) for code in run_code):
        return []
    findings: list[Finding] = []
    offset = 0
    for line in answer.splitlines(keepends=True):
        start, offset = offset, offset + len(line)
        text = line.strip()
        if not text or len(text) > _MAX_CAPTION_CHARS or not _SAMPLE_HEADING_RE.search(text):
            continue
        if _LABEL_RE.search(text + " " + answer[offset:offset + _LABEL_WINDOW]):
            continue
        findings.append(Finding(
            check="claim",
            severity="blocking",
            quote=text,
            issue="The code generated this data with a random generator, but the heading does not "
                  "say it is simulated, so it reads as real measurements.",
            fix="Label it as simulated or illustrative data in the heading or caption.",
        ))
    return findings


def check_evidence(ev: RunEvidence) -> list[Finding]:
    """A missing literature source or calculation blocks; a missing figure only when a computed table needs one.

    Figures are asked for "where they help": with no computed table left
    unplotted, a run without a figure gets a warning, not a block.
    """
    unplotted = unplotted_tables(ev)
    findings = [
        Finding(
            check="evidence",
            severity="warning" if key == "figure" and not unplotted else "blocking",
            quote="",
            issue=f"The run has {EVIDENCE_ITEMS[key][0]}.",
            fix=f"Add {EVIDENCE_ITEMS[key][1]}.",
            key=f"evidence:{key}",
        )
        for key in missing_evidence(ev)
    ]
    for heading, _ in unplotted:
        findings.append(Finding(
            check="evidence",
            severity="warning",
            quote=heading,
            issue="A computed table has no figure.",
            fix="Plot it if it shows a trend, comparison or distribution.",
        ))
    return findings
