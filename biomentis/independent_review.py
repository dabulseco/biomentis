"""Independent review of a finished answer by a second model.

The agent's self-critique runs on the agent's own model, so it shares that
model's blind spots: in a 2026-10-28 CHO-workflow run, deepseek-v4.1-flash:cloud
corrected five of its numbers in its critique round and kept a 50x scale-up
error through both drafts. The evidence check (run_quality) only confirms that
sources, calculations and figures exist, not that the answer agrees with them.

This module has a different model (the Critic model picked in the sidebar)
review the run once it has an answer: whether each change the self-critique
made was valid, whether the final answer's numbers, units and claims match the
code output and sources of the run, and whether the reason given for each
major workflow decision holds. A "needs rework" verdict sends the problems back
to the agent for one correction round, then the reviewer checks again; both
reviews are reported, and anything left is the user's call.

With the completion gate on (biomentis.gate), the same review runs at every
gate round instead, and also checks the run's rubric: `coverage_findings` and
`review_findings` turn the review into gate findings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from biomentis.gate import Finding
from biomentis.run_quality import CRITIC_FEEDBACK_PREFIX, GATE_FEEDBACK_PREFIX, REVIEW_FEEDBACK_PREFIX, current_run

VERDICTS = ("accept", "accept_with_corrections", "needs_rework")
VERDICT_LABELS = {
    "accept": "✅ Accept",
    "accept_with_corrections": "🟡 Accept with corrections",
    "needs_rework": "🔴 Needs rework",
}

# The evidence handed to the reviewer: each step's code and output, cut to fit.
_STEP_CODE_CHARS = 4000
_STEP_OUTPUT_CHARS = 2500
_EVIDENCE_TOTAL_CHARS = 90_000

REVIEW_SYSTEM_PROMPT = """You are an independent scientific reviewer. A research agent (a different model) worked on a task by writing and running code, consulting sources, and drafting an answer; its own self-critique then revised the draft. You check the work. You cannot run code: judge from the code outputs and sources shown in EVIDENCE.

A number computed in code is only as good as its inputs. Correct arithmetic on an unrealistic input gives a wrong answer, so "the code computed it" is never enough on its own. For every assumed input that drives a decision (a rate, a concentration, a yield, an efficiency, a limit), ask whether it is realistic for the field, using your own knowledge of typical published values.

Check, in this order:
1. CHANGES: every substantive difference between FIRST_DRAFT and FINAL_ANSWER. For each, is the change valid (the result is right AND its inputs are realistic), invalid (wrong, made worse, or built on an input outside the typical range, even if the arithmetic is right), or unverified (nothing settles it)? A change that replaces a typical value with an atypical one is invalid unless EVIDENCE shows why.
2. PROBLEMS in FINAL_ANSWER:
   - number: a value that disagrees with what the code in EVIDENCE computed, or that is wrong for the domain
   - assumption: an input to a calculation that is outside the range typically reported for this kind of process, so the computed result cannot be trusted
   - unit: a unit error or an inconsistent unit (e.g. rpm used where rev/s is needed, pg vs ng)
   - unsupported: a factual claim, limit or specification that no source in EVIDENCE supports and that is doubtful
   - contradiction: two parts of the answer that disagree
   - rationale: a design decision whose stated reason is wrong, missing, or does not follow
3. RATIONALE: the 3 to 8 most important decisions in FINAL_ANSWER (choices of method, order of steps, criteria, limits). Is the reason given for each sound?
4. ASSUMPTIONS: for each assumption listed under ASSUMPTION_CHECKS, judge the value FINAL_ANSWER uses against the search results shown for it: "supported" (the sources report similar values), "contradicted" (they report clearly different values; give the typical value and the URL), or "unclear" (the results don't settle it). "supported" and "contradicted" must cite a URL from the results; without one, the status is "unclear". Convert units before comparing, using the answer's own numbers where needed: e.g. HCP in ppm is ng HCP per mg product, so at a titer of 5 g/L, 100 mg/L of HCP is 20,000 ppm. Fill "reported" and "converted" first, then decide the status from those two numbers: a value several-fold outside the reported range is "contradicted", however plausible it sounds. Search results are partial: say "unclear" rather than guess, but a value that the sources clearly contradict is a problem of kind "assumption" and, if a decision rests on it, makes the verdict "needs_rework".
5. COVERAGE: for each item listed under RUBRIC, does FINAL_ANSWER deliver it? "present" (quote the exact text from FINAL_ANSWER that delivers it; without a quote it counts as missing), "missing" (nothing in FINAL_ANSWER delivers it), or "wrong" (FINAL_ANSWER addresses it incorrectly; quote that text and say what is wrong). Report every rubric ID. With no RUBRIC section, return an empty list.

Verdict:
- "needs_rework" if any problem would change a decision, a quantity someone would act on, or a safety/quality limit (including a decision that rests on an unrealistic assumption)
- "accept_with_corrections" if the problems are minor and local
- "accept" if you found no problems

Review FINAL_ANSWER, not FIRST_DRAFT: FIRST_DRAFT is there only to judge the changes. Be specific and conservative: every problem must quote the exact text from FINAL_ANSWER it concerns, and say what is wrong and what the correct value or reasoning is. Do not flag style. Do not invent problems to have something to say.

Return ONLY a JSON object, no markdown fences, with exactly these keys:
{
  "verdict": "accept" | "accept_with_corrections" | "needs_rework",
  "summary": "<2-3 sentences: overall judgment>",
  "changes": [{"change": "<what changed>", "status": "valid" | "invalid" | "unverified", "reason": "<why, citing EVIDENCE where possible>"}],
  "problems": [{"kind": "number" | "assumption" | "unit" | "unsupported" | "contradiction" | "rationale", "quote": "<exact text from FINAL_ANSWER>", "issue": "<what is wrong>", "fix": "<the correct value or reasoning>"}],
  "rationale": [{"decision": "<the decision>", "sound": "yes" | "no" | "unverified", "note": "<one sentence>"}],
  "assumptions": [{"assumption": "<as listed>", "value": "<value FINAL_ANSWER uses>", "reported": "<the value or range the sources report, in their units, or none>", "converted": "<the answer's value in the sources' units, with the arithmetic, e.g. 100 mg/L / 6.76 g/L = 14,800 ppm>", "status": "supported" | "contradicted" | "unclear", "evidence": "<what the sources report, with the URL>"}],
  "coverage": [{"id": "<rubric ID, e.g. D1>", "status": "present" | "missing" | "wrong", "quote": "<exact text from FINAL_ANSWER, or empty>", "issue": "<what is missing or wrong, or empty>"}]
}"""

ASSUMPTIONS_SYSTEM_PROMPT = """You are preparing an independent review of a research answer. Before the review, its key assumptions will be checked against a web search.

List the 3 to 5 assumed input values in FINAL_ANSWER that most affect its decisions or numbers: rates, concentrations, yields, efficiencies, removal factors, specification limits. Choose values the answer assumes rather than computes from other values; values tagged [A] (assumption) are good candidates. Prefer the assumptions that a decision rests on.

For each, give two web search queries that would find typical published values for it: plain words, 5 to 12 of them, no quotes, naming the system. The first uses the answer's own units. The second asks the same thing the way the literature usually reports it: another unit (ppm or ng/mg for impurities, g/L, a ratio to the product, a log reduction), or the measured quantity's usual name. Example for an HCP concentration of 100 mg/L: "CHO harvest cell culture fluid host cell protein concentration mg/L" and "host cell protein ppm harvest cell culture fluid antibody".

Return ONLY a JSON object, no markdown fences:
{"assumptions": [{"assumption": "<what is assumed>", "value": "<the value used, with units>", "quote": "<exact text from FINAL_ANSWER>", "queries": ["<query in the answer's units>", "<query as the literature reports it>"]}]}"""

MAX_ASSUMPTIONS = 5
# Two queries per assumption: in one CHO review (2026-10-28) the query in the
# answer's units (mg/L) found no harvest HCP value at all, while one in ppm
# found "more than 50,000 ppm in the CCF" and "993,331 ppm".
MAX_QUERIES = 2
# Results per query, and the most text kept from each search.
_RESULTS_PER_ASSUMPTION = 3
_SEARCH_TEXT_CHARS = 2000


# Sent once when the reviewer's reply could not be read as the JSON asked for.
REVIEW_RETRY_MESSAGE = (
    "Your reply could not be read as the JSON object requested. Reply again with only that JSON object: "
    "no markdown, no commentary, starting with { and ending with }."
)


@dataclass
class Review:
    """One reviewer verdict, or the reason there isn't one (`error`)."""

    verdict: str = ""
    summary: str = ""
    changes: list[dict] = field(default_factory=list)
    problems: list[dict] = field(default_factory=list)
    rationale: list[dict] = field(default_factory=list)
    assumptions: list[dict] = field(default_factory=list)
    coverage: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def needs_rework(self) -> bool:
        return not self.error and self.verdict == "needs_rework"


def _solution_body(text: str) -> str:
    match = re.search(r"<solution>(.*?)</solution>", text or "", re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _clip(text: str, limit: int) -> str:
    """Cut text to about `limit` characters, keeping its head (60%) and its tail (40%)."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return text[:head] + f"\n... [{len(text) - limit} characters omitted] ...\n" + text[len(text) - tail :]


# Focus excerpts (numbers the gate questioned) share at most this part of the
# evidence budget; each excerpt is a window around a number's first match.
_FOCUS_SHARE = 0.5
_FOCUS_RADIUS = 300
# Room for each step's "[step i] CODE/OUTPUT" headers and clip markers.
_STEP_OVERHEAD_CHARS = 120


def _number_re(n: str) -> re.Pattern:
    """`n` as a whole number, not inside a longer one."""
    return re.compile(rf"(?<![\d.]){re.escape(n)}(?!\d)")


def _focus_matches(output: str, numbers) -> list[re.Match]:
    """The first match in `output` of each of `numbers`, in order of position."""
    found = [m for n in sorted(numbers) if n and (m := _number_re(n).search(output))]
    return sorted(found, key=lambda m: m.start())


def _focus_excerpt(output: str, matches: list[re.Match], limit: int) -> str:
    """Windows around `matches` (merged where they overlap), in at most about `limit` characters.

    The radius shrinks so every number keeps a window when the share is small.
    """
    radius = max(20, min(_FOCUS_RADIUS, (limit // max(1, len(matches)) - 10) // 2))
    spans: list[list[int]] = []
    for m in matches:
        start, end = max(0, m.start() - radius), min(len(output), m.end() + radius)
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return "\n...\n".join(output[a:b].strip() for a, b in spans)


def review_inputs(messages: list[BaseMessage], focus_numbers: set[str] = frozenset()) -> dict:
    """The task, first draft, critiques, final answer and evidence of the latest run.

    For a step whose output shows any of `focus_numbers` (numbers the gate's
    checks questioned, as written in the answer) the reviewer gets a window
    around each such number instead of the clipped start, so it can judge
    them. The windows share at most half the evidence budget, and with focus
    numbers the evidence never exceeds `_EVIDENCE_TOTAL_CHARS`: printing a
    questioned number in every step must not overflow the reviewer's context.
    """
    run = current_run(messages)
    task = run[0].content if run and isinstance(run[0].content, str) else ""
    answers = [_solution_body(m.content) for m in run if isinstance(m, AIMessage) and isinstance(m.content, str)]
    answers = [a for a in answers if a]
    # The self-critic's feedback, and with the completion gate the gate's
    # (findings plus the critic's text): either led to a revision.
    critiques = [
        m.content[len(CRITIC_FEEDBACK_PREFIX) :].strip()
        if m.content.startswith(CRITIC_FEEDBACK_PREFIX)
        else m.content.strip()
        for m in run
        if isinstance(m, HumanMessage)
        and isinstance(m.content, str)
        and m.content.startswith((CRITIC_FEEDBACK_PREFIX, GATE_FEEDBACK_PREFIX))
    ]
    steps: list[tuple[str, str]] = []
    code = None
    for m in run:
        content = m.content if isinstance(m.content, str) else ""
        if content.lstrip().startswith("<observation>"):
            if code is not None:
                steps.append((code, content.strip()[len("<observation>") :].removesuffix("</observation>")))
            code = None
            continue
        match = re.search(r"<execute>(.*?)</execute>", content, re.DOTALL | re.IGNORECASE)
        if match and isinstance(m, AIMessage):
            code = match.group(1)
    focus = {i: found for i, (_, o) in enumerate(steps) if (found := _focus_matches(o, focus_numbers or ()))}
    if not focus:
        # Shrink each step's share when a long run would not fit.
        per_step = max(1, len(steps))
        scale = min(1.0, _EVIDENCE_TOTAL_CHARS / (per_step * (_STEP_CODE_CHARS + _STEP_OUTPUT_CHARS)))
        code_cap = max(300, int(_STEP_CODE_CHARS * scale))
        output_cap = max(500, int(_STEP_OUTPUT_CHARS * scale))
        evidence = "\n\n".join(
            f"[step {i}] CODE:\n{_clip(c, code_cap)}\n[step {i}] OUTPUT:\n{_clip(o, output_cap)}"
            for i, (c, o) in enumerate(steps, 1)
        )
    else:
        evidence = _focused_evidence(steps, focus)
    return {
        "task": task,
        "first_draft": answers[0] if answers else "",
        "final_answer": answers[-1] if answers else "",
        "critiques": critiques,
        "evidence": evidence,
    }


def _focused_evidence(steps: list[tuple[str, str]], focus: dict[int, list[re.Match]]) -> str:
    """Every step, focus steps as number windows, within `_EVIDENCE_TOTAL_CHARS` in all."""
    share = int(_EVIDENCE_TOTAL_CHARS * _FOCUS_SHARE) // len(focus) - _STEP_OVERHEAD_CHARS
    excerpts = {i: _focus_excerpt(steps[i][1], matches, max(0, share)) for i, matches in focus.items()}
    used = sum(len(e) for e in excerpts.values())
    # What is left is split over every step's code and the other steps' output.
    slots = len(steps) + (len(steps) - len(focus)) * _STEP_OUTPUT_CHARS / _STEP_CODE_CHARS
    room = max(0, _EVIDENCE_TOTAL_CHARS - used - len(steps) * _STEP_OVERHEAD_CHARS)
    code_cap = min(_STEP_CODE_CHARS, int(room / slots))
    output_cap = min(_STEP_OUTPUT_CHARS, int(code_cap * _STEP_OUTPUT_CHARS / _STEP_CODE_CHARS))
    parts = []
    for i, (c, o) in enumerate(steps):
        shown = f"(excerpts around the questioned numbers)\n{excerpts[i]}" if i in excerpts else _clip(o, output_cap)
        parts.append(f"[step {i + 1}] CODE:\n{_clip(c, code_cap)}\n[step {i + 1}] OUTPUT:\n{shown}")
    evidence = "\n\n".join(parts)
    # The hard cap: the estimates above leave room, this makes it certain.
    return evidence if len(evidence) <= _EVIDENCE_TOTAL_CHARS else _clip(evidence, _EVIDENCE_TOTAL_CHARS)


def build_assumptions_prompt(inputs: dict) -> str:
    return f"TASK:\n{inputs['task']}\n\nFINAL_ANSWER:\n{inputs['final_answer']}"


def parse_assumptions(text: str) -> list[dict]:
    """The assumptions to check, each with its `queries`; [] when the reply isn't usable.

    A single "query" (the earlier format) is accepted as one query.
    """
    data = _extract_json(text)
    if data is None:
        return []
    items = []
    for a in _dicts(data.get("assumptions")):
        raw = a.get("queries")
        queries = [str(q).strip() for q in raw] if isinstance(raw, list) else []
        if str(a.get("query", "")).strip():
            queries.insert(0, str(a["query"]).strip())
        queries = list(dict.fromkeys(q for q in queries if q))[:MAX_QUERIES]
        if queries:
            items.append({**a, "queries": queries})
    return items[:MAX_ASSUMPTIONS]


# A number with a unit, for picking the sentences of a page that report values.
_UNIT_RE = re.compile(
    r"\d[\d.,]*\s*(?:x\s*10\S*\s*)?"
    r"(ppm|ng/mg|µg/mL|ug/mL|mg/mL|mg/L|g/L|ng/mL|pg/mL|ng/dose|pg/dose|mM|µM|uM|nM|"
    r"h-1|h⁻¹|/h|W/m3|W/m³|rpm|m/s|vvm|mmHg|mOsm/kg|log|LRV|cells/mL|pg/cell/day|%)",
    re.IGNORECASE,
)
# ng/mg and ppm are the same quantity for HCP, so either matches the other.
_UNIT_ALIASES = {"ppm": "ng/mg", "ng/mg": "ppm", "ug/ml": "µg/ml", "µg/ml": "ug/ml"}
_SENTENCES_PER_PAGE = 4
_SENTENCE_CHARS = 320


def numeric_sentences(text: str, query: str, value: str = "", limit: int = _SENTENCES_PER_PAGE) -> list[str]:
    """The sentences of a page that report a value for the assumption.

    A sentence qualifies when it holds a number with a unit and shares two
    words with the query (one, if the unit matches the assumption's own value);
    units matching the assumption's value rank first. Page
    previews chosen by word overlap tended to show prose about the topic and
    not the numbers the reviewer needs to judge the value.
    """
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 3}
    wanted = {u.lower() for u in _UNIT_RE.findall(value or "")}
    wanted |= {_UNIT_ALIASES[u] for u in wanted if u in _UNIT_ALIASES}
    scored = []
    for sentence in re.split(r"(?<=[.!?])\s+|\s*\|\s*\n", re.sub(r"[ \t]+", " ", text or "")):
        sentence = sentence.strip()
        units = {u.lower() for u in _UNIT_RE.findall(sentence)}
        if not units:
            continue
        overlap = sum(1 for w in words if w in sentence.lower())
        # Two words in common, or one plus the assumption's own unit: a single
        # shared word ("cells") lets in every number on the page.
        if overlap < 2 and not (overlap and units & wanted):
            continue
        score = overlap + (3 if units & wanted else 0) + (0 if units == {"%"} else 1)
        scored.append((score, " ".join(sentence.split())[:_SENTENCE_CHARS]))
    scored.sort(key=lambda pair: -pair[0])
    return [sentence for _, sentence in scored[:limit]]


def _cached_page(url: str) -> str:
    """The text of a page the last searches returned (no new request)."""
    from biomentis.tool.literature import _WEB_PAGE_CACHE

    return _WEB_PAGE_CACHE.get(url, "")


def check_assumptions(assumptions: list[dict], search=None, page_text=None) -> str:
    """Search each assumption's query; the results as the ASSUMPTION_CHECKS section.

    `search(query, num_results)` returns formatted results; it defaults to the
    agent's own web search (`search_google`: Ollama's web search when
    OLLAMA_API_KEY is set). For each result URL, `page_text(url)` (default: the
    page text the search returned) supplies the full page, from which the
    sentences reporting values are added. A failed search is recorded, not raised.
    """
    if search is None:
        from biomentis.tool.literature import search_google as search
    if page_text is None:
        page_text = _cached_page
    blocks = []
    for i, a in enumerate(assumptions, 1):
        queries = a.get("queries") or [str(a.get("query", "")).strip()]
        values, seen_urls, sections = [], set(), []
        for query in queries:
            try:
                results = str(search(query, num_results=_RESULTS_PER_ASSUMPTION) or "(no results)")
            except Exception as exc:
                results = f"(search failed: {type(exc).__name__}: {exc})"
            for url in re.findall(r"^URL: (\S+)", results, re.MULTILINE):
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                try:
                    found = numeric_sentences(page_text(url), query, str(a.get("value", "")))
                except Exception:
                    found = []
                values += [f"      - {sentence} [{url}]" for sentence in found]
            sections.append(f"    RESULTS for \"{query}\":\n{_clip(results, _SEARCH_TEXT_CHARS)}")
        blocks.append(
            f"[{i}] ASSUMPTION: {a.get('assumption', '')}\n"
            f"    VALUE USED: {a.get('value', '')}\n"
            f"    QUOTE: {a.get('quote', '')}\n"
            f"    SEARCH: {' | '.join(queries)}\n"
            f"    VALUES REPORTED IN THE RESULTS:\n"
            + ("\n".join(values) if values else "      (none found)")
            + "\n" + "\n".join(sections)
        )
    return "\n\n".join(blocks)


def require_sources(review: Review, assumption_checks: str) -> Review:
    """Mark "unclear" any supported/contradicted assumption that cites no searched URL.

    The verdict must rest on what the search found. A reviewer marked a
    harvest HCP of 100 mg/L "supported" as "typical for CHO harvest", with
    no source; the search had returned no value for it at all.
    """
    urls = set(re.findall(r"https?://[^\s\]\)\"'>]+", assumption_checks or ""))
    for a in review.assumptions:
        status = str(a.get("status", "")).lower()
        if status not in ("supported", "contradicted"):
            continue
        evidence = str(a.get("evidence", ""))
        if not any(url.rstrip(".,;") in evidence for url in urls):
            a["status"] = "unclear"
            a["evidence"] = f"{evidence} (marked {status} without citing a searched source)".strip()
    return review


# How far outside the reported range the answer's value may sit before the
# reviewer's own numbers overrule its status.
OUTSIDE_FACTOR = 3.0

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_SCALE_WORDS = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
_QUANTITY_RE = re.compile(
    r"(?P<bound>[<>≤≥]\s*|up to\s+|above\s+|below\s+)?"
    r"(?:10\s*\^\s*(?P<pow>[-−]?\d+)|10(?P<sup>[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
    r"|(?P<num>\d[\d,]*(?:\.\d+)?)(?:\s*[x×*]\s*10\s*(?:\^\s*(?P<mpow>[-−]?\d+)|(?P<msup>[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+))"
    r"|[eE](?P<epow>[+-]?\d+))?(?P<scale>(?:\s+hundred)?(?:\s+(?:thousand|million|billion))?)"
    r"|(?:several|a few)?\s*(?P<wscale>hundred thousand|thousand|million|billion))"
    r"\s*(?P<unit>(?!(?:to|and|or)\b)[A-Za-zµμ%]+(?:\s*/\s*[A-Za-zµμ]+)?)?",
    re.IGNORECASE,
)


def _unit(text: str | None) -> str:
    unit = re.sub(r"\s+", "", (text or "").lower()).replace("μ", "u").replace("µ", "u")
    return {"ppm": "ng/mg"}.get(unit, unit)


def _quantities(text: str) -> list[tuple[float, str, str]]:
    """(value, normalised unit, bound) for each number in `text`, words like "million" included."""
    found = []
    for m in _QUANTITY_RE.finditer(text or ""):
        if m.group("pow") or m.group("sup"):
            value = 10 ** float((m.group("pow") or m.group("sup")).translate(_SUPERSCRIPTS).replace("−", "-"))
        elif m.group("num"):
            value = float(m.group("num").replace(",", ""))
            power = m.group("mpow") or m.group("msup") or m.group("epow")
            if power:
                value *= 10 ** float(power.translate(_SUPERSCRIPTS).replace("−", "-"))
            for word in (m.group("scale") or "").lower().split():
                value *= 100 if word == "hundred" else _SCALE_WORDS[word]
        elif m.group("wscale"):
            value = 1.0
            for word in m.group("wscale").lower().split():
                value *= 100 if word == "hundred" else _SCALE_WORDS[word]
        else:
            continue
        bound = (m.group("bound") or "").strip().lower()
        bound = ">" if bound in (">", "≥", "above") else "<" if bound in ("<", "≤", "up to", "below") else ""
        found.append((value, _unit(m.group("unit")), bound, m.start(), m.end()))
    # In "100,000 to 1,000,000 ng/mg" or "2-4 ppm" the first number takes the second's unit.
    for i in range(len(found) - 1):
        value, unit, bound, start, end = found[i]
        if not unit and re.fullmatch(r"\s*(?:-|–|‑|—|to)\s*", (text or "")[end : found[i + 1][3]]):
            found[i] = (value, found[i + 1][1], bound, start, end)
    return [(value, unit, bound) for value, unit, bound, _, _ in found]


def _reported_range(text: str, unit: str) -> tuple[float, float] | None:
    values = [(v, b) for v, u, b in _quantities(text) if u == unit]
    if not values:
        return None
    if len(values) == 1:
        value, bound = values[0]
        return (value, float("inf")) if bound == ">" else (0.0, value) if bound == "<" else (value, value)
    numbers = [v for v, _ in values]
    return min(numbers), max(numbers)


def compare_reported(review: Review) -> Review:
    """Mark "contradicted" an assumption whose own numbers put it far outside the reported range.

    The reviewer fills "reported" (in the sources' units) and "converted" (the
    answer's value in those units) but may still judge wrongly: one wrote
    14,793 ng/mg against a reported 100,000-1,000,000 ng/mg and called the
    value realistic. When both are in the same unit and the answer's value is
    more than OUTSIDE_FACTOR times outside the range, the status becomes
    "contradicted". "converted" must start from the answer's own value; that
    value and the result are compared, and either one fitting a reported
    range in its unit keeps the reviewer's status. Ranges written in words
    ("several hundred thousand to several million") are read too.
    """
    flipped = False
    for a in review.assumptions:
        if str(a.get("status", "")).lower() == "contradicted":
            continue
        converted = _quantities(str(a.get("converted", "")))
        used = _quantities(str(a.get("value", "")))
        # Only a conversion of the answer's own value (its first number, not the conditions after it) counts,
        # and only that value and the result are compared, not a divisor such as the titer.
        start = [q for q in converted if used and abs(q[0] - used[0][0]) <= 1e-9 * max(abs(used[0][0]), 1)]
        if not start:
            continue
        outside, inside = [], False
        for value, unit, _ in (start[0], converted[-1]):
            span = _reported_range(str(a.get("reported", "")), unit)
            if not unit or span is None or value <= 0:
                continue
            low, high = span
            if value * OUTSIDE_FACTOR < low or value > high * OUTSIDE_FACTOR:
                outside.append((value, unit, low, high))
            else:
                inside = True
        # Any of the answer's values that fits a reported range in the same unit settles it.
        if outside and not inside:
            value, unit, low, high = outside[-1]
            shown = f"{low:,.4g}" if low == high else f">{low:,.4g}" if high == float("inf") else f"{low:,.4g}–{high:,.4g}"
            a["evidence"] = (
                f"{a.get('evidence', '')} (marked {a.get('status') or 'unclear'} by the reviewer, but its own numbers "
                f"put {value:,.4g} {unit} more than {OUTSIDE_FACTOR:g}× outside the reported {shown} {unit})"
            ).strip()
            a["status"] = "contradicted"
            flipped = True
    if flipped and review.verdict == "accept":
        review.verdict = "accept_with_corrections"
    return review


def build_review_prompt(inputs: dict, assumption_checks: str = "", rubric=None) -> str:
    """The review request; with a `rubric` (gate.rubric.Rubric), a RUBRIC section for COVERAGE."""
    critiques = "\n\n".join(f"[round {i}]\n{c}" for i, c in enumerate(inputs["critiques"], 1)) or "(none)"
    same = inputs["first_draft"] == inputs["final_answer"]
    sections = [
        f"TASK:\n{inputs['task']}",
        f"EVIDENCE (code the agent ran and its output):\n{inputs['evidence'] or '(no code was run)'}",
        "FIRST_DRAFT:\n" + ("(same as FINAL_ANSWER: the self-critique did not change it)" if same else inputs["first_draft"]),
        f"SELF_CRITIQUE (the agent's own feedback, which led to the revision):\n{critiques}",
        "ASSUMPTION_CHECKS (web search results for the answer's key assumptions):\n"
        + (assumption_checks or "(none: no assumptions were checked)"),
        f"FINAL_ANSWER:\n{inputs['final_answer']}",
    ]
    if rubric is not None:
        sections.append(
            "RUBRIC:\nWhat FINAL_ANSWER must deliver, respect and teach; report each ID under coverage.\n"
            + "\n".join(f"- [{item.id}] ({item.kind}) {item.text}" for item in rubric.items)
        )
    return "\n\n".join(sections)


def _extract_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def _dicts(value) -> list[dict]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def parse_review(text: str) -> Review:
    """A Review from the reviewer's reply; `error` set when it isn't usable."""
    data = _extract_json(text)
    if data is None:
        return Review(error="the reviewer's reply was not valid JSON")
    verdict = str(data.get("verdict", "")).strip().lower().replace(" ", "_")
    if verdict not in VERDICTS:
        return Review(error=f"the reviewer gave no valid verdict ({verdict or 'none'})")
    return Review(
        verdict=verdict,
        summary=str(data.get("summary", "")).strip(),
        changes=_dicts(data.get("changes")),
        problems=_dicts(data.get("problems")),
        rationale=_dicts(data.get("rationale")),
        assumptions=_dicts(data.get("assumptions")),
        coverage=_dicts(data.get("coverage")),
    )


def _rubric_id(value) -> str:
    """A rubric ID as reviewers echo it ("[d1]", " D1 ") in canonical form ("D1")."""
    return str(value).strip("[] ").upper()


def coverage_findings(review: Review, rubric) -> list[Finding]:
    """A blocking `rubric` finding for each rubric item the answer does not deliver.

    An item is delivered only when the reviewer marked it "present" with a
    quote; "missing", "wrong", "present" without a quote, any other status,
    and an ID the reviewer did not report all block. The key is the item's
    ID, so the user's "accept as is" holds for the run however the reviewer
    words it next round.
    """
    reported: dict[str, dict] = {}
    for entry in review.coverage:
        reported.setdefault(_rubric_id(entry.get("id", "")), entry)
    findings = []
    for item in rubric.items:
        entry = reported.get(_rubric_id(item.id))
        status = str(entry.get("status", "")).strip().lower() if entry else ""
        quote = str(entry.get("quote", "")).strip() if entry else ""
        note = str(entry.get("issue", "")).strip() if entry else ""
        if status == "present" and quote:
            continue
        if entry is None:
            issue = f"rubric {item.id} ({item.kind}) was not checked by the reviewer: {item.text}"
        elif status == "wrong":
            issue = f"rubric {item.id} ({item.kind}) is addressed wrongly: {item.text}"
        elif status == "present":
            issue = f"rubric {item.id} ({item.kind}) is marked present with no quote, so it counts as missing: {item.text}"
        else:
            issue = f"rubric {item.id} ({item.kind}) is missing: {item.text}"
        if note:
            issue += f" ({note})"
        findings.append(Finding(check="rubric", severity="blocking", quote=quote if status == "wrong" else "",
                                issue=issue, fix=f"deliver {item.id} in the answer", key=f"rubric:{item.id}"))
    return findings


def _norm(text) -> str:
    return " ".join(str(text).lower().split())


def review_findings(review: Review) -> list[Finding]:
    """The review as `review` findings: its problems, and the assumptions the sources contradict.

    Problems block on "needs rework" and are warnings otherwise ("accept"
    has none). Each problem is keyed by its kind and quote, so "accept as is"
    survives the reviewer rewording it next round; problems on one quote are
    merged into one finding. A contradicted assumption (`compare_reported` can
    mark one without the reviewer listing a problem) is always reported: it
    blocks on "needs rework" or when the reviewer marked a decision resting on
    it, else it is a warning.
    """
    if review.error:
        return []
    rework = review.verdict == "needs_rework"
    severity = "blocking" if rework else "warning"
    findings: list[Finding] = []
    by_key: dict[str, Finding] = {}
    for p in review.problems if review.verdict != "accept" else []:
        quote = str(p.get("quote", "")).strip()
        kind = p.get("kind", "problem")
        issue = f"[{kind}] {p.get('issue', '')}".strip()
        key = f"review:{_norm(kind)}:{_norm(quote)}" if quote else ""
        if key and key in by_key:  # same quote: one finding, one accept box
            by_key[key].issue += f"; {issue}"
            continue
        f = Finding(check="review", severity=severity, quote=quote, issue=issue,
                    fix=str(p.get("fix", "")).strip(), key=key)
        findings.append(f)
        if key:
            by_key[key] = f
    if not findings and rework:
        # A rework verdict with no problem listed must still block.
        findings.append(Finding(check="review", severity="blocking", quote="",
                                issue=f"the reviewer asked for rework: {review.summary or 'no reason given'}"))
    for a in review.assumptions:
        if str(a.get("status", "")).lower() != "contradicted":
            continue
        decision = str(a.get("decision_relevant", a.get("decision", ""))).strip().lower() in ("true", "yes", "1")
        name = str(a.get("assumption", "")).strip() or "an assumption"
        reported = str(a.get("reported", "")).strip()
        findings.append(Finding(
            check="review", severity="blocking" if rework or decision else "warning",
            quote=str(a.get("quote", "")).strip(),
            issue=(f"[assumption] {name} = {a.get('value', '')} is contradicted by the sources"
                   + (f" (they report {reported})" if reported else "") + f": {a.get('evidence', '')}").strip(),
            fix=f"check {name} against the sources cited and use a sourced value",
            key=f"review:assumption:{_norm(name)}",
        ))
    return findings


def same_model_family(a: str, b: str) -> bool:
    """Whether two model names share a vendor token: the first segment split on - : / or space."""
    def vendor(name: str) -> str:
        return re.split(r"[-:/ ]", (name or "").strip().lower(), maxsplit=1)[0]

    return bool(vendor(a)) and vendor(a) == vendor(b)


def feedback_for_agent(review: Review, reviewer: str) -> str:
    """The correction request: the reviewer's problems, to fix and nothing else."""
    lines = [f"{REVIEW_FEEDBACK_PREFIX} ({reviewer}) and found problems that must be fixed:", ""]
    for i, p in enumerate(review.problems, 1):
        lines.append(f"{i}. [{p.get('kind', 'problem')}] \"{p.get('quote', '')}\"")
        lines.append(f"   Problem: {p.get('issue', '')}")
        if p.get("fix"):
            lines.append(f"   Suggested fix: {p['fix']}")
    contradicted = [a for a in review.assumptions if a.get("status") == "contradicted"]
    if contradicted:
        lines += ["", "Assumptions the published sources contradict (check them, and use a sourced value):"]
        lines += [f"- {a.get('assumption', '')} = {a.get('value', '')}: {a.get('evidence', '')}" for a in contradicted]
    invalid = [c for c in review.changes if c.get("status") == "invalid"]
    if invalid:
        lines += ["", "Changes from your first draft that the reviewer judged wrong:"]
        lines += [f"- {c.get('change', '')}: {c.get('reason', '')}" for c in invalid]
    lines += [
        "",
        "Fix only these. Where a number is involved, recompute it in code inside <execute></execute> and use "
        "the computed value. If you disagree with a finding, check it in code or against a source and say "
        "why in the answer. Then give the complete corrected final answer inside <solution></solution>.",
    ]
    return "\n".join(lines)


def format_review(review: Review, reviewer: str, after_correction: bool) -> tuple[str, str]:
    """(title, markdown) for showing one review under the answer."""
    title = "🔎 Re-review after correction" if after_correction else "🔎 Independent review"
    if review.error:
        return title, f"**Reviewer:** {reviewer}\n\nNo review: {review.error}."
    parts = [f"**Reviewer:** {reviewer} · **Verdict:** {VERDICT_LABELS[review.verdict]}"]
    if review.summary:
        parts.append(review.summary)
    if review.problems:
        rows = [
            f"- **{p.get('kind', 'problem')}** — “{p.get('quote', '')}”: {p.get('issue', '')}"
            + (f" *Fix:* {p['fix']}" if p.get("fix") else "")
            for p in review.problems
        ]
        parts.append("**Problems found**\n" + "\n".join(rows))
    if review.changes:
        marks = {"valid": "✓", "invalid": "✗", "unverified": "?"}
        rows = [
            f"- {marks.get(c.get('status', ''), '?')} {c.get('change', '')} — {c.get('reason', '')}" for c in review.changes
        ]
        parts.append("**Changes made by the self-critique**\n" + "\n".join(rows))
    if review.rationale:
        marks = {"yes": "✓", "no": "✗", "unverified": "?"}
        rows = [
            f"- {marks.get(str(r.get('sound', '')).lower(), '?')} {r.get('decision', '')} — {r.get('note', '')}"
            for r in review.rationale
        ]
        parts.append("**Workflow rationale**\n" + "\n".join(rows))
    if review.assumptions:
        marks = {"supported": "✓", "contradicted": "✗", "unclear": "?"}
        rows = [
            f"- {marks.get(str(a.get('status', '')).lower(), '?')} {a.get('assumption', '')} "
            f"({a.get('value', '')}"
            + (f"; reported: {a['reported']}; as used: {a['converted']}" if a.get("reported") and a.get("converted") else "")
            + f") — {a.get('evidence', '')}"
            for a in review.assumptions
        ]
        parts.append("**Key assumptions checked against the literature**\n" + "\n".join(rows))
    return title, "\n\n".join(parts)
