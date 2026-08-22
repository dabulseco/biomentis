"""Step-level trace recorder for the Biomentis agent loop.

The agent graph has exactly two nodes -- ``generate`` (LLM produces
``<think>``/``<execute>``/``<solution>``) and ``execute`` (the code in an
``<execute>`` block is run). Every step of every run passes through one of
them, so instrumenting those two points captures the whole trajectory.

What this module records, per step:

* ``generate``: which branch the model took, whether it emitted well-formed
  tags, whether a markdown fence had to be coerced into an ``<execute>``,
  LLM latency / retries / token usage.
* ``execute``: language, code, duration, tools touched, output size, and --
  the point of the whole exercise -- a *classified outcome*.

Outcome classification is deliberately not "did Python raise?". A large
fraction of real Biomentis failures are **silent**: the code runs fine and
returns a string that happens to say ``Error performing web search after 3
attempts: Error code: 401 ... invalid x-api-key``. Exit status is success;
the step accomplished nothing. Those are detected by matching the returned
observation against the error-string conventions the tools in
``biomentis/tool/`` actually use.

Every classified error carries ``suggestions``: concrete things that could be
built -- harness, tooling, prompt, or environment changes -- to stop that
failure recurring. Aggregating those across runs gives a ranked backlog of
agent improvements, which is the reason this recorder exists.

Usage::

    agent = A1(trace=True)              # on by default; BIOMENTIS_TRACE=0 disables
    agent.go("...")                     # writes traces/<run_id>.jsonl

    python -m biomentis.eval.step_trace report traces/
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "StepTrace",
    "classify_observation",
    "load_traces",
    "build_report",
    "format_report",
]

# --------------------------------------------------------------------------
# Failure taxonomy
# --------------------------------------------------------------------------
# Each rule maps an observable signature to (a) a stable class name, (b) the
# layer of the system at fault, and (c) suggestions for what to build. Order
# matters: the first matching rule wins, so specific rules precede generic
# ones.
#
# `layer` is the actionable part -- it says *where* the fix goes:
#   credentials  -> secrets / preflight validation
#   environment  -> installed packages, conda env, binaries
#   data         -> data lake contents and path resolution
#   external_api -> third-party service behaviour (rate limits, outages)
#   tool         -> a Biomentis tool's own code or signature
#   agent_code   -> code the LLM wrote
#   model        -> the LLM's formatting / reasoning behaviour
#   harness      -> the agent loop itself (context, budgets, retries)


@dataclass(frozen=True)
class Rule:
    name: str
    layer: str
    pattern: str
    summary: str
    suggestions: tuple[str, ...]
    flags: int = re.IGNORECASE


RULES: tuple[Rule, ...] = (
    Rule(
        name="auth_invalid_credential",
        layer="credentials",
        pattern=r"invalid x-api-key|authentication_error|401|unauthorized|"
        r"incorrect api key|api key not (found|valid)|permission_error|"
        r"authenticationerror|no api key provided",
        summary="A tool called an external API with a missing or invalid credential.",
        suggestions=(
            "Add a startup preflight that validates every credential the retrieved "
            "tools need (one cheap probe per provider) and reports all failures once, "
            "instead of surfacing them one dead step at a time.",
            "Gate credential-dependent tools out of the retrieved tool list when their "
            "key is absent, so the model is never offered a tool that cannot work.",
            "Return a machine-readable marker (e.g. 'BIOMENTIS_ERROR:MISSING_CREDENTIAL:anthropic') "
            "instead of a prose string, and have the execute node route on it -- right now "
            "the error is invisible to the loop and the model just retries the same call.",
            "Register a fallback backend for the capability (e.g. PubMed / Europe PMC / "
            "a local index for web search) so a dead credential degrades instead of blocking.",
        ),
    ),
    Rule(
        name="rate_limited",
        layer="external_api",
        pattern=r"\b429\b|rate.?limit|too many requests|quota exceeded|"
        r"overloaded_error|\b529\b|throttl",
        summary="An external API rejected the call for rate/quota reasons.",
        suggestions=(
            "Put a shared per-provider token bucket in front of tool calls so parallel "
            "or repeated steps cannot burst past the limit.",
            "Add a response cache keyed by (tool, normalized args) -- agents re-issue "
            "identical queries across steps far more often than they realise.",
            "Use exponential backoff with jitter and surface the retry budget in the "
            "observation, so the model knows to wait rather than to rewrite the query.",
        ),
    ),
    Rule(
        name="missing_dependency",
        layer="environment",
        pattern=r"modulenotfounderror|no module named|cannot import name|"
        r"importerror|command not found|is not recognized as an internal",
        summary="A Python package or CLI binary the step needed is not installed.",
        suggestions=(
            "Add the package to requirements.txt / requirements-optional.txt and pin it.",
            "Declare each tool's dependencies in the tool registry and run an import "
            "preflight over the retrieved set; drop unavailable tools from the prompt "
            "rather than letting the model discover the gap by crashing.",
            "Expose an 'install_package' capability (or a documented conda env per tool "
            "family) so the agent can self-repair instead of dead-ending.",
        ),
    ),
    Rule(
        name="missing_env_var",
        layer="credentials",
        pattern=r"keyerror: ['\"][A-Z0-9_]*(API_KEY|TOKEN|SECRET|PASSWORD)['\"]|"
        r"environment variable [A-Z0-9_]+ (is )?not set",
        summary="Code read an environment variable that is not configured.",
        suggestions=(
            "Validate required environment variables at agent construction and fail loudly "
            "with the exact .env key name.",
            "Keep .env.example in sync with the variables tools actually read (grep "
            "os.environ / os.getenv across biomentis/tool/) and check it in CI.",
        ),
    ),
    Rule(
        name="network_failure",
        layer="external_api",
        pattern=r"connectionerror|connection refused|connection reset|max retries exceeded|"
        r"temporary failure in name resolution|nameresolutionerror|sslerror|"
        r"read timed out|readtimeout|timeouterror|\b50[234]\b|bad gateway|"
        r"service unavailable|remote end closed",
        summary="A network call to an external service failed or timed out.",
        suggestions=(
            "Wrap outbound tool HTTP in one shared session with retry/backoff and a "
            "sane connect+read timeout, so transient blips never reach the model.",
            "Cache successful responses on disk; a re-run of the same task then costs "
            "nothing and is reproducible when the service is down.",
            "Mirror the small, hot reference lookups into the data lake so the common "
            "path has no network dependency at all.",
        ),
    ),
    Rule(
        name="empty_or_not_found_result",
        layer="tool",
        pattern=r"\b404\b|not found|no results? (were )?found|returned no results|"
        r"no records|no matching|empty result|no data available",
        summary="The query was well-formed but the resource or record does not exist.",
        suggestions=(
            "Add an identifier-resolution helper (gene symbol <-> Ensembl/UniProt/HGNC, "
            "species aliases, accession versions) and call it before querying -- most "
            "'not found' results are identifier-format mismatches, not absent data.",
            "Have the tool validate its identifier arguments up front and return the "
            "expected format on mismatch, so the model can correct in one step.",
            "Return near-miss suggestions ('did you mean ...') from the tool rather than "
            "a bare not-found, giving the model something to act on.",
        ),
    ),
    Rule(
        name="file_not_found",
        layer="data",
        pattern=r"filenotfounderror|no such file or directory|"
        r"errno 2|does not exist on disk",
        summary="A file or data lake path the step expected is missing.",
        suggestions=(
            "Add a path-resolution tool that maps a data lake logical name to its real "
            "path and downloads it on demand, instead of the model guessing paths.",
            "List the actual on-disk data lake contents (names + paths) in the system "
            "prompt rather than descriptions alone, so paths are never invented.",
            "Verify expected_data_lake_files at startup and report what is missing.",
        ),
    ),
    Rule(
        name="permission_denied",
        layer="environment",
        pattern=r"permissionerror|access is denied|access denied|\b403\b|forbidden|"
        r"operation not permitted|read-only file system",
        summary="The step lacked filesystem or service permission.",
        suggestions=(
            "Give the agent an explicit scratch/output directory it owns and put it in "
            "the system prompt, so writes never land somewhere unwritable.",
            "If the 403 came from a service, treat it as a credential/entitlement problem "
            "and preflight it with the other credential checks.",
        ),
    ),
    Rule(
        name="wrong_tool_usage",
        layer="tool",
        pattern=r"typeerror: .*(argument|parameter)|unexpected keyword argument|"
        r"missing \d+ required positional|attributeerror: .*has no attribute|"
        r"takes \d+ positional arguments? but",
        summary="A tool or library was called with the wrong signature.",
        suggestions=(
            "Put full call signatures (not just prose descriptions) for retrieved tools "
            "in the system prompt -- this class of failure is almost always the model "
            "guessing at parameters it was never shown.",
            "Add an introspection tool (describe_tool(name) -> signature + docstring + "
            "worked example) the agent can call before using an unfamiliar tool.",
            "Validate arguments at the tool boundary and return the correct signature in "
            "the error, so the fix takes one step instead of several.",
        ),
    ),
    Rule(
        name="undefined_name",
        layer="agent_code",
        pattern=r"nameerror|is not defined",
        summary="Code referenced a variable or function that does not exist in the REPL.",
        suggestions=(
            "The REPL persists globals between steps -- when a name is missing it usually "
            "means an earlier step failed silently. Surface a compact 'current REPL state' "
            "(defined names / loaded dataframes) in the observation so the model tracks it.",
            "Encourage self-contained execute blocks (re-import, re-load) in the system "
            "prompt, so one failed step does not poison every later one.",
        ),
    ),
    Rule(
        name="syntax_error",
        layer="model",
        pattern=r"syntaxerror|indentationerror|unexpected eof while parsing|"
        r"invalid syntax|taberror",
        summary="The generated code did not parse.",
        suggestions=(
            "Compile-check (ast.parse) the code in the execute node before running it and "
            "return the syntax error immediately -- cheaper than a full REPL round trip.",
            "If this clusters on one model, it is a model-capability signal: raise the "
            "generation quality bar for code steps or add few-shot examples.",
        ),
    ),
    Rule(
        name="parse_error_json",
        layer="tool",
        pattern=r"jsondecodeerror|expecting value: line|"
        r"xml.etree|not well-formed|parsererror",
        summary="A tool could not parse a response payload (often an HTML error page).",
        suggestions=(
            "Check content-type and HTTP status before parsing, and return the status "
            "plus a body excerpt -- a JSONDecodeError usually hides a 4xx/5xx.",
            "Add schema validation on tool responses so upstream API changes are reported "
            "as such rather than as parse noise.",
        ),
    ),
    Rule(
        name="out_of_memory",
        layer="harness",
        pattern=r"memoryerror|out of memory|killed process|cannot allocate memory|"
        r"cuda out of memory",
        summary="The step exhausted memory.",
        suggestions=(
            "Provide chunked/streaming loaders for the large data lake files and document "
            "them in the tool descriptions.",
            "Set a memory ceiling on the execution sandbox so an OOM is reported as a "
            "clean, catchable failure rather than killing the process.",
        ),
    ),
    Rule(
        name="deprecated_api",
        layer="environment",
        pattern=r"deprecat|has been removed in|no longer supported|"
        r"futurewarning: .*removed",
        summary="A library or remote API surface the tool targets has changed.",
        suggestions=(
            "Pin the library version in requirements and add a smoke test for the tool.",
            "Track which tools wrap fast-moving external APIs and add contract tests that "
            "run independently of the agent.",
        ),
    ),
    Rule(
        name="generic_tool_error",
        layer="tool",
        # The convention across biomentis/tool/*.py is to return, not raise:
        #   f"Error during ...", f"Error loading ...", f"Error querying ...",
        #   f"Error performing web search after N attempts: ..."
        pattern=r"^\s*error (during|loading|querying|retrieving|reading|running|"
        r"performing|processing|preparing|extracting|getting|fetching|downloading)\b|"
        r"error code: \d+|^\s*error:",
        summary="A Biomentis tool returned an error string rather than a result.",
        suggestions=(
            "Give tool errors a structured, greppable shape instead of free prose, so the "
            "loop can detect them (this recorder has to regex-match prose today).",
            "Decide per tool whether the failure should raise -- a returned error string "
            "reads as success to the execute node and the model often keeps going as if "
            "it had data.",
        ),
    ),
    Rule(
        name="uncaught_exception",
        layer="agent_code",
        pattern=r"traceback \(most recent call last\)|^error in execution:",
        summary="The generated code raised an exception.",
        suggestions=(
            "Confirm the traceback's exception type is covered by a specific rule above; "
            "if this class stays frequent, add one so the failure becomes actionable.",
        ),
    ),
)

# Exception type as it appears in run_with_timeout's "Error in execution: ..."
_EXC_TYPE_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning|Interrupt))\b")

# run_with_timeout / run_bash_script / run_r_code error prefixes (biomentis/utils.py)
_TIMEOUT_RE = re.compile(r"^ERROR: Code execution timed out after (\d+) seconds", re.IGNORECASE)
_EXEC_ERROR_RE = re.compile(r"^Error in execution:\s*", re.IGNORECASE)
_NO_RESULT_RE = re.compile(r"^Error: Execution completed but no result was returned")
_R_BASH_ERROR_RE = re.compile(r"^Error running (R code|Bash script|command)", re.IGNORECASE)
_TRUNCATED_RE = re.compile(r"^The output is too long to be added to context")


TIMEOUT_SUGGESTIONS = (
    "Raise timeout_seconds for the specific tool family rather than globally, or run the "
    "long call as a background job the agent polls, so one slow step does not burn the run.",
    "Add checkpointing to the long-running tools (write partial results to disk) so a "
    "timeout does not discard all the work.",
    "Encourage smaller execute blocks in the system prompt -- timeouts cluster on steps "
    "that try to do a whole analysis in one block.",
)

TRUNCATION_SUGGESTIONS = (
    "Add a summarize/paginate helper so large results are written to a file and the "
    "observation carries a path plus a preview, instead of 10K raw characters.",
    "Teach the tools that produce big payloads to return a compact summary object by "
    "default with an opt-in verbose mode.",
)

EMPTY_OUTPUT_SUGGESTIONS = (
    "The step produced no output, so the model received no feedback and typically re-runs "
    "or guesses. Auto-append the repr of the last expression (IPython-style) to the "
    "observation, or state in the system prompt that results must be printed.",
)

THRASH_SUGGESTIONS = (
    "The agent re-ran essentially the same failing code. Inject an explicit note into the "
    "observation ('you have already tried this and it failed with X') and require a "
    "different approach.",
    "Add a per-subgoal attempt budget that forces a strategy change or a graceful "
    "give-up-and-report instead of looping to the recursion limit.",
)

PARSE_ERROR_SUGGESTIONS = (
    "The model emitted neither <execute> nor <solution>. If this clusters on local Ollama "
    "models, switch those to a tool-calling / structured-output path instead of XML tags.",
    "Add one or two few-shot exchanges in the system prompt showing the exact tag format.",
    "Keep the auto-correction retry, but count it -- two strikes currently ends the run "
    "(a1.py), which shows up as a silent, unexplained termination.",
)

EXECUTE_NOOP_SUGGESTIONS = (
    "The generate node coerced a markdown ``` fence into an execute routing decision, "
    "but the execute node re-parses the message for literal <execute> tags and finds "
    "none -- so the step runs nothing and silently burns a turn. Rewrite the coerced "
    "message to wrap the code in real <execute> tags at coercion time.",
    "Make the execute node's no-match branch loud: append an observation telling the "
    "model its code block was not runnable, instead of returning an unchanged state.",
    "Guard against the resulting infinite bounce (generate -> execute -> generate with "
    "identical content) with a no-progress detector rather than the recursion limit.",
)

MARKDOWN_COERCION_SUGGESTIONS = (
    "The model used a ``` fence and the loop silently rescued it into an <execute>. That "
    "rescue hides a real compliance problem; keep it, but track the rate per model.",
    "For models with a high coercion rate, prefer native tool calling over tag parsing.",
)


def _match_rule(text: str) -> Rule | None:
    for rule in RULES:
        if re.search(rule.pattern, text, rule.flags | re.MULTILINE):
            return rule
    return None


def classify_observation(
    result: str,
    *,
    code: str = "",
    language: str = "python",
) -> dict[str, Any]:
    """Classify one execution result.

    Returns a dict with ``status``, ``error_class``, ``layer``, ``error_type``,
    ``signal`` (the matched excerpt) and ``suggestions``.

    ``status`` is one of:

    ``ok``            -- produced output, no error signature
    ``error``         -- the code raised, or the runner reported failure
    ``silent_error``  -- the code succeeded but a tool returned an error string
                         (e.g. the 401 web-search case); this is the class that
                         binary pass/fail instrumentation misses entirely
    ``timeout``       -- execution exceeded timeout_seconds
    ``empty``         -- ran cleanly but produced nothing for the model to use
    """
    text = (result or "").strip()

    if not text:
        return {
            "status": "empty",
            "error_class": "no_output",
            "layer": "harness",
            "error_type": None,
            "signal": None,
            "summary": "The step produced no output at all.",
            "suggestions": list(EMPTY_OUTPUT_SUGGESTIONS),
        }

    truncated = bool(_TRUNCATED_RE.match(text))

    # 1. Runner-level failures: the execution itself did not complete.
    if _TIMEOUT_RE.match(text):
        return {
            "status": "timeout",
            "error_class": "execution_timeout",
            "layer": "harness",
            "error_type": "Timeout",
            "signal": text[:200],
            "summary": "Execution exceeded the configured timeout.",
            "suggestions": list(TIMEOUT_SUGGESTIONS),
        }

    hard_error = bool(_EXEC_ERROR_RE.match(text) or _R_BASH_ERROR_RE.match(text) or _NO_RESULT_RE.match(text))

    rule = _match_rule(text)
    exc_match = _EXC_TYPE_RE.search(text)
    error_type = exc_match.group(1) if exc_match else None

    if rule is None:
        if hard_error:
            return {
                "status": "error",
                "error_class": "unclassified_error",
                "layer": "agent_code",
                "error_type": error_type,
                "signal": text[:300],
                "summary": "Execution failed but matched no known signature.",
                "suggestions": [
                    "Add a rule to biomentis/eval/step_trace.RULES for this signature so it "
                    "becomes actionable rather than noise.",
                ],
            }
        out: dict[str, Any] = {
            "status": "ok",
            "error_class": None,
            "layer": None,
            "error_type": None,
            "signal": None,
            "summary": None,
            "suggestions": [],
        }
        if truncated:
            out["suggestions"] = list(TRUNCATION_SUGGESTIONS)
        return out

    match = re.search(rule.pattern, text, rule.flags | re.MULTILINE)
    start = max(0, match.start() - 60) if match else 0
    # Back off to a word boundary so the excerpt does not start mid-token.
    if start and (space := text.rfind(" ", start, start + 30)) != -1:
        start = space + 1
    signal = text[start : start + 260]

    # A rule matched. Did the code actually raise, or did a tool hand back an
    # error string while reporting success? The latter is the silent case.
    status = "error" if hard_error else "silent_error"

    suggestions = list(rule.suggestions)
    if status == "silent_error":
        suggestions.append(
            "This failure was silent: the execute node saw a successful return, so the "
            "loop treated a dead step as a good one. Detect tool error strings in the "
            "execute node and mark the observation as failed."
        )
    if truncated:
        suggestions.extend(TRUNCATION_SUGGESTIONS)

    return {
        "status": status,
        "error_class": rule.name,
        "layer": rule.layer,
        "error_type": error_type,
        "signal": signal,
        "summary": rule.summary,
        "suggestions": suggestions,
    }


# --------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------

MAX_CODE_CHARS = 4000
MAX_OBS_CHARS = 2000


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]


def _normalize_code(code: str) -> str:
    """Whitespace/comment-insensitive key, for detecting a step re-run verbatim."""
    lines = []
    for line in (code or "").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


@dataclass
class StepTrace:
    """Append-only JSONL recorder for one agent process.

    One file per run: ``<trace_dir>/<run_id>.jsonl``. Writes are flushed per
    record so a crashed or killed run still leaves a usable trace -- the runs
    that die are the interesting ones.
    """

    trace_dir: str | Path = "traces"
    enabled: bool = True
    store_code: bool = True

    run_id: str | None = field(default=None, init=False)
    step_index: int = field(default=0, init=False)
    _path: Path | None = field(default=None, init=False)
    _run_start: float | None = field(default=None, init=False)
    _seen_code: dict[str, int] = field(default_factory=dict, init=False)
    _counts: dict[str, int] = field(default_factory=dict, init=False)
    _used_tools: set[str] = field(default_factory=set, init=False)
    _retrieved_tools: list[str] = field(default_factory=list, init=False)
    _meta: dict[str, Any] = field(default_factory=dict, init=False)

    # -- lifecycle ---------------------------------------------------------

    def start_run(self, task: str, **meta: Any) -> None:
        if not self.enabled:
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}-{_sha(task or 'run')[:6]}"
        self.step_index = 0
        self._run_start = time.time()
        self._seen_code = {}
        self._counts = {}
        self._used_tools = set()
        self._retrieved_tools = list(meta.get("retrieved_tools") or [])
        self._meta = dict(meta)

        directory = Path(self.trace_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._path = directory / f"{self.run_id}.jsonl"
        self._write(
            {
                "type": "run_start",
                "task": task,
                **dict(meta),
            }
        )

    def end_run(self, outcome: str, detail: str | None = None) -> dict[str, Any]:
        if not self.enabled or self._path is None:
            return {}
        used = sorted(self._used_tools)
        retrieved = list(self._retrieved_tools)
        summary = {
            "type": "run_end",
            "outcome": outcome,
            "detail": detail,
            "steps": self.step_index,
            "duration_s": round(time.time() - (self._run_start or time.time()), 2),
            "status_counts": dict(self._counts),
            "tools_used": used,
            "tools_retrieved_unused": sorted(set(retrieved) - set(used)) if retrieved else [],
            "retrieval_precision": (
                round(len(set(used) & set(retrieved)) / len(set(retrieved)), 3) if retrieved else None
            ),
            "tools_used_not_retrieved": sorted(set(used) - set(retrieved)) if retrieved else [],
        }
        self._write(summary)
        return summary

    # -- step recorders ----------------------------------------------------

    def record_generate(
        self,
        *,
        branch: str,
        raw_length: int,
        latency_s: float,
        llm_attempts: int = 1,
        coerced_from_markdown: bool = False,
        parse_error: bool = False,
        usage: dict[str, Any] | None = None,
        thinking_chars: int = 0,
    ) -> dict[str, Any]:
        """Record one pass through the ``generate`` node."""
        if not self.enabled or self._path is None:
            return {}
        self.step_index += 1
        suggestions: list[str] = []
        if parse_error:
            suggestions.extend(PARSE_ERROR_SUGGESTIONS)
        if coerced_from_markdown:
            suggestions.extend(MARKDOWN_COERCION_SUGGESTIONS)
        if llm_attempts > 1:
            suggestions.append(
                "The LLM call needed a retry to succeed. If this is frequent, the model "
                "endpoint (or the cloud-signed-in Ollama hop) is unstable -- consider a "
                "keepalive ping or a local fallback model."
            )
        record = {
            "type": "generate",
            "step": self.step_index,
            "branch": branch,
            "format_ok": not parse_error,
            "coerced_from_markdown": coerced_from_markdown,
            "parse_error": parse_error,
            "latency_s": round(latency_s, 2),
            "llm_attempts": llm_attempts,
            "raw_chars": raw_length,
            "thinking_chars": thinking_chars,
            "usage": usage or {},
            "suggestions": suggestions,
        }
        if parse_error:
            self._bump("parse_error")
        self._write(record)
        return record

    def record_execute(
        self,
        *,
        code: str,
        result: str,
        language: str,
        duration_s: float,
        tools: list[str] | None = None,
        plots: int = 0,
    ) -> dict[str, Any]:
        """Record one pass through the ``execute`` node, classified."""
        if not self.enabled or self._path is None:
            return {}
        self.step_index += 1
        tools = list(tools or [])
        self._used_tools.update(tools)

        verdict = classify_observation(result, code=code, language=language)
        self._bump(verdict["status"])

        key = _sha(_normalize_code(code))
        repeat_of = self._seen_code.get(key)
        self._seen_code.setdefault(key, self.step_index)

        suggestions = list(verdict["suggestions"])
        if repeat_of is not None and verdict["status"] != "ok":
            suggestions.extend(THRASH_SUGGESTIONS)

        record = {
            "type": "execute",
            "step": self.step_index,
            "language": language,
            "status": verdict["status"],
            "error_class": verdict["error_class"],
            "layer": verdict["layer"],
            "error_type": verdict["error_type"],
            "summary": verdict["summary"],
            "signal": verdict["signal"],
            "duration_s": round(duration_s, 2),
            "tools": tools,
            "plots": plots,
            "code_sha": key,
            "repeat_of_step": repeat_of,
            "output_chars": len(result or ""),
            "truncated": bool(_TRUNCATED_RE.match((result or "").strip())),
            "suggestions": suggestions,
        }
        if self.store_code:
            record["code"] = (code or "")[:MAX_CODE_CHARS]
            record["observation_head"] = (result or "")[:MAX_OBS_CHARS]
        self._write(record)
        return record

    def record_execute_noop(self, message: str) -> dict[str, Any]:
        """Record an execute node that ran but found nothing to execute.

        This happens when the generate node rescued a markdown ``` fence into
        an "execute" routing decision, but left the message body unchanged --
        the execute node re-parses for literal <execute> tags, finds none, and
        returns without running anything. The step is consumed, the model sees
        no observation, and the loop bounces back to generate with the same
        context. It looks like nothing at all in the transcript.
        """
        if not self.enabled or self._path is None:
            return {}
        self.step_index += 1
        self._bump("noop")
        record = {
            "type": "execute",
            "step": self.step_index,
            "language": "none",
            "status": "error",
            "error_class": "execute_node_noop",
            "layer": "harness",
            "error_type": None,
            "summary": "Routed to execute, but the message contained no <execute> block to run.",
            "signal": (message or "")[:260],
            "duration_s": 0.0,
            "tools": [],
            "plots": 0,
            "code_sha": None,
            "repeat_of_step": None,
            "output_chars": 0,
            "truncated": False,
            "suggestions": list(EXECUTE_NOOP_SUGGESTIONS),
        }
        self._write(record)
        return record

    # -- internals ---------------------------------------------------------

    def _bump(self, key: str) -> None:
        self._counts[key] = self._counts.get(key, 0) + 1

    def _write(self, record: dict[str, Any]) -> None:
        if self._path is None:
            return
        record.setdefault("run_id", self.run_id)
        record.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        try:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:  # never let telemetry break a run
            print(f"[step_trace] could not write trace record: {exc}")

    @property
    def path(self) -> Path | None:
        return self._path


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def load_traces(trace_dir: str | Path = "traces") -> list[dict[str, Any]]:
    """Load every record from every ``*.jsonl`` trace under ``trace_dir``."""
    directory = Path(trace_dir)
    records: list[dict[str, Any]] = []
    if not directory.exists():
        return records
    for path in sorted(directory.glob("*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate raw trace records into per-class statistics and a backlog."""
    runs = [r for r in records if r["type"] == "run_start"]
    ends = [r for r in records if r["type"] == "run_end"]
    execs = [r for r in records if r["type"] == "execute"]
    gens = [r for r in records if r["type"] == "generate"]

    status_counts: dict[str, int] = {}
    class_stats: dict[str, dict[str, Any]] = {}
    tool_stats: dict[str, dict[str, int]] = {}
    layer_counts: dict[str, int] = {}

    for rec in execs:
        status = rec.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

        for tool in rec.get("tools") or []:
            entry = tool_stats.setdefault(tool, {"calls": 0, "failures": 0})
            entry["calls"] += 1
            if status != "ok":
                entry["failures"] += 1

        cls = rec.get("error_class")
        if not cls:
            continue
        layer = rec.get("layer") or "unknown"
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        entry = class_stats.setdefault(
            cls,
            {
                "count": 0,
                "runs": set(),
                "layer": layer,
                "summary": rec.get("summary"),
                "silent": 0,
                "examples": [],
                "tools": {},
                "suggestions": [],
            },
        )
        entry["count"] += 1
        entry["runs"].add(rec.get("run_id"))
        if status == "silent_error":
            entry["silent"] += 1
        if rec.get("signal") and len(entry["examples"]) < 3:
            entry["examples"].append(" ".join(str(rec["signal"]).split())[:220])
        for tool in rec.get("tools") or []:
            entry["tools"][tool] = entry["tools"].get(tool, 0) + 1
        for suggestion in rec.get("suggestions") or []:
            if suggestion not in entry["suggestions"]:
                entry["suggestions"].append(suggestion)

    for entry in class_stats.values():
        entry["runs"] = len(entry["runs"])

    thrash = [r for r in execs if r.get("repeat_of_step")]
    parse_errors = [g for g in gens if g.get("parse_error")]
    coerced = [g for g in gens if g.get("coerced_from_markdown")]
    retried_llm = [g for g in gens if (g.get("llm_attempts") or 1) > 1]

    precisions = [e["retrieval_precision"] for e in ends if e.get("retrieval_precision") is not None]

    ranked = sorted(class_stats.items(), key=lambda kv: (-kv[1]["count"], kv[0]))

    return {
        "runs": len(runs),
        "completed_runs": len(ends),
        "outcomes": _tally(e.get("outcome") for e in ends),
        "steps": len(execs) + len(gens),
        "execute_steps": len(execs),
        "generate_steps": len(gens),
        "status_counts": status_counts,
        "step_success_rate": (round(status_counts.get("ok", 0) / len(execs), 3) if execs else None),
        "silent_failure_rate": (round(status_counts.get("silent_error", 0) / len(execs), 3) if execs else None),
        "layer_counts": layer_counts,
        "class_stats": dict(ranked),
        "tool_stats": dict(sorted(tool_stats.items(), key=lambda kv: (-kv[1]["failures"], -kv[1]["calls"]))),
        "thrash_steps": len(thrash),
        "parse_errors": len(parse_errors),
        "markdown_coercions": len(coerced),
        "llm_retries": len(retried_llm),
        "mean_retrieval_precision": (round(sum(precisions) / len(precisions), 3) if precisions else None),
        "unused_retrieved_tools": _tally(tool for e in ends for tool in (e.get("tools_retrieved_unused") or [])),
    }


def _tally(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def format_report(report: dict[str, Any]) -> str:
    """Render :func:`build_report` output as a readable console report."""
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("BIOMENTIS STEP TRACE REPORT")
    add("=" * 78)
    add(f"runs: {report['runs']}  (completed: {report['completed_runs']})")
    add(f"steps: {report['steps']}  (generate: {report['generate_steps']}, execute: {report['execute_steps']})")
    if report["step_success_rate"] is not None:
        add(f"execute step success rate: {report['step_success_rate']:.1%}")
        add(f"silent failure rate:       {report['silent_failure_rate']:.1%}  <- looked like success")
    if report["outcomes"]:
        add("run outcomes: " + ", ".join(f"{k}={v}" for k, v in report["outcomes"].items()))
    add("")

    add("-- step outcomes " + "-" * 60)
    for status, count in sorted(report["status_counts"].items(), key=lambda kv: -kv[1]):
        add(f"  {status:<14} {count}")
    add("")

    if report["layer_counts"]:
        add("-- failures by layer (where the fix goes) " + "-" * 35)
        for layer, count in sorted(report["layer_counts"].items(), key=lambda kv: -kv[1]):
            add(f"  {layer:<14} {count}")
        add("")

    loop_health = [
        ("parse errors (no <execute>/<solution>)", report["parse_errors"]),
        ("markdown fences rescued into <execute>", report["markdown_coercions"]),
        ("LLM calls needing a retry", report["llm_retries"]),
        ("repeated (thrashing) execute steps", report["thrash_steps"]),
    ]
    if any(count for _, count in loop_health):
        add("-- loop health " + "-" * 62)
        for label, count in loop_health:
            if count:
                add(f"  {label}: {count}")
        add("")

    if report["mean_retrieval_precision"] is not None:
        add("-- tool retrieval " + "-" * 59)
        add(f"  mean precision (retrieved tools actually used): {report['mean_retrieval_precision']:.1%}")
        unused = list(report["unused_retrieved_tools"].items())[:10]
        if unused:
            add("  most often retrieved but never used:")
            for tool, count in unused:
                add(f"    {tool} ({count} runs)")
        add("")

    failing_tools = [(t, s) for t, s in report["tool_stats"].items() if s["failures"]]
    if failing_tools:
        add("-- tools by failure count " + "-" * 51)
        for tool, stats in failing_tools[:15]:
            rate = stats["failures"] / stats["calls"] if stats["calls"] else 0
            add(f"  {tool:<40} {stats['failures']}/{stats['calls']} failed ({rate:.0%})")
        add("")

    add("=" * 78)
    add("IMPROVEMENT BACKLOG (ranked by frequency)")
    add("=" * 78)
    if not report["class_stats"]:
        add("No classified failures recorded.")
    for rank, (cls, stats) in enumerate(report["class_stats"].items(), start=1):
        add("")
        silent = f", {stats['silent']} silent" if stats["silent"] else ""
        add(f"{rank}. {cls}  [{stats['layer']}]  x{stats['count']} across {stats['runs']} run(s){silent}")
        if stats["summary"]:
            add(f"   {stats['summary']}")
        if stats["tools"]:
            top = ", ".join(f"{t} ({c})" for t, c in sorted(stats["tools"].items(), key=lambda kv: -kv[1])[:5])
            add(f"   tools involved: {top}")
        for example in stats["examples"][:1]:
            add(f"   e.g. {example}")
        add("   what to build:")
        for suggestion in stats["suggestions"]:
            add(f"     - {suggestion}")
    add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Biomentis step trace reporting")
    parser.add_argument("command", choices=["report", "dump"], help="report: aggregate; dump: raw records")
    parser.add_argument("trace_dir", nargs="?", default="traces", help="directory of *.jsonl traces")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    records = load_traces(args.trace_dir)
    if not records:
        print(f"No trace records found in {args.trace_dir!r}.")
        return 1

    if args.command == "dump":
        for record in records:
            print(json.dumps(record))
        return 0

    report = build_report(records)
    print(json.dumps(report, indent=2, default=str) if args.json else format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
