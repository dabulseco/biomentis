# Step tracing: cataloguing what the agent gets wrong

`biomentis/eval/step_trace.py` records one JSONL line per step of every agent
run and classifies each step's outcome, so failures can be counted and ranked
instead of read one transcript at a time. It exists to answer a single
question: *where should the next harness or tool go?*

`biomentis/eval/biomentis_eval1.py` scores final answers against ground truth.
This is the complement — it scores the trajectory.

## Using it

Tracing is on by default and costs nothing but a few JSON lines per step:

```python
agent = A1()                      # writes traces/<timestamp>-<hash>.jsonl
agent.go("...")
```

Turn it off with `A1(trace=False)` or `BIOMENTIS_TRACE=0`; change the location
with `A1(trace_dir=...)` or `BIOMENTIS_TRACE_DIR`. Records are flushed per
step, so a run that is killed or crashes still leaves a usable trace.

Then aggregate across runs:

```bash
python -m biomentis.eval.step_trace report traces/
python -m biomentis.eval.step_trace report traces/ --json    # for further analysis
python -m biomentis.eval.step_trace dump traces/             # raw records
```

## What a step outcome means

The agent graph is two nodes, and both are instrumented.

`generate` records the branch taken (`solution` / `execute` / `think` /
`retry_parse_error` / `abort_parse_errors`), LLM latency, retry count, token
usage, and two compliance signals: whether the model emitted no tags at all,
and whether it wrote a markdown ``` fence that the loop silently rescued into
an `<execute>`.

`execute` records the code, language, duration, tools touched, output size,
whether the output hit the 10K truncation cap, whether this step re-ran code an
earlier step already tried (thrashing), and a classified status:

| status | meaning |
| --- | --- |
| `ok` | produced output with no error signature |
| `error` | the code raised, or the runner reported failure |
| `silent_error` | the code succeeded and a tool returned an error *string* |
| `timeout` | exceeded `timeout_seconds` |
| `empty` | ran cleanly but produced nothing the model could use |
| `noop` | routed to execute with no `<execute>` block to run |

`silent_error` is the important one. Many tools in `biomentis/tool/` return
their failures rather than raising:

```
Error performing web search after 3 attempts: Error code: 401 - ... invalid x-api-key
```

Exit status is success, the observation looks like data, and the model often
continues as though it had results. Binary pass/fail instrumentation scores
this as a passing step; the classifier matches it against the error-string
conventions the tools actually use and marks it failed.

## Failure classes

Each class carries a `layer` — where the fix goes — and concrete build
suggestions that the report aggregates into a ranked backlog:

| layer | meaning |
| --- | --- |
| `credentials` | secrets and preflight validation |
| `environment` | installed packages, conda envs, binaries |
| `data` | data lake contents and path resolution |
| `external_api` | third-party service behaviour (rate limits, outages) |
| `tool` | a Biomentis tool's own code or signature |
| `agent_code` | the code the model wrote |
| `model` | the model's formatting or reasoning behaviour |
| `harness` | the agent loop itself |

Rules live in `RULES` at the top of `step_trace.py`, matched most-specific
first. When `unclassified_error` shows up in a report, add a rule for it —
that is the intended maintenance loop.

## Run-level signals

`run_end` records how the run actually terminated (`solution`,
`aborted_parse_errors`, `stopped_after_execution`, `no_solution`,
`exception`) — a distinction the transcript hides — plus **retrieval
precision**: the fraction of tools the retriever put in the prompt that the
agent went on to use. Low precision with high `wrong_tool_usage` means the
retriever is fine and the tool descriptions are not; low precision with tools
never called at all means the retriever is the problem.
