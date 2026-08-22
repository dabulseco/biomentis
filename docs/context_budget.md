# Staying inside the context window

The failure, verbatim:

```
Run failed: ResponseError: The prompt is too long: 262960,
model maximum context length: 262144 (status code: 400)
```

Twenty minutes of work, gone, at the point the run had the most to show for
itself.

## Where the window actually goes

The prompt that "caused" it was **1,212 tokens — 0.46% of the window**. A
prompt-length check would have waved it through. Measured, for the app's own
configuration:

| | tokens | share of 262,144 |
| --- | ---: | ---: |
| The user's prompt | 1,212 | 0.46% |
| System prompt, before a word is exchanged | 42,410 | 16.2% |
| └ descriptions of all 224 tools | 32,232 | 12.3% |
| └ data lake catalog | 1,554 | 0.6% |
| └ software library catalog | 2,792 | 1.1% |
| Each execute step (observation capped at 10k chars, plus the assistant turn) | ~3,000–4,000 | ~1.4% |

Every run starts at ~43.6k and adds a few thousand a step that never comes
back out. That is roughly **55–70 steps of runway**, and `recursion_limit` is
500 — so on a long multi-phase task the context is what stops the loop, by
crashing it. Nothing was watching, so nothing spoke up.

(The 32k of tool descriptions is there because `streamlit_app.py` runs with
`use_tool_retriever=False`. Turning the retriever on is the single largest
lever available and has not been pulled — it changes which tools the agent can
see per task, which is a behavior decision, not a budget one.)

## The guard

`biomentis/context_budget.py` — `fit_messages` runs before **both** LLM calls
in the graph (`generate`, and the self-critic, which re-sends the entire
history and is therefore the largest request the graph makes).

```
budget = limit × 0.95 − 8,192 reserved for the answer

pass 1   collapse observations, oldest first, to a one-line placeholder
pass 2   drop whole messages, oldest first
pass 3   cut the largest remaining message by however much is still over
```

Never touched: the system prompt (the contract the loop parses against), the
first human message (the task), and the last four turns (what the model is
reasoning about right now). Pass 3 exists so a request is never *knowingly*
sent over the limit — one 400,000-character observation can exceed the whole
budget on its own, and only cutting it helps.

**The run's own history is never mutated.** Only the outgoing request is
trimmed, so the transcript, the export and the journal stay complete, and the
trim is recomputed from scratch on the next call.

### Finishing instead of dying

Trimming buys steps; it does not buy infinite steps. When the trimmed request
is still at 90% of budget, the agent is told:

> You are nearly out of context for this task. Do NOT run any more code. Write
> your final answer now… Say plainly which parts of the original request you
> were unable to complete and what you would do next.

A run that ends with a partial report and an honest list of what it did not
reach is worth immeasurably more than one that ends with a 400. If the agent
is told this three times and is still calling for more code, the run is ended
for it — each further step only makes it less likely the answer gets written
at all. The forced end also skips the self-critic round, since that would fire
one more full-history request at exactly the wrong moment.

## Where the limit comes from

1. `BIOMNI_MAX_CONTEXT_TOKENS` — an explicit override always wins
2. a limit **learned from a previous context-length error** — the provider's
   own number, for the exact deployment, persisted to
   `runs/context_limits.json` so a given model can only surprise you once
3. `KNOWN_CONTEXT_LIMITS`, matched on model name
4. otherwise **no guard at all**

Point 4 is deliberate. A guessed ceiling that is too low cuts runs short for
no reason; one that is too high protects nothing. Silence is the honest
default, and point 2 means the first failure is also the last.

Ollama's `/api/show` is deliberately **not** consulted: for
`deepseek-v4-pro:cloud` it reports `context_length: 1048576` while the
endpoint enforces `262144`. Trusting it would disable the guard precisely
where it is needed.

Token counts are estimated with `cl100k_base` (or characters ÷ 4 without
tiktoken) rather than the model's own tokenizer, which is why the limit is
discounted by 5% before anything else is subtracted.

## Related: the doomed retry

`_invoke_llm_with_retry` caught bare `Exception` and retried three times with
2s and 4s backoff. A 400 for context length is deterministic, so the original
failure sent **three** 262k-token requests, six seconds apart, before giving
up. Context-length errors are now recognised and re-raised on the first
attempt — and the limit named in the message is learned on the way past.
Genuine transient failures (the stale-cloud-connection case the retry was
written for) still get their three attempts.

## Tests

```bash
python -m biomentis.agent.tests.test_context_budget
```

Covers estimation, each trimming pass, the invariants (system prompt, task and
recent turns preserved; the caller's list never mutated; a request never sent
over the limit), the wrap-up and hard-stop thresholds, limit resolution
including the refusal to invent one, learning a limit from an error and
persisting it, the no-retry behavior, and — closest to the original failure —
an end-to-end run against a stubborn scripted model that ignores every request
to finish, asserting the run ends cleanly in a handful of calls rather than
crashing or spinning to the recursion limit.

## Not done

- **Turning the tool retriever back on** (~32k tokens, ~12% of the window).
- **A pre-flight estimate at submit** and a live headroom caption in the UI.
  Both are useful; neither would have caught this run, which had a green light
  right up to the step that killed it.
