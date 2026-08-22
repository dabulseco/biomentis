# Ending a model turn at its first closing tag

The agent's protocol is one `<execute>` **or** one `<solution>` per turn. A
provider stop sequence is supposed to enforce it. On the default local path it
never did.

## The silent drop

`get_llm` passes `stop_sequences=` to every provider branch. `ChatOpenAI` has
a `stop` field aliased to `stop_sequences`; `ChatAnthropic` is the mirror
image. Both work. `ChatOllama` has a field named `stop` and **no alias**, and
passing the wrong name is accepted with no warning at all:

```python
ChatOllama(model='x', temperature=0.2, stop_sequences=["</execute>"])
#   warnings:     []
#   model_kwargs: <none>
#   options sent: {'temperature': 0.2}      ← the stop is simply gone
```

So a branch that reads identically to its working neighbours can be doing
nothing, and nothing says so. (The Ollama branch didn't even pass the wrong
name — it omitted stop sequences entirely — but the trap is what made it
plausible, and it is what would break a naive fix.)

## What it cost

Everything the model wrote after `</execute>` went into `state["messages"]` as
fact. Two failures follow:

**A fabricated observation.** The model writes its own
`<observation>42</observation>`, that lands in the history, and on the next
turn it reasons over its own fiction as a real result.

**A skipped execution.** `generate` tests `answer_match` before
`execute_match` (`a1.py:1701-1706`), and both regexes search the whole
message. So this response:

```
Let me check.
<execute>print(compute())</execute>
<observation>42</observation>
Great, the answer is <solution>42</solution>
```

takes the **solution** branch and ends the run. The code never executes, and
the answer is built from output that was never computed.

## The fix: enforce the boundary locally

`biomentis/utils.py` — `truncate_after_first_tag(message) -> (kept, dropped)`

The generate node cuts each response at the end of its first `</execute>` or
`</solution>`, immediately after the existing repair for a missing closing
tag and before the branch test. That is what a stop sequence does, applied
where it holds for every provider — including any that ignores `stop` later.

Nothing that was ever used is lost: the `<execute>` and `<solution>`
extractors are non-greedy and only ever read the first block. `</think>` is
deliberately not a boundary — thinking is a preamble to a turn, not a turn.
A tagless message passes through untouched so it still reaches the
parse-error branch that re-prompts the model; truncation must not disguise a
format failure as a clean turn.

When something is cut, the node prints how much and whether it contained a
fabricated `<observation>`. If you want counts rather than lines, adding a
field to `StepTrace.record_generate` is a small follow-up — UI runs are traced
now (see `docs/background_runs.md`).

## The guard: no more silent drops

`biomentis/llm.py` — `STOP_SEQUENCE_EXEMPT_CLIENTS`, `_check_stop_sequences`

`get_llm` is now a thin wrapper: it builds the client via `_build_llm` and
then checks that a caller who asked for stop sequences actually got them,
keying off the client class rather than the source string — the class is the
thing with the wrong field name.

A client that drops them and is **not** in `STOP_SEQUENCE_EXEMPT_CLIENTS`
prints a warning naming the class and pointing at the registry. Two entries
are declared today:

| Client | Why |
| --- | --- |
| `ChatOllama` | Takes `stop=`, not `stop_sequences=`. Left unset on purpose: the default local model is a reasoning model that can emit `</execute>` while planning, and an API-side stop would cut it off mid-thought. Local truncation covers it. |
| `_ChatOpenAIResponsesNoStop` | gpt-5 rejects `stop` on the Responses API, so the subclass sets it on the client and strips it from the payload. Local truncation covers it too. |

The warning should never fire. If it does, an SDK renamed a field or a new
provider branch forgot to forward them.

## Tests

```bash
python -m biomentis.agent.tests.test_stop_sequences
```

Covers what truncation cuts and leaves, that it never removes anything the
extractors would read, the branch flip (the same message ends the run before
truncation and executes after), an end-to-end run through a real `A1` with a
scripted LLM asserting the code runs and the invented answer never becomes the
run's answer, every provider branch either applying stop sequences or
declaring why not, and the `ChatOllama` swallow itself — pinned, so an
upstream fix or regression is noticed.

## Not done

Passing `stop=` on the Ollama branch. It would save the tokens that are now
generated and discarded, but it risks halting a reasoning model mid-thought on
a `</execute>` it wrote while planning — and the correctness problem is
already fixed without it. If you want it, the change is one line plus a
`BIOMNI_STOP_SEQUENCES=0` escape hatch, and the exemption entry comes out.
