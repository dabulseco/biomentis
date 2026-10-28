# Reminder — Consider Later

A low-priority "to revisit" list. Items here are **ideas worth keeping alive** but **not** in the active plan. They are not as strongly considered as the items in [`recommend.md`](./recommend.md), which is the actively-planned backlog.

Lifecycle:

1. An item lives here while it's still an idea and you're not sure it's worth the work.
2. When you decide to act on it, move it to `recommend.md` with a fresh "how to implement" pass.
3. When it's done, delete it from both files (or move to a `CHANGELOG.md` if you keep one).

## 1. Optional: route `advanced_web_search_claude` through Anthropic when the key is set, even on Ollama

**Status today (2026-07-19):** The function auto-falls-back to `search_google` whenever `ANTHROPIC_API_KEY` is unset (see `biomni/tool/literature.py::advanced_web_search_claude`). When the key **is** set, the function still uses Anthropic — but only the original guard was relaxed; no cost guard or opt-in flag was added.

**Why this is worth revisiting:**

You said your cost concern with Anthropic is "minimize it — use it only when there are no other options." Right now the implementation only fires the Anthropic path when the key happens to be in the env, which is too coarse: a stale `ANTHROPIC_API_KEY` in your shell (left over from a different project) would silently start costing you every time the LLM picks this tool.

**What "implementing" this would look like (for whenever you move it to `recommend.md`):**

- [ ] Replace the implicit "key present → use Anthropic" with an explicit opt-in env var, e.g. `BIOMNI_ALLOW_ANTHROPIC_FALLBACK=true` (or `BIOMNI_WEB_SEARCH_USE_ANTHROPIC=true`).
- [ ] Add a per-session cost counter in `default_config` (`_anthropic_web_search_spend_usd` or similar) so a confused small model looping on this tool can't burn through your key.
- [ ] Add a `BIOMNI_WEB_SEARCH_DAILY_BUDGET` env var (suggested default: $1.00) that, when hit, causes the function to silently route to `search_google` for the rest of the session.
- [ ] Optionally distinguish the key used by this tool (`BIOMNI_WEB_SEARCH_API_KEY`) from the chat-loop key, so you can rotate one without affecting the other.
- [ ] Add a one-line note in the agent's tool description (in `biomni/tool/tool_description/literature.py`) so the LLM knows the cost behavior.

**Cost reality check (so this doesn't grow in your head):**

- One call: `max_searches=1` (default), `max_tokens=4096`. Anthropic's `web_search_20250305` tool is roughly **$0.01–$0.10 per call** depending on the model tier and how much text the search returns.
- A typical Biomni task triggers this tool 1–3 times when actually doing protocol research, or 0 times when the LLM picks a different tool.
- A small Ollama model in a confused loop is the realistic worst case: maybe 10+ calls in a row ≈ **$1**.

So the cost is bounded, but the LLM's behavior — not yours — is what determines the spend. A daily cap is the right defense.

## Notes seen in agent runs (added 2026-10-28)

Items 2–5 come from notes that show up in the agent's step observations. Each one lists what the note means, how much it matters, and the options to choose from. **Magnitude** uses three levels:

- **Low:** costs a wasted step at most; answers are not affected.
- **Medium:** can occasionally make an answer less complete or less accurate.
- **High:** regularly breaks runs or gives wrong answers. None of these items is high today.

## 2. "The output is too long to be added to context. Here are the first 10K characters..."

**What it means:** a step printed more than 10,000 characters, so the agent passed only the first 10,000 back to the model (`biomentis/agent/a1.py`, execute node). The code ran in full, so files, variables and plots are intact. It isn't an error.

**Magnitude: Medium.** Results and summaries usually print last (a best match, a p-value, a final table row). When a step prints a big table first, the model never sees the part that matters, and it either guesses or spends a step reprinting it. Usually the model notices and prints a smaller summary next step, which costs a step but not accuracy.

**Options:**

- [ ] **A. Leave it.** No work. The occasional extra step remains.
- [ ] **B. Keep both ends (recommended).** Keep the first ~6K and last ~4K characters, joined by a marker such as `… 12,345 characters omitted …`, and reword the note to match. Small change: also update `_TRUNCATED_RE` in `biomentis/eval/step_trace.py`, and add a test (e.g. in `test_context_budget`) that a long output keeps both ends.

## 3. "The model's reply had no <execute> or <solution> tag — asking it to try again."

**What it means:** the model's reply had neither code to run (`<execute>`) nor a final answer (`<solution>`), so the agent sent a short correction asking for one. Two in a row end the run ("Execution terminated due to repeated parsing errors"). A safety refusal ends the run right away.

**Magnitude: Low, already mostly fixed.** Each note costs one extra model call; answers aren't affected. Fixes already in place: DeepSeek's turns are taken from its thinking when the reply is empty, Claude's tool-call syntax is converted to `<execute>`, and the correction message was reworded so Claude doesn't refuse it. What's left is mostly DeepSeek spending its whole 8,192-token output allowance on thinking during the self-critique round (about 2 notes per DeepSeek design run in the 2026-10-26 comparison).

**Options:**

- [ ] **A. Leave it.** The cap stops loops, and runs finish.
- [ ] **B. Give DeepSeek more output room (`num_predict`) in the self-critique round.** Small change. Should remove most remaining notes for DeepSeek; slightly longer calls.
- [ ] **C. Track the count per model** in the run summary, to see whether it gets worse with a new model before deciding.

## 4. "Error: Protocols.io access token is not configured. Set PROTOCOLS_IO_ACCESS_TOKEN ..."

**What it means:** the model called `search_protocols` or `get_protocol_details` (`biomentis/tool/protocols.py`), which need a protocols.io API token, and none is set in `.env`. Not a bug.

**Magnitude: Low.** Only that step fails; the model moves on to the local protocol library (`biomentis/tool/protocols/`: Addgene, Thermo Fisher) or web and literature search. Answers to lab-methods questions may be less specific when the best protocol is only on protocols.io. Mostly shows up on lab-methods questions.

**Options:**

- [ ] **A. Leave it.** One wasted step when a model tries the tool.
- [ ] **B. Add a token.** Make a free protocols.io account, generate an access token in its developer settings, add `PROTOCOLS_IO_ACCESS_TOKEN=...` to `.env` yourself (never commit it), and restart the app. No code change.
- [ ] **C. Hide the two tools when no token is set**, so the model never tries them. Small code change; the tools come back automatically once a token is added.

## 5. "Error: No module named 'scholarly'"

**What it means:** the model called `query_scholar` (`biomentis/tool/literature.py`), which searches Google Scholar through the `scholarly` package. That package isn't installed in the `biomni` conda env.

**Magnitude: Low.** Only that step fails; the model switches to PubMed (`query_pubmed`), arXiv (`query_arxiv`) or web search, which cover most biology literature. Answers are only affected when a paper is findable solely through Google Scholar, which is rare.

**Options:**

- [ ] **A. Leave it.** One wasted step when a model tries the tool.
- [ ] **B. Install it:** `pip install scholarly==1.7.11` in the `biomni` env (the version the old env files pin). Quick, but unreliable: Google Scholar has no official API, `scholarly` scrapes it through free proxies, and Scholar often blocks that with CAPTCHAs, so the tool would fail some of the time with a different error.
- [ ] **C. Remove `query_scholar` from the model's tool list (recommended).** Small change. Stops the wasted step without adding a flaky dependency.

## 6. Evaluate the knowledge base for constraints and repeatability (added 2026-10-28)

**When:** once runs without the knowledge base (KB) are robust, so there is a baseline to compare against.

**How the KB reaches the agent today:** at the start of each run, the 4 passages that best match the prompt are retrieved (`_AGENT_CONTEXT_K = 4` in `biomentis/agent/tutor/kb.py`). They are added to the prompt, and the agent is asked to cite them. The same prompt against an unchanged KB retrieves the same 4 passages every time, so they are a fixed input to every run.

**Repeatability (a modest gain).** Temperature controls only the model's sampling, which is already small at 0 or 0.2. Most run-to-run variation comes from elsewhere:
- different paths taken early (19 steps vs 6 in the two CHO runs of 2026-10-28), which compound;
- inputs that change: PubMed and web results vary from day to day, and so does which searches the agent chooses;
- cloud models, which can vary slightly even at temperature 0.

The KB reduces the first two: with the key facts already in the prompt, the agent anchors its early decisions on them, so runs tend to converge. It narrows the variation but doesn't remove it.

**Constraints and accuracy (the bigger gain).**
- Limits and specifications come from your documents rather than the model's memory. Errors like "10 pg/dose" (correct: 10 ng/dose) are memory errors that a passage corrects directly.
- The independent reviewer sees the passages too, because they are part of the task it is given. It can then check the answer against your sources, not just its own knowledge, which matters most with a smaller reviewer such as gpt-oss:20b.
- The instruction cards already cite the KB, so vetted sources keep the agent's workflow and the teaching material consistent.

**Limits to plan around (possible changes when the KB is in use):**
- [ ] **Only 4 passages per run, matched to the whole prompt.** For an 8-stage workflow that may cover only one or two stages. Retrieve per stage, or more passages for long workflows.
- [ ] **The evidence check doesn't count the KB as a source** (`missing_evidence` in `biomentis/run_quality.py`). It still demands a PubMed or web search when the KB already holds the answer. Count cited KB passages as evidence.
- [ ] **Answers are only as good as the documents.** Where a document is wrong, or disagrees with the literature, the agent will follow the document. Prefer authoritative sources: ICH Q5A/Q5B/Q5D/Q6B/Q11, the WHO limit on residual DNA, peer-reviewed process papers such as the NISTCHO benchmark. Course notes are a weaker basis.

**The experiment:**
1. Pick one prompt (e.g. the CHO antibody workflow) and fix the settings: same agent model, agent temperature 0, self-critique temperature ~0.3, same self-critique rounds, evidence check on, same Critic model.
2. Run it 3 times **without** the KB. For each run, record:
   - the key decisions (e.g. scale-up criterion, purification order);
   - the key numbers (e.g. agitation speed, residual DNA limit, titer);
   - the independent review verdicts and the number of problems found;
   - the run-quality line.
3. Load the KB and run the same prompt 3 times **with** it, at the same settings, recording the same things.
4. Compare. Signs the KB helps:
   - less spread in decisions and numbers across the 3 runs;
   - fewer review findings, especially on numbers, units and limits;
   - more statements citing KB sources;
   - no new errors where a KB document disagrees with the literature.
5. Optionally, repeat on a second machine or model, to separate model and hardware effects from KB effects.

## 7. Other items TBD
_(none yet — add here as you think of them, promote to `recommend.md` when ready to act)_
