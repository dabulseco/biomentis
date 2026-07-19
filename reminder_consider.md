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

## 2. Other items TBD
_(none yet — add here as you think of them, promote to `recommend.md` when ready to act)_
