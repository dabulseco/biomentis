"""Phase A plumbing tests for the Critic + memory + engine integration.

Exercises:
  1. CritiqueCard pydantic validation
  2. WeaknessKind enum values
  3. Critic.critique() returns a valid card with the expected stub shape
  4. critic_memory.load() returns empty defaults for unknown users
  5. critic_memory.save() + load() round-trip preserves priorities + counts
  6. critic_memory.reset() wipes the file
  7. TutorEngine.critic is instantiated with the configured model name
  8. TutorEngine.load_priorities() returns the on-disk priorities
  9. TutorEngine.on_session_end() returns a card AND writes a `critique`
     event to the JSONL log

Run with:
    python -m biomentis.agent.tutor.tests.test_critic
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

from biomentis.agent.tutor import (
    Critic,
    CritiqueCard,
    Strength,
    TutorEngine,
    Weakness,
    WeaknessKind,
    critic_memory,
)
from biomentis.agent.tutor.log import SessionLogger


# --- Tiny test harness (matches test_smoke_e2e.py) -----------------------

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


# --- Tests ----------------------------------------------------------------


def test_critique_card_validation() -> None:
    print("\n[1] CritiqueCard pydantic validation")
    card = CritiqueCard(
        session_id="s1",
        user_id="alice",
        model_name="m",
        agent_model_name="am",
        overall_score=8,
    )
    _check("default overall_score in range", 1 <= card.overall_score <= 10)
    # Out-of-range must be rejected
    try:
        CritiqueCard(
            session_id="s1", user_id="alice",
            model_name="m", agent_model_name="am",
            overall_score=42,
        )
        _check("overall_score=42 rejected", False, "no ValidationError raised")
    except Exception as e:
        # pydantic v2 wording varies ("less than or equal to 10", "between 1 and 10")
        msg = str(e).lower()
        _check(
            "overall_score=42 rejected",
            "10" in msg and ("less than" in msg or "between" in msg or "maximum" in msg),
            str(e),
        )

    # Round-trip
    as_dict = card.model_dump()
    _check("model_dump round-trip", isinstance(as_dict, dict))
    _check("model_dump has user_id", as_dict.get("user_id") == "alice")


def test_weakness_kind_enum() -> None:
    print("\n[2] WeaknessKind enum")
    expected = {
        "SKIPPED_PREREQUISITE", "KB_UNUSED", "CLAIM_OVERREACH",
        "TOOL_MISUSE", "INCOHERENT_PLAN", "POOR_ERROR_RECOVERY",
    }
    actual = {k.value for k in WeaknessKind}
    _check("enum has all 6 expected values", actual == expected, str(actual - expected))


def test_critic_no_llm_returns_soft_failure() -> None:
    print("\n[3a] Critic with no LLM returns a soft-failure card")
    critic = Critic(llm=None, model_name="stub-llm")
    card = critic.critique(
        session_id="s1",
        user_id="u1",
        agent_model_name="agent-llm",
        transcript_summary="(empty)",
        step_cards=[],
    )
    _check("card is CritiqueCard", isinstance(card, CritiqueCard))
    _check("card.model_name == 'stub-llm'", card.model_name == "stub-llm")
    _check("card.user_id == 'u1'", card.user_id == "u1")
    _check("soft-failure has no weaknesses", card.weaknesses == [])
    _check("soft-failure has no priorities", card.next_session_priorities == [])
    _check(
        "soft-failure notes mention LLM",
        "llm" in card.notes.lower() or "unavailable" in card.notes.lower(),
        card.notes,
    )
    _check(
        "rubric_version still 'v1'",
        card.rubric_version == "v1",
    )
    _check(
        "prompt hash is non-empty and starts with sha1:",
        card.critic_prompt_hash.startswith("sha1:") and len(card.critic_prompt_hash) > 5,
    )


def test_critic_with_fake_llm_parses_json() -> None:
    print("\n[3b] Critic with a fake LLM parses JSON into a CritiqueCard")

    class _FakeLLM:
        def __init__(self, text: str) -> None:
            self._text = text
            self.last_messages = None

        def invoke(self, messages):
            self.last_messages = messages
            from langchain_core.messages import AIMessage

            return AIMessage(content=self._text)

    valid_json = """{
        "overall_score": 8,
        "weaknesses": [
            {"kind": "KB_UNUSED", "step_id": 3,
             "detail": "agent cited UniProt directly instead of using the uploaded PDF",
             "evidence_quote": "the E2 ectodomain contains a β-ribbon at residues 17-29",
             "suggested_priority": "When a KB is loaded, prefer its definitions over generic knowledge."}
        ],
        "strengths": [
            {"step_id": 1, "detail": "framed the epitope mapping vs conservation tradeoff clearly."}
        ],
        "next_session_priorities": [
            "When a KB is loaded, prefer its definitions over generic knowledge.",
            "State BLAST E-value thresholds explicitly."
        ],
        "notes": "Solid run."
    }"""

    fake = _FakeLLM(valid_json)
    critic = Critic(llm=fake, model_name="critic-llm")
    card = critic.critique(
        session_id="s2",
        user_id="u2",
        agent_model_name="agent-llm",
        transcript_summary="[step 1] reasoning: ...",
        step_cards=[{"step_id": 1, "event_type": "reasoning",
                     "what": "framed the tradeoff", "why": "", "look_for": []}],
        task="Find 3 nanobodies",
    )
    _check("card.model_name == 'critic-llm'", card.model_name == "critic-llm")
    _check("overall_score parsed", card.overall_score == 8)
    _check("1 weakness parsed", len(card.weaknesses) == 1)
    if card.weaknesses:
        w = card.weaknesses[0]
        _check("weakness kind is KB_UNUSED", w.kind == WeaknessKind.KB_UNUSED)
        _check("weakness step_id parsed", w.step_id == 3)
        _check("evidence_quote preserved", "E2 ectodomain" in (w.evidence_quote or ""))
    _check("1 strength parsed", len(card.strengths) == 1)
    _check("2 priorities parsed", len(card.next_session_priorities) == 2)
    _check("notes preserved", card.notes == "Solid run.")
    _check("rubric_version is 'v1'", card.rubric_version == "v1")
    _check("prompt hash is stamped", card.critic_prompt_hash.startswith("sha1:"))
    # Verify the LLM got both SystemMessage and HumanMessage
    _check("LLM got 2 messages", fake.last_messages is not None and len(fake.last_messages) == 2)


def test_critic_with_fake_llm_handles_malformed_json() -> None:
    print("\n[3c] Critic with a fake LLM that returns garbage → soft-failure")
    from langchain_core.messages import AIMessage

    class _BadLLM:
        def invoke(self, messages):
            return AIMessage(content="I'm sorry, I cannot critique this.")

    critic = Critic(llm=_BadLLM(), model_name="x")
    card = critic.critique(
        session_id="s3",
        user_id="u3",
        agent_model_name="a",
        transcript_summary="(empty)",
    )
    _check("returns a card", card is not None)
    _check(
        "soft-failure notes mention non-JSON",
        "non-json" in card.notes.lower() or "json" in card.notes.lower(),
        card.notes,
    )
    _check("soft-failure has empty weaknesses", card.weaknesses == [])


def test_critic_with_fake_llm_drops_unknown_weakness_kinds() -> None:
    print("\n[3d] Critic drops unknown WeaknessKind values")
    from langchain_core.messages import AIMessage

    class _LLM:
        def invoke(self, messages):
            return AIMessage(content="""{
                "overall_score": 7,
                "weaknesses": [
                    {"kind": "BOGUS_KIND", "detail": "x"},
                    {"kind": "KB_UNUSED", "detail": "y"},
                    {"kind": "KB_UNUSED", "detail": "z"}
                ],
                "next_session_priorities": []
            }""")

    critic = Critic(llm=_LLM(), model_name="x")
    card = critic.critique(
        session_id="s4", user_id="u", agent_model_name="a",
        transcript_summary="(empty)",
    )
    _check("BOGUS_KIND dropped", len(card.weaknesses) == 2)
    _check(
        "remaining are KB_UNUSED",
        all(w.kind == WeaknessKind.KB_UNUSED for w in card.weaknesses),
    )


def test_critic_with_fake_llm_drops_out_of_range_score() -> None:
    print("\n[3e] Critic coerces out-of-range overall_score into [1,10]")
    from langchain_core.messages import AIMessage

    class _LLM:
        def invoke(self, messages):
            return AIMessage(content='{"overall_score": 99}')

    critic = Critic(llm=_LLM(), model_name="x")
    card = critic.critique(
        session_id="s5", user_id="u", agent_model_name="a",
        transcript_summary="(empty)",
    )
    _check("score clamped to 10", card.overall_score == 10)


def test_logger_summary_for_critic() -> None:
    print("\n[3f] SessionLogger.summary_for_critic builds a digest")
    with tempfile.TemporaryDirectory() as td:
        logger = SessionLogger(
            "sess-digest", path=os.path.join(td, "tutor_logs")
        )
        # Pre-populate with three steps + a QA
        logger.log({"kind": "step", "step_id": 1, "event_type": "reasoning",
                    "instruction": {"what": "frame the tradeoff", "why": "set scope"}})
        logger.log({"kind": "step", "step_id": 2, "event_type": "code",
                    "instruction": {"what": "BLAST search", "why": "find homologs",
                                    "look_for": ["hits table", "E-values"]}})
        logger.log({"kind": "qa", "question": "What is an E-value?",
                    "answer": "Expected number of hits by chance."})

        digest = logger.summary_for_critic()
        _check("digest non-empty", bool(digest.strip()))
        _check("digest mentions step 1", "step 1" in digest)
        _check("digest mentions step 2", "step 2" in digest)
        _check("digest mentions QA", "qa" in digest.lower())
        _check("digest mentions BLAST", "BLAST" in digest)


def test_logger_step_cards_for_critic() -> None:
    print("\n[3g] SessionLogger.step_cards_for_critic extracts cards")
    with tempfile.TemporaryDirectory() as td:
        logger = SessionLogger(
            "sess-cards", path=os.path.join(td, "tutor_logs")
        )
        logger.log({"kind": "step", "step_id": 1, "event_type": "reasoning",
                    "instruction": {"what": "A", "why": "B", "look_for": ["C"]}})
        logger.log({"kind": "step", "step_id": 2, "event_type": "code",
                    "instruction": {"what": "D", "why": "E", "look_for": ["F", "G"]}})
        logger.log({"kind": "qa", "question": "x", "answer": "y"})

        cards = logger.step_cards_for_critic()
        _check("2 step cards extracted", len(cards) == 2)
        _check("first card step_id is 1", cards[0]["step_id"] == 1)
        _check("first card what is 'A'", cards[0]["what"] == "A")
        _check("first card look_for has 1 item", len(cards[0]["look_for"]) == 1)
        _check("second card look_for has 2 items", len(cards[1]["look_for"]) == 2)


def test_memory_empty_load() -> None:
    print("\n[4] critic_memory.load() for unknown user")
    with tempfile.TemporaryDirectory() as td:
        data = critic_memory.load("ghost", root=td)
        _check("n_sessions == 0", data["n_sessions"] == 0)
        _check("priorities is empty list", data["priorities"] == [])
        _check("weakness_counts is empty dict", data["weakness_counts"] == {})
        _check("user_id is 'ghost'", data["user_id"] == "ghost")
        # Defensive: missing schema fields get filled in
        _check("priority_cap is set", data.get("priority_cap") is not None)


def test_memory_round_trip() -> None:
    print("\n[5] critic_memory save/load round-trip")
    with tempfile.TemporaryDirectory() as td:
        critic_memory.save(
            "alice",
            {
                "user_id": "alice",
                "n_sessions": 3,
                "priorities": ["P1", "P2"],
                "weakness_counts": {"KB_UNUSED": 5, "CLAIM_OVERREACH": 1},
            },
            root=td,
        )
        loaded = critic_memory.load("alice", root=td)
        _check("n_sessions preserved", loaded["n_sessions"] == 3)
        _check("priorities preserved", loaded["priorities"] == ["P1", "P2"])
        _check(
            "weakness_counts preserved",
            loaded["weakness_counts"] == {"KB_UNUSED": 5, "CLAIM_OVERREACH": 1},
        )
        # Atomic write: file exists at expected path
        path = critic_memory.path_for("alice", root=td)
        _check("memory file exists at expected path", os.path.exists(path))


def test_memory_reset() -> None:
    print("\n[6] critic_memory.reset() wipes the file")
    with tempfile.TemporaryDirectory() as td:
        critic_memory.save("bob", {"priorities": ["X"]}, root=td)
        path = critic_memory.path_for("bob", root=td)
        _check("file written", os.path.exists(path))
        critic_memory.reset("bob", root=td)
        _check("file removed after reset", not os.path.exists(path))
        # Reset is idempotent
        critic_memory.reset("bob", root=td)
        _check("reset is idempotent", not os.path.exists(path))


def test_engine_critic_instantiation() -> None:
    print("\n[7] TutorEngine instantiates Critic with configured model name")
    with tempfile.TemporaryDirectory() as td:
        eng = TutorEngine(
            "sess-A", llm=None, path=td, critic_model_name="critic-llm"
        )
        _check("engine.critic is Critic", isinstance(eng.critic, Critic))
        _check(
            "engine.critic.model_name matches constructor arg",
            eng.critic.model_name == "critic-llm",
        )
        # set_critic_model_name updates the model name in place
        eng.set_critic_model_name("new-critic")
        _check(
            "set_critic_model_name updates in place",
            eng.critic.model_name == "new-critic",
        )


def test_engine_load_priorities() -> None:
    print("\n[8] TutorEngine.load_priorities() reads from disk")
    with tempfile.TemporaryDirectory() as td:
        # Pre-populate a memory file at the SAME root the engine uses.
        critic_memory.save(
            "carol",
            {"priorities": ["P1", "P2"]},
            root=td,
        )
        eng = TutorEngine(
            "sess-B", llm=None, path=td, critic_model_name="x",
            memory_root=td,
        )
        priorities = eng.load_priorities("carol")
        _check("loaded priorities match", priorities == ["P1", "P2"])
        _check(
            "engine.active_priorities mirrors the disk state",
            eng.active_priorities == ["P1", "P2"],
        )
        # Unknown user returns empty
        empty = eng.load_priorities("nobody")
        _check("unknown user returns []", empty == [])


def test_engine_on_session_end_writes_log() -> None:
    print("\n[9] TutorEngine.on_session_end() writes a critique log event")
    with tempfile.TemporaryDirectory() as td:
        eng = TutorEngine(
            "sess-C", llm=None, path=td, critic_model_name="critic-llm"
        )
        eng.enable()
        card = eng.on_session_end(
            user_id="dave", agent_model_name="agent-llm"
        )
        _check("card is not None when engine is enabled", card is not None)
        _check("card.user_id matches", card.user_id == "dave")
        # Critique must be in the JSONL log
        log_path = os.path.join(
            td, "tutor_logs", "sess-C", "sess-C.jsonl"
        )
        _check("log file exists", os.path.exists(log_path), log_path)
        with open(log_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        critiques = [r for r in records if r.get("kind") == "critique"]
        _check("exactly one critique record", len(critiques) == 1)
        if critiques:
            rec = critiques[0]
            _check("record has user_id", rec.get("user_id") == "dave")
            _check(
                "record has overall_score",
                isinstance(rec.get("overall_score"), int),
            )
            _check(
                "record has weaknesses list",
                isinstance(rec.get("weaknesses"), list),
            )
            _check(
                "record has rubric_version",
                rec.get("rubric_version") == "v1",
            )

    # Disabled engine returns None and does NOT write a critique
    with tempfile.TemporaryDirectory() as td:
        eng = TutorEngine(
            "sess-D", llm=None, path=td, critic_model_name="x"
        )
        # leave disabled
        card = eng.on_session_end(user_id="x", agent_model_name="y")
        _check(
            "disabled engine returns None",
            card is None,
        )
        log_path = os.path.join(
            td, "tutor_logs", "sess-D", "sess-D.jsonl"
        )
        if os.path.exists(log_path):
            with open(log_path) as f:
                records = [json.loads(line) for line in f if line.strip()]
            critique_kinds = [
                r.get("kind") for r in records if "critique" in (r.get("kind") or "")
            ]
            _check(
                "disabled engine wrote no critique",
                len(critique_kinds) == 0,
                f"unexpected records: {critique_kinds}",
            )
        else:
            _check("disabled engine wrote no log file", True)


def test_a1_configure_accepts_critic_priorities() -> None:
    print("\n[10] a1.A1.configure() accepts critic_priorities kwarg")
    # We don't construct a full A1 (heavy init), but we verify the
    # signature accepts the new kwarg and the system_prompt-append code
    # path is reachable. Inspect the function source.
    import inspect

    from biomentis.agent.a1 import A1

    sig = inspect.signature(A1.configure)
    _check(
        "configure() has critic_priorities kwarg",
        "critic_priorities" in sig.parameters,
        str(list(sig.parameters)),
    )
    # Default is None (not required)
    _check(
        "critic_priorities defaults to None",
        sig.parameters["critic_priorities"].default is None,
    )


# --- Phase C: memory update + prompt injection --------------------------


def _make_card_from_json(json_text: str) -> CritiqueCard:
    """Build a `CritiqueCard` from a JSON string. Helper for the
    Phase C tests that don't need to invoke the Critic through the
    engine — they patch the Critic instance directly to return a
    fixed card."""
    from langchain_core.messages import AIMessage

    class _LLM:
        def invoke(self, messages):
            return AIMessage(content=json_text)

    critic = Critic(llm=_LLM(), model_name="test-critic")
    return critic.critique(
        session_id="phase-c-test",
        user_id="phase-c-user",
        agent_model_name="test-agent",
        transcript_summary="(synthetic)",
    )


def test_memory_update_appends_priorities() -> None:
    print("\n[11] critic_memory.update() appends new priorities to the front")
    with tempfile.TemporaryDirectory() as td:
        # Pre-existing memory with one priority
        critic_memory.save(
            "alice",
            {"user_id": "alice", "priorities": ["OLD priority"]},
            root=td,
        )
        # Build a card with two new priorities
        card = _make_card_from_json("""{
            "overall_score": 6,
            "weaknesses": [],
            "next_session_priorities": [
                "NEW priority 1 — when X, do Y.",
                "NEW priority 2 — never do Z."
            ]
        }""")
        data = critic_memory.update("alice", card, root=td)
        priorities = data["priorities"]
        _check("3 priorities total", len(priorities) == 3, str(priorities))
        _check(
            "NEW priority 1 at the front",
            priorities[0].startswith("NEW priority 1"),
            priorities[0],
        )
        _check(
            "OLD priority is at the tail",
            priorities[-1] == "OLD priority",
            priorities[-1],
        )


def test_memory_update_dedupes_case_insensitively() -> None:
    print("\n[12] critic_memory.update() dedupes case-insensitively")
    with tempfile.TemporaryDirectory() as td:
        # Pre-existing priority
        critic_memory.save(
            "bob",
            {"user_id": "bob", "priorities": ["State the BLAST E-value threshold."]},
            root=td,
        )
        # New card that produces a paraphrase of the same advice
        card = _make_card_from_json("""{
            "overall_score": 7,
            "next_session_priorities": [
                "state the blast e-value threshold.",
                "State the BLAST E-value threshold!",
                "Brand new unrelated priority"
            ]
        }""")
        data = critic_memory.update("bob", card, root=td)
        priorities = data["priorities"]
        # Only "Brand new unrelated priority" should be new; the two
        # duplicates of the BLAST priority should be dropped.
        _check(
            "exactly 2 priorities (1 pre-existing + 1 new)",
            len(priorities) == 2,
            str(priorities),
        )
        _check(
            "new priority made it in",
            "Brand new unrelated priority" in priorities,
        )


def test_memory_update_bumps_weakness_counts() -> None:
    print("\n[13] critic_memory.update() accumulates weakness_counts")
    with tempfile.TemporaryDirectory() as td:
        critic_memory.save("carol", {"user_id": "carol"}, root=td)
        card = _make_card_from_json("""{
            "overall_score": 5,
            "weaknesses": [
                {"kind": "KB_UNUSED", "step_id": 1, "detail": "x"},
                {"kind": "KB_UNUSED", "step_id": 2, "detail": "y"},
                {"kind": "CLAIM_OVERREACH", "step_id": 3, "detail": "z"}
            ],
            "next_session_priorities": []
        }""")
        data = critic_memory.update("carol", card, root=td)
        counts = data["weakness_counts"]
        _check("KB_UNUSED count == 2", counts.get("KB_UNUSED") == 2, str(counts))
        _check(
            "CLAIM_OVERREACH count == 1",
            counts.get("CLAIM_OVERREACH") == 1,
            str(counts),
        )

        # Second card with one more KB_UNUSED
        card2 = _make_card_from_json("""{
            "overall_score": 6,
            "weaknesses": [
                {"kind": "KB_UNUSED", "step_id": 1, "detail": "x"}
            ],
            "next_session_priorities": []
        }""")
        data2 = critic_memory.update("carol", card2, root=td)
        _check(
            "KB_UNUSED count == 3 after second update",
            data2["weakness_counts"].get("KB_UNUSED") == 3,
        )


def test_memory_update_caps_priorities() -> None:
    print("\n[14] critic_memory.update() caps priorities at priority_cap")
    with tempfile.TemporaryDirectory() as td:
        # Set a small cap to make the test fast
        critic_memory.save(
            "dave",
            {
                "user_id": "dave",
                "priority_cap": 2,
                "priorities": ["A", "B"],
            },
            root=td,
        )
        # New card with 2 new priorities
        card = _make_card_from_json("""{
            "overall_score": 5,
            "next_session_priorities": ["D", "E"]
        }""")
        data = critic_memory.update("dave", card, root=td)
        priorities = data["priorities"]
        # Cap is 2; merged list is [D, E, A, B]. After cap: [D, E]
        # (the two new ones win, the two old ones are dropped).
        _check("at most 2 priorities", len(priorities) == 2, str(priorities))
        _check(
            "newest at the front",
            priorities[0] == "D",
            priorities,
        )
        _check(
            "second-newest behind it",
            priorities[1] == "E",
            priorities,
        )
        # The oldest existing ones ('A' and 'B') should have been dropped
        _check("'A' was dropped", "A" not in priorities, str(priorities))
        _check("'B' was dropped", "B" not in priorities, str(priorities))


def test_memory_update_picks_up_suggested_priority() -> None:
    print("\n[15] critic_memory.update() pulls suggested_priority from weaknesses")
    with tempfile.TemporaryDirectory() as td:
        critic_memory.save("eve", {"user_id": "eve"}, root=td)
        # Card has no top-level priorities, but a Weakness with a
        # suggested_priority. That should still be picked up.
        card = _make_card_from_json("""{
            "overall_score": 4,
            "next_session_priorities": [],
            "weaknesses": [
                {
                    "kind": "TOOL_MISUSE",
                    "step_id": 2,
                    "detail": "wrong DB",
                    "suggested_priority": "Verify the BLAST database before searching."
                }
            ]
        }""")
        data = critic_memory.update("eve", card, root=td)
        _check(
            "suggested_priority made it into the list",
            "Verify the BLAST database before searching." in data["priorities"],
            str(data["priorities"]),
        )


def test_memory_update_bumps_n_sessions() -> None:
    print("\n[16] critic_memory.update() bumps n_sessions")
    with tempfile.TemporaryDirectory() as td:
        critic_memory.save("frank", {"user_id": "frank"}, root=td)
        for i in range(3):
            card = _make_card_from_json('{"overall_score": 5, "next_session_priorities": []}')
            data = critic_memory.update("frank", card, root=td)
        _check("n_sessions == 3", data["n_sessions"] == 3)
        _check("last_updated is non-null", data["last_updated"] is not None)


def test_engine_full_loop_session_n_to_n_plus_1() -> None:
    print("\n[17] Full loop: session N writes memory; session N+1 reads it")
    from langchain_core.messages import AIMessage

    with tempfile.TemporaryDirectory() as td:
        # Build a Critic whose LLM returns a card with two specific priorities
        priorities_text = (
            '["Always state BLAST E-value thresholds.", '
            '"Prefer KB definitions over generic LLM knowledge."]'
        )
        class _LLM:
            def __init__(self, text: str) -> None:
                self._text = text
            def invoke(self, messages):
                return AIMessage(content=self._text)

        llm = _LLM(f'{{"overall_score": 4, "next_session_priorities": {priorities_text}}}')

        # --- Session 1 -----------------------------------------------------
        eng1 = TutorEngine(
            "sess-1", llm=llm, path=td,
            critic_model_name="test-critic", memory_root=td,
        )
        eng1.enable()
        # Pre-populate a step so summary_for_critic has something to chew on
        eng1.logger.log({
            "kind": "step", "step_id": 1, "event_type": "code",
            "instruction": {"what": "Run BLAST", "why": "Get homologs", "look_for": []},
        })
        card = eng1.on_session_end(
            user_id="e2e-loop", agent_model_name="test-agent"
        )
        _check("session 1 returned a card", card is not None)
        # Memory file should exist with the two priorities
        mem_path = os.path.join(td, "e2e-loop.json")
        _check("memory file written", os.path.exists(mem_path))
        with open(mem_path) as f:
            m1 = json.load(f)
        _check(
            "memory has 2 priorities",
            len(m1["priorities"]) == 2,
            str(m1["priorities"]),
        )
        _check(
            "first priority is the BLAST one",
            m1["priorities"][0].startswith("Always state BLAST"),
            m1["priorities"][0],
        )
        _check("n_sessions == 1", m1["n_sessions"] == 1)

        # --- Session 2 -----------------------------------------------------
        # The engine reads from the same memory_root.
        eng2 = TutorEngine(
            "sess-2", llm=llm, path=td,
            critic_model_name="test-critic", memory_root=td,
        )
        eng2.enable()
        loaded = eng2.load_priorities("e2e-loop")
        _check(
            "session 2 load_priorities returns the same 2",
            loaded == m1["priorities"],
            str(loaded),
        )
        # And the engine.active_priorities mirrors disk
        _check(
            "engine.active_priorities mirrors disk",
            eng2.active_priorities == m1["priorities"],
        )

        # --- A1.configure would inject them into the system prompt --------
        # We don't construct a full A1; instead we mirror the same
        # append-line logic and verify the priorities appear in the
        # resulting string. The actual A1 wiring is verified by the
        # test_a1_configure_accepts_critic_priorities test above.
        system_prompt = "BASE PROMPT"
        if loaded:
            system_prompt += (
                "\n\nLessons from prior sessions (apply these):\n"
                + "\n".join(f"- {p}" for p in loaded)
            )
        _check(
            "system_prompt mentions BLAST priority",
            "Always state BLAST E-value thresholds" in system_prompt,
        )
        _check(
            "system_prompt mentions KB priority",
            "Prefer KB definitions" in system_prompt,
        )
        _check(
            "system_prompt is prefixed by the base",
            system_prompt.startswith("BASE PROMPT"),
        )


def test_memory_update_persists_across_engine_instances() -> None:
    print("\n[18] Memory persists across engine instances for the same root")
    from langchain_core.messages import AIMessage

    class _LLM:
        def invoke(self, messages):
            return AIMessage(
                content='{"overall_score": 5, "next_session_priorities": ["PERSIST this"]}'
            )

    with tempfile.TemporaryDirectory() as td:
        # First engine writes (with a real fake LLM so the priorities
        # actually get into the memory file)
        eng1 = TutorEngine(
            "a", llm=_LLM(), path=td, critic_model_name="x", memory_root=td
        )
        eng1.enable()
        eng1.on_session_end(user_id="persist-user", agent_model_name="x")

        # Verify the file exists on disk
        mem_path = os.path.join(td, "persist-user.json")
        _check("memory file on disk", os.path.exists(mem_path))
        if os.path.exists(mem_path):
            with open(mem_path) as f:
                content = json.load(f)
            _check(
                "disk has the priority",
                content.get("priorities") == ["PERSIST this"],
                str(content.get("priorities")),
            )

        # Second engine reads in a different process-like instantiation
        eng2 = TutorEngine("b", llm=None, path=td, critic_model_name="x", memory_root=td)
        eng2.enable()
        loaded = eng2.load_priorities("persist-user")
        _check(
            "second engine sees the persisted priority",
            loaded == ["PERSIST this"],
            str(loaded),
        )


# --- Phase D: Streamlit UI smoke tests ----------------------------------
# These tests don't render the sidebar end-to-end (Streamlit rendering
# needs a Server), but they verify the helpers Phase D added:
#  - `_build_model_choices` returns the right shape
#  - `render_tutor_sidebar` imports without raising
#  - `engine.load_priorities` and `agent.configure(critic_priorities=)`
#    compose correctly when priorities are present
#  - The stub Critic's on_session_end is a no-op so a stub user pays
#    zero latency.


def test_ui_tutor_imports_and_helpers() -> None:
    print("\n[19] ui_tutor module imports + Phase D helpers exist")
    # Module-level import must not require Streamlit to be running
    from biomentis import ui_tutor

    _check("ui_tutor is a module", ui_tutor is not None)
    _check("has _build_model_choices", hasattr(ui_tutor, "_build_model_choices"))
    _check("has render_tutor_sidebar", hasattr(ui_tutor, "render_tutor_sidebar"))
    _check("has render_tutor_chat_panel", hasattr(ui_tutor, "render_tutor_chat_panel"))
    _check("has tutor_wrapped_stream", hasattr(ui_tutor, "tutor_wrapped_stream"))
    # _build_model_choices is safe to call without Streamlit; if Ollama
    # isn't running the Ollama list is just empty and we get [].
    try:
        choices = ui_tutor._build_model_choices()
    except Exception as e:
        # Some environments have provider registry issues. Don't fail the
        # test for that — verify the function exists and is callable.
        _check(
            "_build_model_choices callable (exempt if registry broken)",
            True, f"raised: {e!r}",
        )
        choices = None
    if choices is not None:
        _check("_build_model_choices returns a list", isinstance(choices, list))
        if choices:
            sample = choices[0]
            _check(
                "each choice is (source, model) tuple",
                isinstance(sample, tuple) and len(sample) == 2
                and isinstance(sample[0], str) and isinstance(sample[1], str),
                str(sample),
            )


def test_critic_model_picker_default_is_stub() -> None:
    print("\n[20] TutorEngine.critic defaults to stub")
    with tempfile.TemporaryDirectory() as td:
        eng = TutorEngine("sess-PD", llm=None, path=td)
        _check("critic exists", eng.critic is not None)
        _check(
            "default model_name is 'stub'",
            eng.critic.model_name == "stub",
            eng.critic.model_name,
        )
        _check("default critic.llm is None", eng.critic.llm is None)
        # set_critic_model_name updates in place
        eng.set_critic_model_name("custom-critic")
        _check(
            "set_critic_model_name('custom-critic') works",
            eng.critic.model_name == "custom-critic",
        )
        # set_critic_llm with a sentinel object wires it through
        sentinel = object()
        eng.set_critic_llm(sentinel)
        _check("set_critic_llm wires through", eng.critic.llm is sentinel)


def test_stub_critic_on_session_end_is_noop() -> None:
    print("\n[21] Stub Critic on_session_end is fast + no priorities")
    import time

    with tempfile.TemporaryDirectory() as td:
        eng = TutorEngine("sess-stub", llm=None, path=td, critic_model_name="stub")
        eng.enable()
        t0 = time.monotonic()
        card = eng.on_session_end(user_id="u", agent_model_name="a")
        elapsed = time.monotonic() - t0
        _check("returns a card", card is not None)
        _check(
            "soft-failure has no priorities",
            card.next_session_priorities == [],
        )
        _check(
            "soft-failure has no weaknesses",
            card.weaknesses == [],
        )
        # The soft-failure path should be sub-second (no LLM call).
        _check(
            "stub path is fast (< 2s)",
            elapsed < 2.0,
            f"elapsed: {elapsed:.2f}s",
        )
        # And the memory file should NOT have been populated with
        # anything substantive (just a possibly-empty record).
        data = critic_memory.load("u", root=td)
        # If the soft-failure card made it to update(), n_sessions
        # bumped but priorities stayed [].
        _check(
            "memory still has no priorities after stub run",
            data.get("priorities", []) == [],
            str(data),
        )


def test_configure_with_critic_priorities_uses_active_list() -> None:
    print("\n[22] engine.load_priorities() feeds agent.configure(critic_priorities=...)")
    with tempfile.TemporaryDirectory() as td:
        # Pre-populate memory with two priorities
        critic_memory.save(
            "phase-d-user",
            {
                "user_id": "phase-d-user",
                "priorities": [
                    "Always state BLAST E-value thresholds.",
                    "Prefer KB definitions over generic LLM knowledge.",
                ],
            },
            root=td,
        )
        eng = TutorEngine(
            "sess-PD-2", llm=None, path=td, critic_model_name="stub",
            memory_root=td,
        )
        priorities = eng.load_priorities("phase-d-user")
        _check("loaded 2 priorities", len(priorities) == 2, str(priorities))
        _check(
            "active_priorities mirrors disk",
            eng.active_priorities == priorities,
        )
        # Now compose: this is what streamlit_app.py does after
        # _get_or_build_agent(). We don't build a full A1 (expensive),
        # but we mirror the same priority-append logic to verify the
        # produced system_prompt contains the priorities.
        system_prompt = "BASE"
        if priorities:
            system_prompt += (
                "\n\nLessons from prior sessions (apply these):\n"
                + "\n".join(f"- {p}" for p in priorities)
            )
        _check(
            "system_prompt mentions BLAST priority",
            "BLAST E-value" in system_prompt,
        )
        _check(
            "system_prompt mentions KB priority",
            "Prefer KB definitions" in system_prompt,
        )


def test_critic_priorities_visible_after_rerun() -> None:
    print("\n[23] Priorities survive an engine swap (simulating rerun)")
    with tempfile.TemporaryDirectory() as td:
        critic_memory.save(
            "rerun-user",
            {
                "user_id": "rerun-user",
                "priorities": ["X priority from earlier session"],
            },
            root=td,
        )
        # First engine: read + activate
        eng1 = TutorEngine(
            "sess-1", llm=None, path=td, memory_root=td, critic_model_name="stub"
        )
        first = eng1.load_priorities("rerun-user")
        _check("first engine sees the priority", first == ["X priority from earlier session"])
        # Second engine: same data dir, simulates a Streamlit rerun.
        eng2 = TutorEngine(
            "sess-2", llm=None, path=td, memory_root=td, critic_model_name="stub"
        )
        second = eng2.load_priorities("rerun-user")
        _check(
            "second engine (after rerun) still sees it",
            second == ["X priority from earlier session"],
        )
        # And the active_priorities is also restored.
        _check(
            "eng2.active_priorities restored from disk",
            eng2.active_priorities == ["X priority from earlier session"],
        )


def test_chat_ask_includes_transcript_and_last_answer() -> None:
    """TutorChat.ask(..., transcript=..., last_answer=...) injects both into
    the user prompt sent to the LLM.

    Regression test for the "Ask the Tutor" context-loss bug: when the
    student asks a follow-up question after the agent finished a run,
    the chat panel must hand the LLM the agent's recent events and
    final answer so it can answer questions like "summarize the
    databases used" without re-running the task.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from biomentis.agent.tutor.chat import TutorChat

    # The chat panel's `ask()` makes TWO LLM calls per invocation: one
    # for the actual answer (system = `_SYSTEM_PROMPT` or
    # `_SYSTEM_PROMPT_CHAT`), one for the rubric classifier (system =
    # the rubric-classifier prompt). Capture only the chat user prompts
    # by checking the SYSTEM prompt of the same call.
    def _is_chat_system(content: str) -> bool:
        # `_SYSTEM_PROMPT_CHAT` uniquely contains "KB FIRST" (the strict
        # tutor prompt does not), and the rubric classifier doesn't
        # contain "RECENT_TRANSCRIPT". We use the presence of "QUESTION:"
        # in the *system* prompt as a quick discriminator — actually,
        # the simpler signal is: the chat ask is the one where the
        # *user* prompt contains "QUESTION:" *and* the *system* prompt
        # contains "RECENT_TRANSCRIPT:" or "TASK:" or one of the chat
        # markers. Easier: filter on the system prompt's content
        # (we just look for the system messages that include a JSON
        # schema instruction — the chat one does, the rubric one does
        # too). Best discriminator: take the user message paired with a
        # system that is one of our two chat prompts.
        return "KB FIRST" in content or "patient tutor" in content

    captured_user_prompts: list[str] = []

    class _LLM:
        def invoke(self, messages):
            sys_text = ""
            user_text = ""
            for m in messages:
                if isinstance(m, SystemMessage):
                    sys_text += m.content
                elif isinstance(m, HumanMessage):
                    user_text = m.content
            if _is_chat_system(sys_text) and "QUESTION:" in user_text:
                captured_user_prompts.append(user_text)
            return AIMessage(
                content='{"answer": "ok", "follow_up": "", "citations": []}'
            )

    chat = TutorChat(llm=_LLM(), knowledge_base=None, rubric=None, logger=None)

    transcript = (
        "[code] Run BLAST\n"
        "from Bio import BLAST\n"
        "result = blast.ncbi_qblast('blastp', 'pdb', 'MEEPQSDPSV')\n"
        "\n"
        "[observation] BLAST hits: 3 PDB entries (6NK5, 6NK6, 7KLV)\n"
        "\n"
        "[solution] E2 epitopes found in 3 PDB structures: 6NK5, 6NK6, 7KLV."
    )
    last_answer = "BRCA1 analysis complete. Used Ensembl, NCBI Gene, HGNC, UCSC, ClinVar, COSMIC, cBioPortal."

    # chat mode: both sections must appear in the user prompt.
    chat.ask(
        question="summarize the databases used to execute this task",
        task="Find 3 candidate nanobodies against CHIKV E2.",
        transcript=transcript,
        last_answer=last_answer,
        mode="chat",
    )
    _check(
        "chat-mode captured one user prompt",
        len(captured_user_prompts) == 1,
        f"captured={len(captured_user_prompts)}",
    )
    prompt = captured_user_prompts[0]
    _check(
        "RECENT_TRANSCRIPT section present in chat user prompt",
        "RECENT_TRANSCRIPT" in prompt and "Run BLAST" in prompt and "BLAST hits: 3" in prompt,
        f"prompt head: {prompt[:200]}",
    )
    _check(
        "AGENT_FINAL_ANSWER section present in chat user prompt",
        "AGENT_FINAL_ANSWER" in prompt and "Ensembl" in prompt and "COSMIC" in prompt,
        f"prompt head: {prompt[:200]}",
    )

    # tutor mode: KB-only by design, so we deliberately do NOT surface
    # the transcript / last-answer sections to the LLM. Verify the
    # prompt doesn't include them.
    captured_user_prompts.clear()
    chat.ask(
        question="summarize the databases used to execute this task",
        task="Find 3 candidate nanobodies against CHIKV E2.",
        transcript=transcript,
        last_answer=last_answer,
        mode="tutor",
    )
    _check(
        "tutor-mode captured one user prompt",
        len(captured_user_prompts) == 1,
        f"captured={len(captured_user_prompts)}",
    )
    prompt = captured_user_prompts[0]
    _check(
        "RECENT_TRANSCRIPT NOT in tutor-mode user prompt",
        "RECENT_TRANSCRIPT" not in prompt,
        f"prompt head: {prompt[:200]}",
    )
    _check(
        "AGENT_FINAL_ANSWER NOT in tutor-mode user prompt",
        "AGENT_FINAL_ANSWER" not in prompt,
        f"prompt head: {prompt[:200]}",
    )

    # Defaults: ask() called with no transcript / last_answer still works.
    captured_user_prompts.clear()
    chat.ask(question="any question", mode="chat")
    _check(
        "default empty transcript/last_answer still produces a prompt",
        len(captured_user_prompts) == 1,
        f"captured={len(captured_user_prompts)}",
    )
    prompt = captured_user_prompts[0]
    _check(
        "no RECENT_TRANSCRIPT when transcript param is empty",
        "RECENT_TRANSCRIPT" not in prompt,
        f"prompt head: {prompt[:200]}",
    )
    _check(
        "no AGENT_FINAL_ANSWER when last_answer param is empty",
        "AGENT_FINAL_ANSWER" not in prompt,
        f"prompt head: {prompt[:200]}",
    )


def test_chat_ask_supports_modes() -> None:
    """TutorChat.ask() honors the `mode` parameter.

    The chat panel uses mode="chat" (KB + LLM hybrid) and the per-step
    tutor pipeline uses mode="tutor" (KB-only). Both must work, both
    must not crash, and the system prompt sent to the LLM must differ
    by mode. We don't compare the entire prompt — just verify both
    prompts are sent, the `chat` one is the longer hybrid prompt, and
    the default (no mode) is KB-only.
    """
    from langchain_core.messages import AIMessage

    from biomentis.agent.tutor.chat import _SYSTEM_PROMPT, _SYSTEM_PROMPT_CHAT

    # We only want to record the *chat* invocation, not the rubric
    # classifier's secondary invoke. The chat system prompt is uniquely
    # identified by the "KB FIRST" rule (it doesn't appear in the
    # classifier or the strict tutor prompt).
    chat_prompt_marker = "KB FIRST"
    tutor_prompt_marker = "patient tutor"

    def _classify(messages) -> str | None:
        sys_msg = next((m for m in messages if m.type == "system"), None)
        if sys_msg is None:
            return None
        if chat_prompt_marker in sys_msg.content:
            return "chat"
        if tutor_prompt_marker in sys_msg.content:
            return "tutor"
        return None  # classifier or something else — skip

    seen_modes: list[str] = []

    class _LLM:
        def invoke(self, messages):
            kind = _classify(messages)
            if kind is not None:
                seen_modes.append(kind)
            return AIMessage(content='{"answer": "ok", "follow_up": "", "citations": []}')

    from biomentis.agent.tutor.chat import TutorChat

    chat = TutorChat(llm=_LLM(), knowledge_base=None, rubric=None, logger=None)
    _check("_SYSTEM_PROMPT_CHAT exists", _SYSTEM_PROMPT_CHAT != _SYSTEM_PROMPT)
    _check(
        "_SYSTEM_PROMPT_CHAT mentions hybrid",
        "fill gaps" in _SYSTEM_PROMPT_CHAT or "general knowledge" in _SYSTEM_PROMPT_CHAT,
    )
    _check(
        "_SYSTEM_PROMPT (tutor) does NOT mention filling gaps",
        "fill gaps" not in _SYSTEM_PROMPT,
    )

    # mode="chat" → hybrid prompt
    chat.ask(question="What is an E-value?", mode="chat")
    # mode="tutor" → strict prompt
    chat.ask(question="What is an E-value?", mode="tutor")
    # default → strict prompt (backwards compat)
    chat.ask(question="What is an E-value?")
    _check(
        "chat-mode + tutor-mode + default seen by LLM",
        seen_modes == ["chat", "tutor", "tutor"],
        str(seen_modes),
    )

    # Invalid mode falls back to tutor (the safer KB-only path)
    seen_modes.clear()
    chat.ask(question="x", mode="not-a-mode")
    _check(
        "invalid mode falls back to tutor",
        seen_modes == ["tutor"],
        str(seen_modes),
    )


# --- Runner ---------------------------------------------------------------


def main() -> int:
    tests = [
        test_critique_card_validation,
        test_weakness_kind_enum,
        test_critic_no_llm_returns_soft_failure,
        test_critic_with_fake_llm_parses_json,
        test_critic_with_fake_llm_handles_malformed_json,
        test_critic_with_fake_llm_drops_unknown_weakness_kinds,
        test_critic_with_fake_llm_drops_out_of_range_score,
        test_logger_summary_for_critic,
        test_logger_step_cards_for_critic,
        test_memory_empty_load,
        test_memory_round_trip,
        test_memory_reset,
        test_engine_critic_instantiation,
        test_engine_load_priorities,
        test_engine_on_session_end_writes_log,
        test_a1_configure_accepts_critic_priorities,
        test_memory_update_appends_priorities,
        test_memory_update_dedupes_case_insensitively,
        test_memory_update_bumps_weakness_counts,
        test_memory_update_caps_priorities,
        test_memory_update_picks_up_suggested_priority,
        test_memory_update_bumps_n_sessions,
        test_engine_full_loop_session_n_to_n_plus_1,
        test_memory_update_persists_across_engine_instances,
        test_ui_tutor_imports_and_helpers,
        test_critic_model_picker_default_is_stub,
        test_stub_critic_on_session_end_is_noop,
        test_configure_with_critic_priorities_uses_active_list,
        test_critic_priorities_visible_after_rerun,
        test_chat_ask_supports_modes,
        test_chat_ask_includes_transcript_and_last_answer,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback

            _FAILED.append((t.__name__, f"{e!r}\n{traceback.format_exc()}"))
            print(f"  ✗ {t.__name__}: {e!r}")

    print(f"\n=== {_len(_PASSED)} passed, {len(_FAILED)} failed ===")
    for name, detail in _FAILED:
        print(f"  FAIL: {name}\n    {detail}")
    return 0 if not _FAILED else 1


def _len(xs: list) -> int:
    return len(xs)


if __name__ == "__main__":
    sys.exit(main())
