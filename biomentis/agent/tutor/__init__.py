"""Biomentis-Tutor: an optional instructional layer over the A1 agent.

The package is *not* imported by `A1` itself; the Streamlit app decides whether
to instantiate a `TutorEngine` and pass it into `launch_streamlit_demo` via
the optional `stream_fn` kwarg. When the engine is not present, the agent
behaves exactly as it did before this layer existed.

Public surface (all re-exported here for convenience):

    from biomentis.agent.tutor import (
        TutorEngine,        # orchestrator (engine.py)
        KnowledgeBase,      # Chroma-backed KB (kb.py)
        InstructionCard,    # per-step teaching card (instruction.py)
        Rubric,             # Bloom + DOK + teacher rubric (rubric.py)
        SessionLogger,      # JSONL log writer (log.py)
        Critic,             # end-of-session reviewer (critic.py)
        CritiqueCard,       # structured critic output (critic.py)
        Weakness,           # one flagged failure (critic.py)
        WeaknessKind,       # failure taxonomy (critic.py)
        Strength,           # one positive observation (critic.py)
        critic_memory,      # per-user priority memory (memory.py)
    )
"""

from biomentis.agent.tutor.chat import ChatTurn, TutorChat
from biomentis.agent.tutor.critic import Critic, CritiqueCard, Strength, Weakness, WeaknessKind
from biomentis.agent.tutor.engine import TutorEngine
from biomentis.agent.tutor.instruction import InstructionCard, InstructionGenerator
from biomentis.agent.tutor.kb import KnowledgeBase
from biomentis.agent.tutor import memory as critic_memory
from biomentis.agent.tutor.log import SessionLogger
from biomentis.agent.tutor.rubric import Rubric, RubricClassification, RubricObjective

__all__ = [
    "TutorEngine",
    "KnowledgeBase",
    "InstructionCard",
    "InstructionGenerator",
    "Rubric",
    "RubricClassification",
    "RubricObjective",
    "SessionLogger",
    "TutorChat",
    "ChatTurn",
    "Critic",
    "CritiqueCard",
    "Weakness",
    "WeaknessKind",
    "Strength",
    "critic_memory",
]
