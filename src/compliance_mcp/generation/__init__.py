"""Generacion con citaciones verificadas contra el corpus."""
from .context import AnswerContext, build_context
from .engine import AnswerEngine
from .providers import ProviderChain, ProviderError, build_chain
from .schema import AnswerDraft, Citation, VerifiedAnswer
from .verify import verify_citations

__all__ = [
    "AnswerContext",
    "AnswerDraft",
    "AnswerEngine",
    "Citation",
    "ProviderChain",
    "ProviderError",
    "VerifiedAnswer",
    "build_chain",
    "build_context",
    "verify_citations",
]
