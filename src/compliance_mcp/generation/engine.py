"""Orquestador: recuperar -> generar -> validar esquema -> verificar -> servir.

La politica que aplica este modulo es lo que hace que la tasa de citas
alucinadas SERVIDAS sea cero por construccion y no por confianza en el modelo:
lo que no se verifica no se sirve, y una respuesta que se queda sin respaldo se
convierte en rehuso. La tasa de alucinacion del modelo EN BRUTO se sigue
midiendo aparte, en el arnes de evaluacion; si se midiera solo lo servido, el
gate seria vacuo.
"""
from __future__ import annotations

from typing import Any

from ..config import Config
from ..cost import Cost
from ..cost import compute as compute_cost
from ..ingest import ControlRecord
from ..observability import StageTimings, current_trace_id, log_event, stage
from ..retrieval.search import Retriever, SearchFilters
from .context import AnswerContext, build_context
from .providers import ProviderChain, build_chain
from .schema import AnswerDraft, ProviderInfo, Verification, VerifiedAnswer
from .verify import verify_citations

# Texto servido cuando la respuesta del modelo no sobrevive a la verificacion.
# Se dice lo que paso, no "no lo se": el usuario merece saber que hubo una
# respuesta y que se retuvo por no estar respaldada.
UNSUPPORTED_MESSAGE = (
    "I could not give a sourced answer to this. A draft answer was produced but its "
    "citations did not verify against the retrieved catalog passages, so it was withheld."
)
NO_CONTEXT_MESSAGE = (
    "No catalog passages were retrieved for this question, so there is nothing to answer from."
)


class AnswerEngine:
    """Un motor por proceso: reutiliza el Retriever, que es caro de construir."""

    def __init__(self, retriever: Retriever, chain: ProviderChain, config: Config) -> None:
        self.retriever = retriever
        self.chain = chain
        self.config = config
        self._top_k: int = config.get("generation.context.top_k")
        self._method: str = config.get("retrieval.method")
        self._refuse_without_citations: bool = config.get(
            "generation.citations.refuse_without_citations"
        )
        self._require_inline: bool = config.get(
            "generation.citations.require_inline_refs_verified"
        )

    @classmethod
    def build(
        cls, config: Config, *, provider: str = "chain", with_dense: bool = True
    ) -> AnswerEngine:
        retriever = Retriever.build(config, with_dense=with_dense)
        return cls(retriever, build_chain(config, provider=provider), config)

    # ----------------------------------------------------------------- publico
    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
        method: str | None = None,
    ) -> VerifiedAnswer:
        timings = StageTimings()
        top_k = self._top_k if top_k is None else top_k

        hits = self.retriever.search(
            question,
            top_k=top_k,
            filters=filters,
            method=method or self._method,
            timings=timings,
        )
        records = [self.retriever.records[hit.control_id] for hit in hits]
        context = self.build_context(records)

        with stage(timings, "generation.provider"):
            completion, provider_info = self.chain.generate(question, context)

        with stage(timings, "generation.verify"):
            verification = verify_citations(completion.draft, context, self.config)

        cost = compute_cost(self.config, provider_info.model, completion.usage)
        answer = self._apply_policy(
            question=question,
            draft=completion.draft,
            verification=verification,
            context=context,
            retrieved=[hit.to_dict() for hit in hits],
            provider=provider_info,
            timings=timings,
            usage=completion.usage,
            cost=cost,
        )
        log_event(
            "generation.answered",
            refused=answer.refused,
            refusal_reason=answer.refusal_reason,
            citations_emitted=verification.emitted,
            citations_verified=len(verification.verified),
            forced_refusal=verification.forced_refusal,
            provider=provider_info.name,
            model=provider_info.model,
            input_tokens=completion.usage.get("input_tokens"),
            output_tokens=completion.usage.get("output_tokens"),
            cost_usd=cost.usd,
            timings_ms=timings.as_dict(),
        )
        return answer

    def build_context(self, records: list[ControlRecord]) -> AnswerContext:
        return build_context(records, self.retriever.control_ids(), self.config)

    # ---------------------------------------------------------------- politica
    def _apply_policy(
        self,
        *,
        question: str,
        draft: AnswerDraft,
        verification: Verification,
        context: AnswerContext,
        retrieved: list[dict[str, Any]],
        provider: ProviderInfo,
        timings: StageTimings,
        usage: dict[str, int],
        cost: Cost,
    ) -> VerifiedAnswer:
        def build(refused: bool, reason: str | None, text: str, citations: list) -> VerifiedAnswer:
            return VerifiedAnswer(
                question=question,
                refused=refused,
                refusal_reason=reason,
                answer=text,
                citations=citations,
                verification=verification,
                retrieved=retrieved,
                provider=provider,
                timings=timings.as_dict(),
                trace_id=current_trace_id(),
                usage=usage,
                cost_usd=cost.usd,
            )

        if draft.refused:
            # Un rehuso no lleva citas aunque el modelo adjunte alguna: si no se
            # afirma nada, no hay nada que respaldar.
            reason = draft.refusal_reason or (
                "no_relevant_control" if context.entries else "not_in_corpus"
            )
            return build(True, reason, draft.answer or NO_CONTEXT_MESSAGE, [])

        verified = verification.verified
        unsupported = verification.unsupported_inline_refs if self._require_inline else []

        if (self._refuse_without_citations and not verified) or unsupported:
            verification.forced_refusal = True
            return build(True, "unsupported_by_context", UNSUPPORTED_MESSAGE, [])

        return build(False, None, draft.answer, verified)
