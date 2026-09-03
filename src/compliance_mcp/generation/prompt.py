"""Construccion del prompt.

El prompt no es un parametro sino programa, asi que vive aqui y no en
config.yaml: cambiarlo cambia el comportamiento medido y debe pasar por
revision de codigo y por el arnes de evaluacion.

Tres decisiones deliberadas:

* El contexto se numera y se etiqueta con el `control_id` canonico, que es la
  clave exacta que debe devolver en `citations`. Pedirle que cite "AC-2(1)" y
  luego buscar "ac-2.1" es fabricarse un fallo de verificacion.
* Se le dice explicitamente que citar es copiar, no parafrasear. Una cita
  parafraseada no pasa la verificacion literal, asi que una respuesta correcta
  parafraseada acaba en rehuso: peor resultado que decirselo.
* Rehusar se presenta como resultado valido y con taxonomia, no como fallo. El
  golden set tiene 15 casos donde rehusar es la unica respuesta correcta.
"""
from __future__ import annotations

from .context import AnswerContext
from .schema import REFUSAL_REASONS

SYSTEM_PROMPT = """You answer questions about the NIST SP 800-53 Rev 5 control catalog.

You are given a numbered set of catalog passages. They are your ONLY source of
truth. You know nothing else: not other frameworks, not costs, not vendors, not
what changed between revisions, not vulnerability data.

Rules:

1. Answer only what the passages support. If they do not contain the answer,
   refuse.
2. Every claim you make must be backed by a citation. A citation is a
   VERBATIM span copied character-for-character from a passage, not a
   paraphrase and not a summary. Citations are checked against the passages by
   exact string match; a rewritten quote is rejected and takes the answer with
   it.
3. Use `control_id` exactly as given in the passage header (lowercase, e.g.
   "ac-2" or "ac-2.1"), and `part` exactly as given (e.g. "statement").
4. In the prose, refer to a control with its bracketed label, e.g. "[AC-2]".
   Only refer to controls you actually cite.
5. Parameters in square brackets like [Assignment: organization-defined
   frequency] are values the organization sets. Report them as
   organization-defined; never invent a number, a period, or a threshold.
6. Be concise. Two to five sentences is usually right.

Refuse by setting `refused: true`, an empty `citations` list, a
`refusal_reason` from the allowed set, and an `answer` that states plainly what
you cannot answer and why. Refusing when the passages do not support an answer
is the correct outcome, not a failure."""


def render_context(context: AnswerContext) -> str:
    blocks: list[str] = []
    for index, entry in enumerate(context.entries, start=1):
        header = (
            f"[{index}] control_id: {entry.control_id} | label: {entry.label} | "
            f"title: {entry.title} | family: {entry.family_title} | status: {entry.status}"
        )
        if entry.baselines:
            header += f" | baselines: {', '.join(entry.baselines)}"
        body = "\n".join(f"<{name}>\n{text}\n</{name}>" for name, text in entry.parts.items())
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def build_user_message(question: str, context: AnswerContext) -> str:
    reasons = ", ".join(REFUSAL_REASONS)
    if not context.entries:
        # Sin contexto no hay nada citable: se le pide el rehuso explicitamente
        # en vez de dejarle inventar desde el vacio.
        return (
            f"PASSAGES: (none retrieved)\n\nQUESTION: {question}\n\n"
            f"No passages were retrieved, so refuse. Allowed refusal_reason values: {reasons}."
        )
    return (
        f"PASSAGES:\n\n{render_context(context)}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer from the passages above, or refuse. "
        f"Allowed refusal_reason values: {reasons}."
    )
