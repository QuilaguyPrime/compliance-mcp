"""Servidor MCP sobre el catalogo NIST SP 800-53 Rev 5.

Tres herramientas, en orden de compromiso creciente con lo que dicen:

* `search_controls` — recuperacion pura. Devuelve controles, no prosa.
* `get_control`     — lectura literal de un registro del catalogo. Sin modelo.
* `answer_question` — recuperacion + generacion. Toda cita que sale de aqui ha
                      sido verificada contra el texto que se le mostro al
                      modelo; lo que no verifica no se sirve.

El indice se construye UNA vez al arrancar y se reutiliza. Reconstruirlo por
peticion implicaria re-embeber el corpus entero en cada llamada.

Los logs van a stderr (ver observability.configure_logging) porque el
transporte stdio usa stdout para el protocolo: un print de mas rompe la sesion.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from . import __version__
from .config import Config, load_config
from .generation.engine import AnswerEngine
from .generation.providers import ProviderError
from .observability import configure_logging, log_event, trace_context
from .retrieval.search import SearchFilters
from .validation import ToolInputError, ToolInputValidator

SERVER_NAME = "compliance-mcp"
INSTRUCTIONS = """Consulta el catalogo NIST SP 800-53 Rev 5.

Usa `search_controls` para localizar controles, `get_control` para leer uno
entero y `answer_question` cuando quieras una respuesta redactada: esta ultima
solo afirma lo que puede citar literalmente del catalogo, y rehusa cuando el
catalogo no lo soporta. Las respuestas incluyen sus citas verificadas."""


def _tool_error(message: str) -> Exception:
    """Error deliberado de una herramienta.

    mcp distingue el error que una herramienta lanza a proposito (su mensaje
    llega a quien llama) del que se le escapa (mensaje generico, se queda en el
    servidor). Un argumento invalido es lo primero: el mensaje ES la respuesta
    util.
    """
    from mcp.server.mcpserver.exceptions import ToolError

    return ToolError(message)


def build_server(config: Config, *, engine: AnswerEngine | None = None) -> Any:
    """Monta el servidor. `engine` se inyecta en los tests para no cargar el
    indice denso ni una cadena de proveedores con red."""
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:  # pragma: no cover - depende del extra `serve`
        raise RuntimeError(
            "Falta el paquete `mcp`. Instala el extra: pip install -e '.[serve]'"
        ) from exc

    engine = engine or AnswerEngine.build(config)
    retriever = engine.retriever
    validate = ToolInputValidator(retriever.records, config)
    max_top_k: int = config.get("retrieval.max_top_k")
    default_top_k: int = config.get("retrieval.top_k")
    default_include_withdrawn: bool = config.get("retrieval.default_include_withdrawn")

    server = MCPServer(
        name=SERVER_NAME,
        instructions=INSTRUCTIONS,
        version=__version__,
    )

    @server.tool(
        description=(
            "Busca controles del catalogo NIST SP 800-53 Rev 5 por lenguaje natural o "
            "por jerga NIST. Devuelve controles ordenados por relevancia, con un "
            "extracto de cada uno. No redacta respuestas: para eso usa answer_question."
        )
    )
    def search_controls(
        query: str,
        top_k: int | None = None,
        family: str | None = None,
        baseline: str | None = None,
        kind: str | None = None,
        include_withdrawn: bool | None = None,
    ) -> dict[str, Any]:
        """Busca controles.

        Args:
            query: Pregunta o terminos de busqueda.
            top_k: Cuantos controles devolver (1..max_top_k de config).
            family: Filtra por familia, p.ej. "ac", "au", "ir".
            baseline: Filtra por baseline: "low", "moderate", "high" o "privacy".
            kind: "control" para controles base, "enhancement" para mejoras.
            include_withdrawn: Incluye controles retirados. Por defecto no.
        """
        with trace_context() as trace_id:
            # Un filtro mal escrito devolveria cero resultados y quien llama
            # concluiria que el catalogo no cubre el tema. Se falla diciendo que
            # valores existen, que es recuperable en el mismo turno.
            try:
                query = validate.query(query)
                filters = SearchFilters(
                    family=validate.family(family),
                    baseline=validate.baseline(baseline),
                    kind=validate.kind(kind),
                    include_withdrawn=(
                        default_include_withdrawn
                        if include_withdrawn is None
                        else include_withdrawn
                    ),
                )
                top_k = validate.top_k(top_k)
            except ToolInputError as exc:
                log_event("tool.search_controls.rejected", error=str(exc))
                raise _tool_error(str(exc)) from exc
            hits = retriever.search(
                query,
                top_k=default_top_k if top_k is None else top_k,
                filters=filters,
                method=config.get("retrieval.method"),
            )
            log_event("tool.search_controls", query=query, n_hits=len(hits))
            return {
                "query": query,
                "n_hits": len(hits),
                "hits": [hit.to_dict() for hit in hits],
                "trace_id": trace_id,
            }

    @server.tool(
        description=(
            "Devuelve un control completo del catalogo por su id (p.ej. 'ac-2' o "
            "'AC-2(1)'), con statement, guidance, contenido de evaluacion, controles "
            "relacionados, baselines y referencias. Incluye los retirados, con su "
            "control de destino."
        )
    )
    def get_control(control_id: str) -> dict[str, Any]:
        """Lee un control entero.

        Args:
            control_id: Id del control, en cualquiera de sus formas: "ac-2",
                "AC-2", "AC-2(1)" o "ac-2.1".
        """
        from .generation.schema import normalize_control_id

        with trace_context() as trace_id:
            try:
                control_id = validate.query(control_id, field="control_id")
            except ToolInputError as exc:
                raise _tool_error(str(exc)) from exc
            canonical = normalize_control_id(control_id)
            record = retriever.records.get(canonical)
            log_event("tool.get_control", control_id=canonical, found=record is not None)
            if record is None:
                return {
                    "found": False,
                    "control_id": canonical,
                    "error": (
                        f"El control '{control_id}' no existe en el catalogo. "
                        f"Usa search_controls para localizarlo."
                    ),
                    "trace_id": trace_id,
                }
            payload = record.to_dict()
            payload["enhancements"] = [
                {"control_id": eid, "label": retriever.records[eid].label,
                 "title": retriever.records[eid].title}
                for eid in record.enhancement_ids
                if eid in retriever.records
            ]
            return {"found": True, "control": payload, "trace_id": trace_id}

    @server.tool(
        description=(
            "Responde una pregunta sobre NIST SP 800-53 Rev 5 usando solo el catalogo. "
            "Cada afirmacion va respaldada por una cita literal verificada contra el "
            "texto del control; si nada queda verificado, rehusa en vez de responder. "
            "Devuelve tambien los controles recuperados y el detalle de la verificacion."
        )
    )
    def answer_question(
        question: str,
        top_k: int | None = None,
        family: str | None = None,
        baseline: str | None = None,
    ) -> dict[str, Any]:
        """Responde con citas verificadas.

        Args:
            question: La pregunta, en lenguaje natural.
            top_k: Cuantos controles pasarle al generador como contexto.
            family: Restringe el contexto a una familia, p.ej. "ir".
            baseline: Restringe el contexto a un baseline: "low", "moderate",
                "high" o "privacy".
        """
        with trace_context():
            try:
                question = validate.query(question, field="question")
                top_k = validate.top_k(top_k)
                family = validate.family(family)
                baseline = validate.baseline(baseline)
            except ToolInputError as exc:
                log_event("tool.answer_question.rejected", error=str(exc))
                raise _tool_error(str(exc)) from exc
            filters = None
            if family or baseline:
                filters = SearchFilters(
                    family=family,
                    baseline=baseline,
                    include_withdrawn=default_include_withdrawn,
                )
            try:
                answer = engine.answer(question, top_k=top_k, filters=filters)
            except ProviderError as exc:
                # Que ningun proveedor pueda responder no es un fallo interno del
                # servidor: es una condicion que quien llama puede entender y
                # arreglar, casi siempre una credencial que falta. Sin esto el
                # mensaje se queda en el servidor y el cliente ve un error
                # generico. search_controls y get_control siguen funcionando.
                log_event("tool.answer_question.no_provider", error=str(exc))
                raise _tool_error(
                    "No hay ningun proveedor de generacion disponible, asi que "
                    f"answer_question no puede responder: {exc}. Las herramientas "
                    "search_controls y get_control no necesitan proveedor."
                ) from exc
            return answer.to_dict()

    log_event(
        "server.ready",
        tools=["search_controls", "get_control", "answer_question"],
        controls=len(retriever.records),
        chunks=len(retriever.chunks),
        strategy=config.get("chunking.active"),
        max_top_k=max_top_k,
    )
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=SERVER_NAME)
    parser.add_argument("--config", default=None)
    parser.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(config)
    build_server(config).run(args.transport)
    return 0


if __name__ == "__main__":
    sys.exit(main())
