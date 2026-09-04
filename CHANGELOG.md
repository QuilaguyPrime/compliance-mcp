# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Este proyecto sigue [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-04

Primera version etiquetada. El sistema esta completo y servible; su capa de
recuperacion esta medida y la de generacion no. Esa asimetria es la razon de que
esto sea 0.2.0 y no 1.0.0: la cadena de proveedores se sirve pero nunca se ha
evaluado, y una version mayor afirmaria un grado de verificacion que no existe.

### Que hace

Servidor MCP sobre el catalogo NIST SP 800-53 Rev 5 con tres herramientas:
`search_controls` (recuperacion pura), `get_control` (un registro literal del
catalogo) y `answer_question` (respuesta redactada en la que toda cita ha sido
verificada contra el texto exacto que se mostro al modelo). Recuperacion hibrida
BM25 + densa con fusion RRF ponderada, roll-up al control padre y filtros de
metadatos. Salida validada por esquema, y politica de rehuso: lo que no verifica
no se sirve.

### Medido

Ablacion de recuperacion sobre el split de test, 22 casos respondibles de 30,
con IC95 por bootstrap en cada punto. Configuracion servida, estrategia C con
recuperacion hibrida:

| metrica | valor | IC95 |
| --- | --- | --- |
| recall@1 | 0.3636 | [0.1818, 0.5909] |
| recall@5 | 0.8636 | [0.7273, 1.0000] |
| MRR | 0.5780 | [0.4326, 0.7235] |
| nDCG@10 | 0.4641 | [0.3371, 0.5976] |

La alternativa mas cercana, estrategia A, saca recall@5 0.8182 y gana en
recall@1 (0.4091) y MRR (0.6088). Con n=22 cada diferencia es de un solo caso y
los intervalos se solapan casi por completo: la eleccion de C descansa en que
`top_k = 5` mete los cinco controles enteros en el contexto, no en que los datos
la demuestren.

El efecto aislado del roll-up al padre es la unica comparacion no marginal de la
ablacion: sobre la estrategia C mueve recall@5 de 0.7273 a 0.8636, tres casos.

Todas las cifras salen de `data/derived/ablation.json`, con procedencia
verificable.

### Sin medir

La cadena real de proveedores nunca se ha ejecutado contra el golden set. No
existen cifras de fidelidad de citas en bruto, recall de rehuso, tasa de rehuso
falso, coste por consulta ni latencia de punta a punta. El arnes que las
produciria esta implementado y solo se ha corrido contra el baseline extractivo,
cuya precision de citacion es 1.0 por construccion y no por merito, razon por la
que nunca se inyecta como evidencia en el gate.

La verificacion de citas esta implementada y cubierta por tests unitarios contra
entradas construidas, pero nunca ejercitada sobre la cadena completa. Que una
respuesta servida solo lleve citas verificadas es cierto por construccion del
codigo; no es una afirmacion del mismo tipo que `recall@5 = 0.8636`.

Es una decision de alcance, no trabajo aplazado.

### Procedencia

Todo artefacto de evaluacion lleva cuatro digests que el gate de CI compara con
el arbol: corpus ingerido, secciones de configuracion que mueven los numeros
(incluidas las semillas de split y bootstrap), golden set y codigo de `src/`.
El digest de codigo normaliza finales de linea y ordena por ruta POSIX, de modo
que un clon en Windows o en Linux produce el mismo valor.

Los generadores de artefactos se niegan a producir numeros desde un arbol con
cambios sin commitear, y se niegan tambien donde no hay commit resoluble, que es
el caso dentro de la imagen. `--allow-dirty` es la unica valvula, y marca el
artefacto de forma que el gate lo rechaza.

El indice guarda el fingerprint del texto exacto que se embebio y lo recomprueba
al cargarlo: un indice caducado da `StaleIndexError` en vez de resultados peores
en silencio.

### Empaquetado

Imagen multi-etapa que ingesta el catalogo, construye el indice y hornea el
modelo de embeddings, de modo que arranca sin red ni paso previo. Plataforma
objetivo `linux/amd64`, declarada en el compose: 1.99 GB, 12 capas. Usuario
no-root. Un job de CI construye la imagen sin ninguna clave de API, exige que el
preflight falle solo por la ausencia de credenciales y ejecuta una consulta real
con `--network none`.

La imagen sirve, no evalua: no lleva `git`, asi que un artefacto producido
dentro no podria atarse a ningun commit, y el guardia de procedencia lo impide.

### Limitaciones conocidas

- La cadena de generacion se sirve sin evaluar (ver arriba).
- Dos de los tres umbrales del gate, `min_citation_precision` y
  `max_hallucinated_citation_rate`, nunca han evaluado nada: leen un bloque que
  solo produce una corrida real de generacion. Estan puestos y aplicados, y no
  han tenido ocasion de fallar.
- Las versiones de dependencias no entran en la procedencia. `pyproject.toml`
  declara rangos y no versiones fijas, asi que hashearlo afirmaria mas de lo que
  puede sostener. El instrumento correcto es un lockfile y este repo no tiene
  ninguno: ver [#1](https://github.com/QuilaguyPrime/compliance-mcp/issues/1).
- El campo `must_not_invent` del golden set esta escrito en prosa para un humano
  y no se comprueba automaticamente.
- El bus de eventos y el indice viven en memoria de un solo proceso.
- Con n=22 casos respondibles, casi ninguna comparacion entre configuraciones es
  concluyente. Los intervalos estan publicados para que se vea.
- `README.es.md` es una traduccion que va por detras del README en ingles.

[0.2.0]: https://github.com/QuilaguyPrime/compliance-mcp/releases/tag/v0.2.0
