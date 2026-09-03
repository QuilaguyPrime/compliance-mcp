# compliance-mcp

*[Read in English](README.md)*

> El README en ingles es el canonico y va por delante: incluye el diagrama
> de arquitectura y la tabla de ablacion con la estrategia de chunking. Esta
> version puede quedarse atras.

Servidor MCP sobre el catalogo NIST SP 800-53 Rev 5 con recuperacion hibrida,
citaciones verificadas contra el corpus y salida validada por esquema.

> **Estado: en construccion (fase 4 de 5).** El README definitivo, con la tabla
> de ablacion, el coste por consulta y las latencias medidas, se escribe en la
> fase 5. Hasta entonces este fichero no contiene ningun numero: los numeros
> solo se publican cuando salen de una corrida real.

## Que hace

Tres herramientas MCP, en orden de compromiso creciente con lo que dicen:

| Herramienta | Que devuelve |
| --- | --- |
| `search_controls` | Controles ordenados por relevancia. Recuperacion pura, sin modelo generativo. Filtra por familia, baseline, tipo y retirados. |
| `get_control` | Un registro del catalogo entero y literal: statement, guidance, contenido de evaluacion SP 800-53A, relacionados, baselines y referencias. Incluye los retirados con su control de destino. |
| `answer_question` | Una respuesta redactada en la que **toda cita ha sido verificada**, mas los controles recuperados y el detalle de la verificacion. |

## La regla de la casa: lo que no verifica, no se sirve

Una citacion es `(control_id, part, quote)`, y `quote` tiene que aparecer
literalmente en el texto **que se le mostro al modelo** — no en el corpus
entero. La diferencia importa: si un modelo reproduce palabra por palabra un
trozo real de AC-2 que nunca se le enseno, eso no es una cita, es memoria
parametrica que resulto acertar.

Cada cita recibe un veredicto: verificada, control inexistente, control real
que no estaba en el contexto, cita que no aparece en el control, parte
inexistente o cita demasiado corta para anclar nada. Lo que no verifica se
descarta; si una respuesta se queda sin ninguna cita verificada, o afirma cosas
sobre un control que no puede respaldar, se convierte en rehuso. Rehusar es un
resultado valido del sistema, no un fallo suyo.

Por eso la tasa de citas alucinadas **servidas** es cero por construccion. Como
eso convertiria el gate de CI en un tramite, la evaluacion mide aparte la tasa
**en bruto**, sobre lo que el modelo emitio antes de que la politica descartara
nada, y es esa la que el gate hace cumplir.

## De donde sale cada numero, y como se sabe que no esta caducado

El nombre del fichero de embeddings solo codifica estrategia y modelo. Cambiar
`ingest.param_resolution_passes` o una plantilla de chunking y volver a hacer
ingest cambia el TEXTO de los chunks pero no su NUMERO: el `.npy` anterior sigue
cargando, las formas siguen cuadrando y cada vector pasa a corresponder a un
texto distinto del que dice. No hay excepcion ni aviso, y la evaluacion publica
numeros de un indice que no es el que se sirve.

Contra eso, tres cierres:

* **Manifiesto de indice.** Al construirlo se guarda el fingerprint del texto
  exacto que se embebio. Al cargarlo se recomputa y se compara: si no cuadra,
  `StaleIndexError` y no se sirve nada. Fingerprint de contenido, no de nombre
  de fichero.
* **Procedencia en los resultados.** Toda salida de evaluacion lleva commit,
  digest del corpus y digest de la configuracion que la produjo. El gate de CI
  compara esa procedencia con el arbol actual y rechaza resultados que vengan de
  otro corpus u otra config: un `ablation.json` viejo commiteado ya no pasa el
  gate sin haber medido nada.
* **Preflight** (`make doctor`). Comprueba corpus, frescura del indice,
  coherencia del golden set, extras instalados y presencia de credenciales, sin
  llamar a ninguna API. `--require` acota que comprobaciones deciden el codigo de
  salida, para que cada job de CI exija exactamente lo que va a usar.

## Como se ejecuta

```bash
make install-serve          # nucleo + indice denso + proveedores
make ingest index           # corpus OSCAL -> registros -> indice hibrido
cp .env.example .env        # y rellenar las claves
make doctor                 # preflight, sin red
make serve                  # servidor MCP por stdio (pasa el preflight antes)
```

Registrado en un cliente MCP:

```json
{
  "mcpServers": {
    "compliance": {
      "command": "/ruta/al/repo/.venv/bin/python",
      "args": ["-m", "compliance_mcp.server"],
      "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }
    }
  }
}
```

El indice se construye una vez al arrancar y se reutiliza entre peticiones. Los
logs salen por stderr en JSON, con un `trace_id` por peticion que atraviesa
recuperacion, generacion y verificacion; stdout queda para el protocolo.

## Como se evalua

```bash
make eval                              # ablacion de recuperacion (split de test)
make eval-generation                   # generacion con la cadena real (gasta API)
PROVIDER=extractive make eval-generation   # el suelo: baseline sin red
```

El golden set son 60 casos escritos a mano: 30 respondibles estratificados por
estilo de consulta, 15 en los que rehusar es la unica respuesta correcta y 15
adversariales. El split train/test es determinista por hash del id del caso;
los hiperparametros se ajustan solo en train y se reporta test.

La evaluacion de generacion mide tres cosas que conviene no mezclar: fidelidad
de las citas en bruto, comportamiento de rehuso (recall de rehuso **y** tasa de
rehuso falso, que hay que leer juntos porque rehusar siempre saca 1.0 en el
primero) y cifras afirmadas sin fuente. El baseline extractivo copia, asi que su
precision de citacion es 1.0 por construccion y no por merito: nunca se inyecta
como evidencia en el gate.

El campo `must_not_invent` del golden set esta escrito como criterio en prosa
para un humano, no como cadena comparable. No se comprueba automaticamente:
esos casos se vuelcan en el bloque `manual_review` de
`data/derived/generation.json` para adjudicacion humana.

El coste por consulta se calcula con los tokens que reporta el proveedor y con
los precios de `config.yaml`, que llevan fecha de contraste. Un modelo sin
precio declarado da coste `None`, no cero: cero es un numero y se acaba sumando
a un total que parece medido.

### Que hace cumplir de verdad el gate de CI

`min_recall_at_5` esta en 0.60 frente a un 0.8636 medido, y la holgura es
deliberada. Con n=22 un solo caso vale 0.0455 y el IC95 va de 0.7273 a 1.0000,
asi que un umbral cenido a la medicion fallaria por ruido de muestreo cada vez
que un caso cambiase de lado, y un gate que falla sin que nada se haya roto
acaba desactivado. En 0.60 detecta regresion catastrofica -un indice caducado,
una fusion mal ponderada, un corpus que no cuadra-, no calidad marginal. Para lo
marginal estan los intervalos de confianza de `ablation.json`.

El gate exigente es `min_citation_precision: 0.95` con
`max_hallucinated_citation_rate: 0.0`, aplicados sobre la salida en bruto del
modelo antes de que la politica descarte nada. Ninguno se ha ejecutado nunca:
los dos dependen de la evaluacion de generacion, que gasta API y sigue
pendiente. Un gate que hoy no puede fallar es informacion, no verguenza: dice
con precision que parte del sistema esta medida y cual no.

### Lo que todavia no esta medido

La cadena real de proveedores no se ha ejecutado nunca contra el golden set, asi
que **no hay cifras de fidelidad de citas, de rehuso ni de coste del sistema que
se sirve**. Lo unico medido de punta a punta es la recuperacion y el baseline
extractivo. Requieren gasto de API y quedan para la fase 5: correr la evaluacion
de generacion, adjudicar a mano el bloque `manual_review` y escribir el README
definitivo con sus intervalos de confianza.

## Configuracion

`config.yaml` es la fuente unica de verdad. Ningun literal numerico ni nombre de
modelo vive en el codigo, y el cargador falla ruidosamente si falta una clave en
vez de caer a un valor por defecto silencioso: un fallback silencioso convierte
un error de configuracion en un resultado de evaluacion equivocado.

## Licencia

MIT. Copyright (c) 2026 Juan Camilo Amaya Quilaguy. El texto completo esta en
[LICENSE](LICENSE).

Los catalogos de `data/raw/` son publicaciones del NIST (SP 800-53 Rev. 5, en
formato OSCAL) y son de dominio publico; la licencia MIT cubre este codigo, no
esos datos de origen.
