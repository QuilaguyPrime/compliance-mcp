# syntax=docker/dockerfile:1.7

# =============================================================== builder ===
# Instala dependencias, ingesta el catalogo OSCAL y construye el indice. Nada
# de esta etapa llega al runtime salvo lo que se copia explicitamente.
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf

# /app no es arbitrario. `config.project_root()` es `parents[2]` del fichero
# config.py, o sea que la raiz del proyecto se deduce de DONDE vive el codigo.
# La etapa de runtime tiene que dejar el arbol en esta misma ruta o config.yaml
# y todo lo de data/ dejan de resolverse.
WORKDIR /app

RUN python -m venv /opt/venv

# torch desde el indice CPU de PyTorch, y antes que nada para que quede en su
# propia capa. Desde PyPI, `pip install torch` en linux arrastra las ruedas
# nvidia-* de CUDA: cientos de MB que un proceso que solo codifica consultas
# cortas no usa jamas.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md ./
COPY src ./src

# Editable, y no por comodidad de desarrollo. Una instalacion normal deja
# config.py dentro de site-packages, con lo que project_root() pasa a apuntar
# al interior del venv y load_config() no encuentra config.yaml. Comprobado.
RUN pip install -e ".[dense,serve]"

COPY config.yaml ./
COPY data/raw ./data/raw
COPY data/golden ./data/golden

# El modelo de embeddings se descarga aqui, como efecto de construir el indice,
# y queda en HF_HOME. Hace falta tambien EN RUNTIME: `DenseRetriever.encode_query`
# carga el SentenceTransformer para codificar cada consulta. Sin el en la imagen,
# la primera pregunta intentaria salir a la red.
#
# `--all` y no solo la estrategia servida: `make doctor` sin argumentos exige
# ablation_index, que comprueba A, B y C. Son 13 MB contra los cientos de MB de
# torch y del modelo; no es aqui donde se ahorra.
RUN python -m compliance_mcp.build_index ingest \
 && python -m compliance_mcp.build_index index --all \
 && rm -rf /app/data/raw

# =============================================================== runtime ===
# Solo lo necesario para servir: venv, modelo, codigo, config, corpus ingerido
# e indice. Sin catalogo OSCAL crudo, sin tests, sin toolchain de compilacion.
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# make pesa poco mas de 1 MB y hace que `make doctor` y `make serve` signifiquen
# dentro del contenedor exactamente lo mismo que en el README.
RUN apt-get update \
 && apt-get install --no-install-recommends -y make \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
# El modelo tiene que ser escribible por appuser: huggingface_hub coge locks
# dentro de HF_HOME incluso en modo offline.
COPY --from=builder --chown=appuser:appuser /opt/hf /opt/hf
COPY --from=builder --chown=appuser:appuser /app /app
COPY --chown=appuser:appuser Makefile LICENSE ./

# El Makefile invoca .venv/bin/python. Un enlace evita duplicar el venv y deja
# una sola forma de llamar a las cosas dentro y fuera del contenedor.
RUN ln -s /opt/venv /app/.venv

USER appuser

# ENTRYPOINT es el interprete y CMD el modulo, para que `docker run <img> -m
# compliance_mcp.doctor` funcione sin reescribir el entrypoint.
ENTRYPOINT ["/opt/venv/bin/python"]
CMD ["-m", "compliance_mcp.server"]
