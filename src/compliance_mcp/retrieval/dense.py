"""Retriever denso por embeddings.

Se usa producto punto sobre vectores normalizados (equivale a coseno). No hay
FAISS: con 2210 chunks x 768 dimensiones el indice cabe en 6.5 MB y una busqueda
exhaustiva en numpy tarda menos de un milisegundo. Meter un ANN aqui seria
complejidad sin beneficio medible, y ademas introduciria perdida de recall.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Config


class DenseRetriever:
    def __init__(self, embeddings: np.ndarray, config: Config) -> None:
        self._embeddings = embeddings
        self._config = config
        self._prefix = config.get("retrieval.dense.query_prefix")
        self._model = None  # cargado de forma perezosa: consultar no siempre hace falta

    @staticmethod
    def load_model(config: Config):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(config.get("retrieval.dense.model"))

    @classmethod
    def encode_corpus(cls, texts: list[str], config: Config) -> np.ndarray:
        model = cls.load_model(config)
        return model.encode(
            texts,
            batch_size=config.get("retrieval.dense.batch_size"),
            normalize_embeddings=config.get("retrieval.dense.normalize"),
            show_progress_bar=False,
        ).astype("float32")

    @classmethod
    def from_file(cls, path: Path, config: Config) -> DenseRetriever:
        if not path.exists():
            raise FileNotFoundError(f"Faltan los embeddings en {path}. Ejecuta `make index`.")
        return cls(np.load(path), config)

    def encode_query(self, query: str) -> np.ndarray:
        if self._model is None:
            self._model = self.load_model(self._config)
        return self._model.encode(
            [self._prefix + query],
            normalize_embeddings=self._config.get("retrieval.dense.normalize"),
            show_progress_bar=False,
        ).astype("float32")[0]

    def rank(self, query: str, pool: int) -> list[int]:
        scores = self._embeddings @ self.encode_query(query)
        return np.argsort(-scores)[:pool].tolist()
