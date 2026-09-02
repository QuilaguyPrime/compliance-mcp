"""Retriever lexico BM25.

El tokenizador es UNO SOLO y se aplica igual a documentos y consultas. En la
version previa del repo el corpus se tokenizaba con una regex y la consulta con
`str.split()`; ese desajuste hundio recall@5 de 0.711 a 0.111 y hacia parecer
que BM25 era inutil cuando el bug era la tokenizacion.
"""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from ..config import Config


class BM25Retriever:
    def __init__(self, texts: list[str], config: Config) -> None:
        params = config.section("retrieval.bm25")
        self._pattern = re.compile(params["token_pattern"])
        self._stopwords = set(params["stopwords"])
        self._index = BM25Okapi(
            [self.tokenize(t) for t in texts], k1=params["k1"], b=params["b"]
        )

    def tokenize(self, text: str) -> list[str]:
        return [t for t in self._pattern.findall((text or "").lower()) if t not in self._stopwords]

    def rank(self, query: str, pool: int) -> list[int]:
        """Indices de chunk ordenados por score BM25 descendente."""
        scores = self._index.get_scores(self.tokenize(query))
        return np.argsort(-scores)[:pool].tolist()
