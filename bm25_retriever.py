from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[^\W_]", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class RetrievalHit:
    doc_id: str
    score: float
    rank: int


class BM25:
    """Small deterministic BM25 implementation for one closed candidate pool."""

    def __init__(
        self,
        documents: Sequence[Tuple[str, str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("BM25 requires at least one document")
        self.k1 = k1
        self.b = b
        self.doc_ids = [doc_id for doc_id, _ in documents]
        self.tokens = [tokenize(text) for _, text in documents]
        self.doc_lengths = [len(tokens) for tokens in self.tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        self.idf = {
            term: math.log(1.0 + (len(self.doc_ids) - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def score(self, query: str, index: int) -> float:
        query_terms = tokenize(query)
        frequencies = self.term_frequencies[index]
        length = self.doc_lengths[index]
        denominator_norm = 1.0 - self.b + self.b * length / max(self.avgdl, 1e-12)
        score = 0.0
        for term in query_terms:
            if term not in frequencies:
                continue
            tf = frequencies[term]
            numerator = self.idf.get(term, 0.0) * tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * denominator_norm
            score += numerator / denominator
        return score

    def search(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        if top_k <= 0:
            return []
        scored = [
            (self.score(query, index), self.doc_ids[index], index)
            for index in range(len(self.doc_ids))
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievalHit(doc_id=doc_id, score=score, rank=rank)
            for rank, (score, doc_id, _) in enumerate(scored[:top_k], start=1)
        ]

