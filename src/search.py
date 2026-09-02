"""Hybrid retrieval: dense vectors and BM25, fused with reciprocal rank fusion.

Hybrid is not optional here. The vault is dense with exact tokens - reg plates,
boiler model numbers, policy references, postcodes - where dense retrieval
underperforms and BM25 is decisive.
"""

from __future__ import annotations

import numpy as np

from .embedder import Embedder
from .index import VaultIndex, tokenize

RRF_K = 60
CANDIDATES = 50

# BM25 is down-weighted relative to dense. Measured on this vault: at equal
# weight the lexical list pulls topically-wrong chunks to position one when a
# query happens to share a common term with an unrelated note ("renew" matches
# certbot renewal as readily as insurance renewal).
SPARSE_WEIGHT = 0.5

# If the query's terms appear in no more than this many chunks corpus-wide, the
# query is a lookup rather than a question - a reg plate, a licence key, a part
# code - and the lexical hit is the answer. See _is_lookup.
LOOKUP_MAX_MATCHES = 5


def _top_indices(scores: np.ndarray, limit: int) -> list[int]:
    if scores.size == 0:
        return []
    limit = min(limit, scores.size)
    # argpartition is O(n) - the full sort only touches the candidate window.
    partitioned = np.argpartition(-scores, limit - 1)[:limit]
    return partitioned[np.argsort(-scores[partitioned])].tolist()


def fuse(dense: list[int], sparse: list[int]) -> list[tuple[int, float]]:
    """Weighted reciprocal rank fusion.

    score(d) = w / (RRF_K + rank(d)), summed over the lists containing d.
    """
    scores: dict[int, float] = {}
    for weight, ranking in ((1.0, dense), (SPARSE_WEIGHT, sparse)):
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + weight / (RRF_K + rank)
    return sorted(scores.items(), key=lambda item: -item[1])


def _is_lookup(lexical_matches: int) -> bool:
    """True when the query's terms are near-unique in the corpus.

    Rank fusion alone cannot handle this case. A document found by only one of
    the two rankers scores 1/(RRF_K+1) whichever ranker found it, so BM25's
    correct rank-one hit for an exact identifier ties with the dense list's
    rank-one hit and loses the tie-break. Measured on this vault, dense
    retrieval scores 0/40 on exact identifiers where BM25 scores 40/40, so the
    tie must be resolved in favour of the lexical hit - but only when the match
    is genuinely near-unique, or the same rule would promote an incidental
    keyword hit over a correct semantic one.
    """
    return 0 < lexical_matches <= LOOKUP_MAX_MATCHES


async def search(index: VaultIndex, embedder: Embedder, query: str, k: int) -> list[dict]:
    if index.size == 0:
        return []

    query_vector = await embedder.embed_query(query)
    dense_ranking = _top_indices(index.matrix @ query_vector, CANDIDATES)

    sparse_ranking: list[int] = []
    lexical_matches = 0
    if index.bm25 is not None:
        tokens = tokenize(query)
        if tokens:
            sparse_scores = np.asarray(index.bm25.get_scores(tokens))
            lexical_matches = int((sparse_scores > 0).sum())
            # Drop non-matching candidates. argpartition returns a full window
            # regardless of score, so without this a query matching nothing
            # lexically contributes 50 arbitrary votes to the fusion.
            sparse_ranking = [
                i for i in _top_indices(sparse_scores, CANDIDATES) if sparse_scores[i] > 0
            ]

    ranked = fuse(dense_ranking, sparse_ranking)
    if sparse_ranking and _is_lookup(lexical_matches):
        top = sparse_ranking[0]
        ranked = [(top, ranked[0][1] if ranked else 1.0)] + [r for r in ranked if r[0] != top]

    results: list[dict] = []
    for doc, score in ranked[:k]:
        chunk = index.chunks[doc]
        results.append(
            {
                "path": chunk["path"],
                "title": chunk["title"],
                "breadcrumb": chunk["breadcrumb"],
                "line": chunk["line"],
                "score": round(score, 5),
                "text": chunk["text"],
            }
        )
    return results


def format_results(query: str, results: list[dict]) -> str:
    """Markdown, with the source path as a heading above each chunk, so the
    model can cite it without a second call."""
    if not results:
        return f'No vault matches for "{query}".'

    blocks = [f'{len(results)} result(s) for "{query}":\n']
    for position, result in enumerate(results, start=1):
        location = result["path"]
        if result["breadcrumb"]:
            location += f" > {result['breadcrumb']}"
        blocks.append(
            f"### {position}. {location}\n"
            f"*score {result['score']} - line {result['line']}*\n\n"
            f"{result['text'].strip()}\n"
        )
    return "\n".join(blocks)
