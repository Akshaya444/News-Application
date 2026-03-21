from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RelevanceResult:
    index: int
    similarity: float


def _article_texts(titles: Sequence[str], descriptions: Sequence[str], contents: Sequence[str]) -> List[str]:
    texts: List[str] = []
    for t, d, c in zip(titles, descriptions, contents):
        parts = [t or "", d or "", c or ""]
        texts.append(" \n".join(p.strip() for p in parts if p and p.strip()))
    return texts


def relevance_scores(
    *,
    titles: Sequence[str],
    descriptions: Sequence[str],
    contents: Sequence[str],
    query: str,
) -> Tuple[np.ndarray, List[RelevanceResult]]:
    if not query.strip():
        return np.array([]), []

    texts = _article_texts(titles, descriptions, contents)
    if not texts:
        return np.array([]), []

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    docs = texts + [query]
    X = vectorizer.fit_transform(docs)
    articles_X = X[:-1]
    query_X = X[-1]

    sims = cosine_similarity(articles_X, query_X).ravel()
    results: List[RelevanceResult] = [
        RelevanceResult(index=int(i), similarity=float(sims[i]))
        for i in range(len(sims))
    ]
    results.sort(key=lambda r: r.similarity, reverse=True)
    return sims, results


def filter_by_query(
    *,
    titles: Sequence[str],
    descriptions: Sequence[str],
    contents: Sequence[str],
    query: str,
    min_similarity: float = 0.22,
    top_k_fallback: int = 8,
) -> List[int]:
    """
    Return indices of articles that are relevant to the query,
    based on cosine similarity with a TF-IDF model.
    """
    sims, _ = relevance_scores(
        titles=titles,
        descriptions=descriptions,
        contents=contents,
        query=query,
    )
    if sims.size == 0:
        return list(range(len(titles)))

    # Token-based filtering: keep only articles that contain all query tokens.
    # This prevents unrelated "Meta/Meta..." type results from slipping in.
    tokens = [t for t in re.split(r"[^a-z0-9]+", query.lower().strip()) if len(t) >= 2]
    if not tokens:
        # If query is too short, fall back to similarity ranking.
        order = np.argsort(-sims)[: min(top_k_fallback, len(sims))]
        return [int(i) for i in order]

    all_texts = [f"{titles[i]} {descriptions[i]} {contents[i]}".lower() for i in range(len(titles))]

    # First try strict match: all tokens must appear.
    candidate_idxs: List[int] = []
    for i in range(len(all_texts)):
        text = all_texts[i]
        if all(tok in text for tok in tokens):
            candidate_idxs.append(i)

    if candidate_idxs:
        ranked = sorted(candidate_idxs, key=lambda i: float(sims[i]), reverse=True)
        return [int(i) for i in ranked[: min(top_k_fallback, len(ranked))]]

    # If strict match returns nothing, try looser match: any token appears.
    any_candidate_idxs: List[int] = []
    for i in range(len(all_texts)):
        text = all_texts[i]
        if any(tok in text for tok in tokens):
            any_candidate_idxs.append(i)

    if any_candidate_idxs:
        ranked = sorted(any_candidate_idxs, key=lambda i: float(sims[i]), reverse=True)
        return [int(i) for i in ranked[: min(top_k_fallback, len(ranked))]]

    # If strict token match finds nothing, don't return empty; relax with similarity.
    order = np.argsort(-sims)[: min(top_k_fallback, len(sims))]
    relaxed = [int(i) for i in order if float(sims[i]) >= min_similarity]
    if relaxed:
        return relaxed
    return [int(i) for i in order]


@dataclass(frozen=True)
class SimilarResult:
    index: int
    score: float


def _article_text(title: str, description: str, content: str) -> str:
    parts = [title or "", description or "", content or ""]
    return " \n".join(p.strip() for p in parts if p and p.strip())


def build_tfidf_matrix(
    *,
    titles: Sequence[str],
    descriptions: Sequence[str],
    contents: Sequence[str],
) -> Tuple[TfidfVectorizer, np.ndarray]:
    corpus = [_article_text(t, d, c) for t, d, c in zip(titles, descriptions, contents)]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
    )
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X


def most_similar_articles(
    X,
    *,
    article_index: int,
    top_k: int = 5,
) -> List[SimilarResult]:
    n = X.shape[0]
    if n == 0:
        return []
    if article_index < 0 or article_index >= n:
        return []
    sims = cosine_similarity(X[article_index], X).ravel()
    sims[article_index] = -1.0
    top_k = max(0, min(top_k, n - 1))
    idxs = np.argsort(-sims)[:top_k]
    results: List[SimilarResult] = []
    for i in idxs:
        score = float(sims[i])
        if score <= 0:
            continue
        results.append(SimilarResult(index=int(i), score=score))
    return results


def cluster_articles(
    X,
    *,
    n_clusters: int = 4,
    random_state: int = 42,
) -> Optional[List[int]]:
    n = X.shape[0]
    if n < 3:
        return None
    k = max(2, min(n_clusters, n))
    model = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = model.fit_predict(X)
    return [int(x) for x in labels]

