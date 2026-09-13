"""The RAG pipeline from Li et al. SS IV-A, not just the retrieval step.

The survey specifies three stages. Vanilla RAG implements only the middle one:

    PRE-RETRIEVAL   understand what the query is actually asking      SS IV-A-1
    RETRIEVAL       search the corpus                                 SS IV-A-2
    POST-RETRIEVAL  rerank, filter, compress what came back           SS IV-A-3

This module adds the outer two, plus Hybrid RAG from SS IV-B-3.

Why each one earns its place here
---------------------------------
PRE-RETRIEVAL. MedHallu questions are paper titles, not search queries:
"Do mitochondria play a role in remodelling lace plant leaves during programmed
cell death?" Fed to a textbook index verbatim, that returns general apoptosis
chapters. SS IV-A-1 calls this intent understanding -- the retriever must go
"beyond surface-level keyword matching". We implement query rewriting and
multi-query expansion.

POST-RETRIEVAL. We measured the cost of distractors directly (the oracle+rag
condition). Filtering is the paper's answer to exactly that, and unlike
pre-retrieval it needs no extra LLM calls if you score by similarity.

HYBRID. SS IV-B-3 fuses sparse and dense retrieval because they fail
differently: sparse misses paraphrase, dense misses rare exact terms like drug
names and gene symbols. Medical text is full of both. Reciprocal Rank Fusion
needs no training and no extra model.

Deliberately NOT implemented, and why
-------------------------------------
  GraphRAG / CG-RAG (SS IV-B-1)  needs a document graph built over the corpus
  KG-RAG (SS IV-B-2)             needs UMLS or SNOMED; licence-gated. The paper
                                 recommends this for medicine specifically, so
                                 it is the strongest extension, not a rejection
  Broad retrieval (SS IV-C)      web search; not reproducible for a benchmark
  Tool-augmented (SS V-B)        no tool a hallucination judge would call
  Symbolic reasoning (SS V-C)    medical claims do not linearise cleanly

Usage
-----
  python src/pipeline.py --stages hybrid,rerank --k 3
  python src/pipeline.py --stages rewrite,hybrid,rerank --backend ollama:qwen2.5:3b-instruct
"""
import argparse
import re

import numpy as np

import data
import rag
from config import COL_QUESTION, RESULTS_DIR


# --------------------------------------------------------------------------
# PRE-RETRIEVAL (SS IV-A-1)
# --------------------------------------------------------------------------
REWRITE_PROMPT = """You are preparing a search query for a medical textbook index.

Rewrite the question below as a short query of the medical CONCEPTS a textbook
would index: anatomy, mechanisms, drug classes, disease processes. Drop study
design, dates, sample sizes and question phrasing.

Return only the query, no explanation.

Question: {question}
Query:"""

MULTI_QUERY_PROMPT = """Generate 3 different search queries for finding medical
textbook passages relevant to this question. Vary the angle: one on mechanism,
one on the clinical condition, one on the general principle.

Return exactly 3 lines, one query per line, nothing else.

Question: {question}
Queries:"""


def rewrite_query(backend, question: str) -> str:
    """Turn a paper-title question into textbook vocabulary.

    Falls back to the original question on any failure -- a broken rewrite
    should degrade to vanilla RAG, not to an empty query.
    """
    try:
        out = _complete(backend, REWRITE_PROMPT.format(question=question), max_tokens=60)
        out = out.strip().strip('"').split("\n")[0]
        return out if len(out) > 10 else question
    except Exception:
        return question


def expand_queries(backend, question: str, n: int = 3) -> list[str]:
    """Multi-query expansion: several angles on the same information need.

    The original question is always included, so expansion can only add
    coverage, never lose it.
    """
    try:
        out = _complete(backend, MULTI_QUERY_PROMPT.format(question=question), max_tokens=150)
        lines = [re.sub(r"^\s*[-*\d.)]+\s*", "", ln).strip()
                 for ln in out.strip().split("\n")]
        queries = [ln for ln in lines if len(ln) > 10][:n]
        return [question] + queries
    except Exception:
        return [question]


def _complete(backend, prompt: str, max_tokens: int = 100) -> str:
    """Raw completion. Backends expose `judge`; this reaches past it."""
    if backend.name.startswith("ollama"):
        import requests
        r = requests.post(f"{backend.host}/api/chat", timeout=backend.timeout, json={
            "model": backend.model, "stream": False,
            "options": {"temperature": 0.3, "num_predict": max_tokens},
            "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["message"]["content"]
    if hasattr(backend, "client") and backend.name.startswith("anthropic"):
        resp = backend.client.messages.create(
            model=backend.model, max_tokens=max_tokens, temperature=0.3,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in resp.content if b.type == "text")
    if hasattr(backend, "client"):
        resp = backend.client.chat.completions.create(
            model=backend.model, max_tokens=max_tokens, temperature=0.3,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content
    raise RuntimeError(f"{backend.name} cannot generate text")


# --------------------------------------------------------------------------
# HYBRID RETRIEVAL (SS IV-B-3)
# --------------------------------------------------------------------------
def reciprocal_rank_fusion(rankings: list[np.ndarray], k: int, rrf_k: int = 60) -> np.ndarray:
    """Fuse several ranked lists by 1/(rrf_k + rank).

    RRF combines rankings without needing the scores to be comparable, which
    matters here: TF-IDF cosine and embedding cosine are not on the same scale.
    rrf_k=60 is the standard constant from Cormack et al.; it damps the
    influence of any single list's top hit.
    """
    n_queries = rankings[0].shape[0]
    fused = np.zeros((n_queries, k), dtype=int)
    for q in range(n_queries):
        scores = {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking[q]):
                scores[int(doc)] = scores.get(int(doc), 0.0) + 1.0 / (rrf_k + rank + 1)
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        fused[q, :len(top)] = top
    return fused


class HybridRetriever:
    """Sparse + dense, fused with RRF (SS IV-B-3).

    They fail differently. Sparse misses paraphrase; dense misses rare exact
    tokens -- drug names, gene symbols, abbreviations -- which medical text is
    dense with. Fusing recovers both.
    """

    def __init__(self, corpus, dense_model=None):
        self.name = "hybrid(tfidf+dense)"
        self.sparse = rag.TfidfRetriever(corpus)
        self.dense = rag.DenseRetriever(corpus, dense_model) if dense_model \
            else rag.DenseRetriever(corpus)

    def search(self, queries, k):
        pool = max(k * 4, 20)   # fuse over a deeper pool than we return
        return reciprocal_rank_fusion(
            [self.sparse.search(queries, pool), self.dense.search(queries, pool)], k)


# --------------------------------------------------------------------------
# POST-RETRIEVAL (SS IV-A-3)
# --------------------------------------------------------------------------
def rerank_and_filter(query: str, doc_ids, corpus, embedder=None,
                      keep: int = 3, min_score: float = 0.15):
    """Rescore retrieved passages against the query, drop weak ones.

    Retrieval ranks a passage against the whole corpus; reranking asks a
    sharper question -- how relevant is THIS passage to THIS query. Dropping
    everything below `min_score` is the part that matters: the oracle+rag
    condition showed distractors carry a real cost, and a filtered empty list
    is better than three confident irrelevancies.

    Returns (kept_ids, dropped_count).
    """
    if not len(doc_ids):
        return [], 0

    texts = [corpus[int(j)] for j in doc_ids]
    if embedder is not None:
        q = embedder.model.encode([query], normalize_embeddings=True)[0]
        d = embedder.model.encode(texts, normalize_embeddings=True)
        scores = d @ q
    else:
        # No embedder: fall back to token overlap. Crude, but it still removes
        # passages with almost nothing in common with the query.
        qt = set(re.findall(r"[a-z]{4,}", query.lower()))
        scores = np.array([
            len(qt & set(re.findall(r"[a-z]{4,}", t.lower()))) / max(len(qt), 1)
            for t in texts])

    order = np.argsort(-scores)
    kept = [int(doc_ids[i]) for i in order if scores[i] >= min_score][:keep]
    return kept, len(doc_ids) - len(kept)


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------
def run_pipeline(questions, corpus, stages, k=3, backend=None,
                 retriever=None, min_score=0.15):
    """Run the configured stages and return (passages, stats).

    `stages` is any subset of {"rewrite", "expand", "hybrid", "rerank"}.
    With none of them this is vanilla RAG, which is the point of comparison.
    """
    stats = {"rewritten": 0, "expanded": 0, "dropped": 0}

    # --- pre-retrieval -----------------------------------------------------
    query_sets = [[q] for q in questions]
    if "rewrite" in stages:
        if backend is None:
            raise SystemExit("--stages rewrite needs --backend (it calls an LLM)")
        query_sets = [[rewrite_query(backend, q)] for q in questions]
        stats["rewritten"] = len(questions)
    if "expand" in stages:
        if backend is None:
            raise SystemExit("--stages expand needs --backend (it calls an LLM)")
        query_sets = [expand_queries(backend, q) for q in questions]
        stats["expanded"] = len(questions)

    # --- retrieval ---------------------------------------------------------
    if retriever is None:
        retriever = HybridRetriever(corpus) if "hybrid" in stages \
            else rag.TfidfRetriever(corpus)

    pool = k * 3 if "rerank" in stages else k   # rerank needs candidates to cut
    per_question = []
    for queries in query_sets:
        ranked = retriever.search(queries, pool)
        if len(queries) > 1:
            merged = reciprocal_rank_fusion([ranked[i:i + 1] for i in range(len(queries))],
                                            pool)[0]
        else:
            merged = ranked[0]
        per_question.append(merged)

    # --- post-retrieval ----------------------------------------------------
    embedder = getattr(retriever, "dense", None) if "rerank" in stages else None
    if "rerank" in stages and embedder is None and isinstance(retriever, rag.DenseRetriever):
        embedder = retriever

    passages = []
    for question, ids in zip(questions, per_question):
        if "rerank" in stages:
            ids, dropped = rerank_and_filter(question, ids, corpus, embedder,
                                             keep=k, min_score=min_score)
            stats["dropped"] += dropped
        else:
            ids = list(ids[:k])
        passages.append("\n\n".join(corpus[int(j)] for j in ids) if ids else "")

    stats["empty_after_filter"] = sum(1 for p in passages if not p)
    return passages, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pqa_labeled")
    ap.add_argument("--corpus", default="textbooks")
    ap.add_argument("--stages", default="hybrid,rerank",
                    help="comma list of: rewrite, expand, hybrid, rerank ('' = vanilla)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--backend", default=None, help="needed for rewrite/expand")
    ap.add_argument("--min-score", type=float, default=0.15)
    args = ap.parse_args()

    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    df = data.load(args.config).head(args.limit).reset_index(drop=True)
    corpus = rag.get_corpus(args.corpus, df)
    questions = list(df[COL_QUESTION].astype(str))

    backend = None
    if stages & {"rewrite", "expand"}:
        from backends import get_backend
        backend = get_backend(args.backend or "ollama:qwen2.5:3b-instruct")

    print(f"corpus  : {args.corpus} ({len(corpus):,})")
    print(f"stages  : {sorted(stages) or ['(vanilla)']}")
    print(f"queries : {len(questions)}")

    passages, stats = run_pipeline(questions, corpus, stages, args.k, backend,
                                   min_score=args.min_score)
    print()
    print("stats:", stats)

    if backend is not None:
        print()
        print("=" * 78)
        print("Pre-retrieval: what the query became")
        print("=" * 78)
        for q in questions[:3]:
            print()
            print("  original:", q[:110])
            print("  rewritten:", rewrite_query(backend, q)[:110])

    print()
    print("=" * 78)
    print("First question, what survived the pipeline")
    print("=" * 78)
    print()
    print("QUESTION:", questions[0][:200])
    print()
    print(passages[0][:700] if passages[0] else "  (everything filtered out)")

    out = RESULTS_DIR / f"pipeline_{'-'.join(sorted(stages)) or 'vanilla'}.csv"
    import pandas as pd
    pd.DataFrame({"question": questions, "retrieved": passages}).to_csv(out, index=False)
    print()
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
