"""Retrieval over an EXTERNAL medical knowledge source.

What RAG has to mean here
-------------------------
MedHallu ships a `Knowledge` field per row: the exact PubMed abstract the
question was written from. Handing that to the judge is the paper's "with
knowledge" setting -- call it ORACLE. Perfect retrieval, guaranteed correct,
zero noise. No deployed system has it.

An earlier version of this module retrieved from those same Knowledge fields.
That was wrong. Searching the benchmark's own answer key is not retrieval-
augmented generation; the correct passage is present by construction, so the
task is only ever "rank it first". It measures ranking, not knowledge access.

Real RAG searches a corpus the benchmark knows nothing about, where the needed
fact may simply not be there. That is the hard part, and it is what this module
now does.

Corpora
-------
From MedRAG (Xiong et al., Benchmarking RAG for Medicine). All ungated.

  textbooks    125,847 snippets from 18 medical textbooks.  ~101 MB.  DEFAULT.
  statpearls    9,330 clinical reference articles.          small.
  pubmed       23.9M abstract snippets.                     tens of GB -- needs
                                                            a real machine.
  wikipedia    general encyclopaedia.                       large.

Textbooks is the default because it fits a laptop and is a genuinely different
source from MedHallu's PubMed-derived questions. A medical textbook may or may
not contain the specific finding a 2012 retrospective laparotomy study reports
-- and that uncertainty is the realistic case.

  self         MedHallu's own Knowledge fields. Kept ONLY as an ablation, to
               show the gap between real retrieval and searching the answer
               key. Do not report this as RAG.

Usage
-----
  python src/rag.py --corpus textbooks --k 3
  python src/rag.py --corpus textbooks --k 5 --retriever dense
  python src/rag.py --corpus self --k 3        # ablation, not a RAG result
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import data
from config import CACHE_DIR, COL_KNOWLEDGE, COL_QUESTION, DATA_DIR, RESULTS_DIR


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------
EXTERNAL_CORPORA = {
    "textbooks": "MedRAG/textbooks",
    "statpearls": "MedRAG/statpearls",
    "pubmed": "MedRAG/pubmed",
    "wikipedia": "MedRAG/wikipedia",
}


def load_external_corpus(name: str, max_docs: int = 0) -> list[str]:
    """Download and cache a MedRAG corpus. Returns a list of passages.

    Cached as parquet after the first run, so the 101 MB textbooks download
    happens once.
    """
    import io
    import requests

    if name not in EXTERNAL_CORPORA:
        raise SystemExit(f"unknown corpus {name!r}; choose from "
                         f"{list(EXTERNAL_CORPORA)} or 'self'")
    repo = EXTERNAL_CORPORA[name]
    local = DATA_DIR / f"corpus_{name}.parquet"

    if local.exists():
        frame = pd.read_parquet(local)
    else:
        print(f"downloading {repo} (first run only)...")
        index = requests.get(
            f"https://huggingface.co/api/datasets/{repo}/parquet", timeout=120).json()
        config = list(index)[0]
        files = index[config].get("train") or list(index[config].values())[0]
        frames = []
        for i, url in enumerate(files, 1):
            print(f"  shard {i}/{len(files)}")
            resp = requests.get(url, timeout=1800)
            resp.raise_for_status()
            frames.append(pd.read_parquet(io.BytesIO(resp.content)))
        frame = pd.concat(frames, ignore_index=True)
        frame.to_parquet(local, index=False)
        print(f"  cached -> {local}")

    # MedRAG uses `contents` (title prefixed) where available; it retrieves
    # better than the bare body because the book name carries topic signal.
    column = "contents" if "contents" in frame.columns else "content"
    passages = frame[column].astype(str).tolist()
    if max_docs:
        passages = passages[:max_docs]
    return passages


def build_self_corpus(df: pd.DataFrame) -> list[str]:
    """MedHallu's own Knowledge fields. ABLATION ONLY.

    Retrieving from here means the correct passage is always present, so
    results are an upper bound that no real system reaches. Useful to quantify
    how much easier that makes the task; not a RAG result.
    """
    return [data._flatten_knowledge(v) for v in df[COL_KNOWLEDGE]]


def get_corpus(name: str, df: pd.DataFrame | None = None, max_docs: int = 0) -> list[str]:
    if name == "self":
        if df is None:
            raise SystemExit("corpus 'self' needs the dataframe")
        return build_self_corpus(df)
    return load_external_corpus(name, max_docs)


# --------------------------------------------------------------------------
# Retrievers
# --------------------------------------------------------------------------
class TfidfRetriever:
    """Sparse lexical retrieval. No torch, no downloads, runs in a second.

    Weak on paraphrase -- a question and its source passage often share few
    exact words -- which is precisely why the dense retriever below is worth
    the extra 350 MB. Keep this as the cheap baseline.
    """

    name = "tfidf"

    def __init__(self, corpus: list[str]):
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english",
                                          sublinear_tf=True)
        self.doc_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, queries: list[str], k: int) -> np.ndarray:
        q = self.vectorizer.transform(queries)
        scores = (q @ self.doc_matrix.T).toarray()
        return np.argsort(-scores, axis=1)[:, :k]


class DenseRetriever:
    """Sentence-embedding retrieval with cosine similarity.

    The corpus is small enough (10k docs) that a plain matrix multiply beats
    setting up FAISS. Embeddings are cached to disk keyed by model and corpus
    size, since encoding 10k passages on CPU takes a couple of minutes.
    """

    def __init__(self, corpus: list[str],
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dep
        self.name = f"dense:{model_name.split('/')[-1]}"
        self.model = SentenceTransformer(model_name, device="cpu")

        cache = CACHE_DIR / f"corpus_{model_name.replace('/', '_')}_{len(corpus)}.pkl"
        if cache.exists():
            self.doc_embeddings = pickle.loads(cache.read_bytes())
        else:
            self.doc_embeddings = self.model.encode(
                corpus, normalize_embeddings=True, batch_size=32,
                show_progress_bar=True)
            cache.write_bytes(pickle.dumps(self.doc_embeddings))

    def search(self, queries: list[str], k: int) -> np.ndarray:
        q = self.model.encode(queries, normalize_embeddings=True, batch_size=32,
                              show_progress_bar=True)
        scores = q @ self.doc_embeddings.T
        return np.argsort(-scores, axis=1)[:, :k]


def get_retriever(spec: str, corpus: list[str]):
    if spec == "tfidf":
        return TfidfRetriever(corpus)
    if spec.startswith("dense"):
        _, _, model = spec.partition(":")
        return DenseRetriever(corpus, model or "sentence-transformers/all-MiniLM-L6-v2")
    raise SystemExit(f"unknown retriever {spec!r} (use 'tfidf' or 'dense[:model]')")


# --------------------------------------------------------------------------
# Evaluation and use
# --------------------------------------------------------------------------
def recall_at_k(ranked: np.ndarray, k: int, gold: np.ndarray | None = None) -> float:
    """Fraction of questions whose own document appears in the top k.

    `gold[i]` is the corpus position of question i's own passage. It defaults to
    arange, which is right only when the queries ARE the corpus. Once you
    evaluate a sample of questions against a full corpus -- which you should,
    since shrinking the corpus to the sample makes retrieval trivially easy --
    the positions differ and must be passed in.
    """
    gold = (np.arange(len(ranked)) if gold is None else np.asarray(gold))[:, None]
    return float((ranked[:, :k] == gold).any(axis=1).mean())


def mrr(ranked: np.ndarray, gold: np.ndarray | None = None) -> float:
    """Mean reciprocal rank of the correct document."""
    gold = (np.arange(len(ranked)) if gold is None else np.asarray(gold))[:, None]
    hits = ranked == gold
    total = 0.0
    for row in range(len(ranked)):
        pos = np.flatnonzero(hits[row])
        if len(pos):
            total += 1.0 / (pos[0] + 1)
    return total / len(ranked)


def retrieve_knowledge(df: pd.DataFrame, retriever, k: int,
                       corpus: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    """Return the concatenated top-k passages per question, plus the rankings.

    This is what gets handed to the judge in place of the oracle context. The
    top-k are joined in rank order, so a retriever that puts the right passage
    first is rewarded by the model reading it first.
    """
    corpus = corpus if corpus is not None else build_corpus(df)
    ranked = retriever.search(list(df[COL_QUESTION].astype(str)), k)
    passages = ["\n\n".join(corpus[j] for j in row) for row in ranked]
    return passages, ranked


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pqa_labeled")
    ap.add_argument("--corpus", default="textbooks",
                    help="textbooks | statpearls | pubmed | wikipedia | self")
    ap.add_argument("--retriever", default="tfidf", help="tfidf | dense[:model]")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=20, help="questions to show")
    ap.add_argument("--max-docs", type=int, default=0, help="cap corpus size")
    args = ap.parse_args()

    df = data.load(args.config).head(args.limit).reset_index(drop=True)
    corpus = get_corpus(args.corpus, df, args.max_docs)
    print(f"corpus    : {args.corpus}  ({len(corpus):,} passages)")

    retriever = get_retriever(args.retriever, corpus)
    passages, ranked = retrieve_knowledge(df, retriever, args.k, corpus)
    print(f"retriever : {retriever.name}   k={args.k}   queries={len(df)}")

    if args.corpus == "self":
        print()
        print(f"recall@{args.k}: {recall_at_k(ranked, args.k):.3f}   "
              f"MRR: {mrr(ranked):.3f}")
        print("(only computable because the gold passage IS the corpus --")
        print(" this is the ablation, not a RAG result)")
    else:
        print()
        print("No recall@k: an external corpus has no labelled gold passage for")
        print("these questions. That is the point -- judge the retrieval by")
        print("whether detection F1 improves, not by rank.")

    print()
    print("=" * 78)
    print("What got retrieved for the first question")
    print("=" * 78)
    row = df.iloc[0]
    print()
    print("QUESTION")
    print(" ", str(row[COL_QUESTION])[:300])
    print()
    print("ORACLE -- what MedHallu ships, for comparison")
    print(" ", data._flatten_knowledge(row[COL_KNOWLEDGE])[:300], "...")
    print()
    print(f"RETRIEVED from {args.corpus}:")
    for rank, j in enumerate(ranked[0], 1):
        print()
        print(f"  [{rank}]", corpus[j][:280], "...")

    out = RESULTS_DIR / f"retrieval_{args.corpus}_{args.retriever.replace(':', '-')}.csv"
    pd.DataFrame({"question": df[COL_QUESTION],
                  "retrieved": passages}).to_csv(out, index=False)
    print()
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
