"""MedRAG-style retrieval for MedHallu hallucination detection.

Implements the pipeline from Xiong et al., "Benchmarking Retrieval-Augmented
Generation for Medicine" (MIRAGE/MEDRAG), adapted from question answering to
hallucination detection:

    BM25 + MedCPT  ->  RRF fusion  ->  MedCPT rerank
                   ->  confidence threshold  ->  cited evidence
                   ->  abstain when evidence is insufficient

Read this before trusting it
----------------------------
MedRAG reports +1.4% to +10.7% from RRF fusion, but those are **QA accuracy**
numbers on MIRAGE. This is a different task: judging a given answer, not
producing one. Nobody has measured whether the gains transfer. Add one stage at
a time and check.

Corpus choice matters more than it looks
----------------------------------------
MedCPT is trained on PubMed search logs. Pairing it with medical textbooks
wastes most of what makes it good. PubMedQA's `pqa_unlabeled` split -- 61,249
abstracts -- is the same source MedHallu was built from but a *different slice*
(MedHallu uses pqa_labeled + pqa_artificial), so it is genuinely external while
still being the text MedCPT understands.

Costs on a laptop CPU, measured in advance so nothing surprises you
------------------------------------------------------------------
  BM25 index over 61k abstracts          ~1 minute, no model
  MedCPT encoding of 61k abstracts       ~45-90 minutes ONCE, then cached
  MedCPT query encoding                  ~0.1 s per question
  MedCPT cross-encoder reranking         ~1-2 s per question (32 candidates)

Stage 4 roughly doubles per-question time. Stages 1-3 are one-time costs.

Dependencies beyond the base install:
    pip install rank_bm25 transformers torch --index-url ... (CPU torch)
"""
import os
import pickle
import re

import numpy as np

CACHE_DIR = os.environ.get("MEDRAG_CACHE", "./medrag_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def load_pubmed_corpus(max_docs=0):
    """PubMedQA pqa_unlabeled: 61,249 abstracts, ~66 MB. Cached after first use.

    External to MedHallu by construction -- MedHallu draws on the labeled and
    artificial splits, this is the unlabeled one.
    """
    import io
    import pandas as pd
    import requests

    cache = os.path.join(CACHE_DIR, "pubmed_corpus.parquet")
    if os.path.exists(cache):
        frame = pd.read_parquet(cache)
    else:
        print("downloading PubMedQA pqa_unlabeled (once, ~66 MB)...")
        index = requests.get(
            "https://huggingface.co/api/datasets/qiaojin/PubMedQA/parquet",
            timeout=120).json()["pqa_unlabeled"]
        url = (index.get("train") or list(index.values())[0])[0]
        resp = requests.get(url, timeout=1800)
        resp.raise_for_status()
        raw = pd.read_parquet(io.BytesIO(resp.content))
        frame = pd.DataFrame({
            "id": raw["pubid"].astype(str),
            "text": [" ".join(c["contexts"]) if hasattr(c, "keys") else " ".join(c)
                     for c in raw["context"]],
        })
        frame.to_parquet(cache, index=False)
        print(f"  cached -> {cache}")

    passages = frame["text"].astype(str).tolist()
    return passages[:max_docs] if max_docs else passages


# ---------------------------------------------------------------------------
# Stage 1: retrievers
# ---------------------------------------------------------------------------
class BM25Retriever:
    """Lexical retrieval. Not TF-IDF -- BM25 saturates term frequency and
    normalises by document length, which matters when abstracts vary from 100
    to 500 words.
    """

    name = "bm25"

    def __init__(self, corpus):
        from rank_bm25 import BM25Okapi
        self.corpus = corpus
        self.index = BM25Okapi([self._tok(d) for d in corpus])

    @staticmethod
    def _tok(text):
        return re.findall(r"[a-z0-9]+", str(text).lower())

    def search(self, query, k):
        scores = self.index.get_scores(self._tok(query))
        top = np.argsort(-scores)[:k]
        return list(top), scores[top]


class MedCPTRetriever:
    """MedCPT bi-encoder (Jin et al. 2023), trained on PubMed search logs.

    Separate encoders for queries and articles, scored by inner product -- a
    query and the abstract that answers it are not the same kind of text, so
    one shared encoder would be the wrong model.
    """

    name = "medcpt"

    def __init__(self, corpus, batch_size=32):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.corpus = corpus
        self.q_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.q_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").eval()

        cache = os.path.join(CACHE_DIR, f"medcpt_embeddings_{len(corpus)}.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as fh:
                self.doc_emb = pickle.load(fh)
        else:
            print(f"encoding {len(corpus):,} passages with MedCPT "
                  "(one-time, 45-90 min on CPU)...")
            a_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
            a_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").eval()
            chunks = []
            with torch.no_grad():
                for i in range(0, len(corpus), batch_size):
                    batch = corpus[i:i + batch_size]
                    enc = a_tok(batch, truncation=True, padding=True,
                                max_length=512, return_tensors="pt")
                    # MedCPT uses the [CLS] representation, not mean pooling.
                    chunks.append(a_model(**enc).last_hidden_state[:, 0, :].numpy())
                    if i % (batch_size * 50) == 0:
                        print(f"  {i:,}/{len(corpus):,}")
            self.doc_emb = np.vstack(chunks)
            with open(cache, "wb") as fh:
                pickle.dump(self.doc_emb, fh)
            print(f"  cached -> {cache}")

    def search(self, query, k):
        with self.torch.no_grad():
            enc = self.q_tok([query], truncation=True, padding=True,
                             max_length=64, return_tensors="pt")
            q = self.q_model(**enc).last_hidden_state[:, 0, :].numpy()[0]
        scores = self.doc_emb @ q          # inner product, per the model card
        top = np.argsort(-scores)[:k]
        return list(top), scores[top]


# ---------------------------------------------------------------------------
# Stage 2: RRF fusion
# ---------------------------------------------------------------------------
def rrf_fuse(ranked_lists, k, rrf_k=60):
    """Reciprocal Rank Fusion (Cormack et al. 2009), MedRAG's RRF-2.

    Fuses by rank, not score, which is the point: BM25 scores are unbounded
    and MedCPT inner products are not, so they cannot be averaged directly.
    """
    scores = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            scores[int(doc)] = scores.get(int(doc), 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:k]


# ---------------------------------------------------------------------------
# Stage 3: reranking
# ---------------------------------------------------------------------------
class MedCPTReranker:
    """MedCPT cross-encoder. Scores query and passage jointly.

    Retrieval asks "is this passage about roughly this topic"; reranking asks
    "does this passage answer this query". The cross-encoder can do the second
    because it sees both texts at once -- which is also why it cannot be
    pre-computed, and so only runs over the shortlist.
    """

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "ncbi/MedCPT-Cross-Encoder").eval()

    def rerank(self, query, passages, keep):
        if not passages:
            return [], []
        pairs = [[query, p] for p in passages]
        with self.torch.no_grad():
            enc = self.tok(pairs, truncation=True, padding=True,
                           max_length=512, return_tensors="pt")
            scores = self.model(**enc).logits.squeeze(-1).numpy()
        scores = np.atleast_1d(scores)
        order = np.argsort(-scores)[:keep]
        return [passages[i] for i in order], scores[order]


# ---------------------------------------------------------------------------
# Stage 4: confidence, abstention, citation
# ---------------------------------------------------------------------------
def evidence_is_sufficient(scores, threshold):
    """Is the retrieved evidence good enough to judge on?

    Uses the single best score, not the mean: one strongly relevant passage is
    enough to decide a claim, and averaging lets two weak passages drag a good
    one below the line.

    Thresholds are scale-dependent -- MedCPT cross-encoder logits, MedCPT inner
    products and BM25 scores are not comparable. Calibrate per configuration by
    printing the distribution before choosing a number.
    """
    return bool(len(scores)) and float(np.max(scores)) >= threshold


def format_cited_evidence(passages, max_chars=1200):
    """Number the passages so the judge can point at which one it used.

    Citation is not decoration here: a verdict that cites [2] is checkable, and
    one that cites nothing while claiming contradiction is a visible failure.
    """
    parts = []
    budget = max_chars // max(len(passages), 1)
    for i, passage in enumerate(passages, 1):
        parts.append(f"[{i}] {str(passage)[:budget]}")
    return "\n\n".join(parts)


ABSTAIN_NOTE = (
    "\n\nNOTE: retrieval found no strongly relevant evidence for this question. "
    "If you cannot judge the answer from your own knowledge with confidence, "
    "answer '2' (not sure) rather than guessing.")


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
class MedRAGPipeline:
    """Configurable so stages can be added one at a time and measured.

    Turning everything on at once means a change in F1 cannot be attributed to
    any stage. Start with `use_medcpt=False, use_rerank=False` -- that is BM25
    only -- and enable one flag per run.
    """

    def __init__(self, corpus, use_medcpt=True, use_rerank=True,
                 k=3, pool=32, threshold=None):
        self.corpus = corpus
        self.k = k
        self.pool = pool          # MedRAG retrieves 32 snippets by default
        self.threshold = threshold
        self.bm25 = BM25Retriever(corpus)
        self.medcpt = MedCPTRetriever(corpus) if use_medcpt else None
        self.reranker = MedCPTReranker() if use_rerank else None

        stages = ["bm25"]
        if use_medcpt:
            stages.append("medcpt+rrf")
        if use_rerank:
            stages.append("rerank")
        if threshold is not None:
            stages.append(f"threshold={threshold}")
        self.name = " -> ".join(stages)

    def retrieve(self, question):
        """Return (evidence_text, sufficient, scores)."""
        bm_ids, bm_scores = self.bm25.search(question, self.pool)

        if self.medcpt is not None:
            mc_ids, _ = self.medcpt.search(question, self.pool)
            doc_ids = rrf_fuse([bm_ids, mc_ids], self.pool)
            # RRF returns fused ranks, not comparable scores. Re-derive a score
            # only if something downstream needs one.
            scores = bm_scores[:len(doc_ids)] if self.reranker is None else None
        else:
            doc_ids, scores = bm_ids, bm_scores

        passages = [self.corpus[int(j)] for j in doc_ids]

        if self.reranker is not None:
            passages, scores = self.reranker.rerank(question, passages, self.k)
        else:
            passages = passages[:self.k]
            scores = np.asarray(scores[:self.k]) if scores is not None else np.array([])

        sufficient = True
        if self.threshold is not None:
            sufficient = evidence_is_sufficient(scores, self.threshold)

        return format_cited_evidence(passages), sufficient, scores

    def knowledge_for(self, question):
        """What to hand the judge. Appends the abstain note when evidence is thin."""
        evidence, sufficient, _ = self.retrieve(question)
        if self.threshold is not None and not sufficient:
            return evidence + ABSTAIN_NOTE
        return evidence


def calibrate_threshold(pipeline, questions, percentile=25):
    """Pick a threshold from the data instead of guessing.

    Returns the score at the given percentile across a sample of questions --
    i.e. abstain on roughly the worst `percentile`% of retrievals. Absolute
    values differ by orders of magnitude between BM25 and cross-encoder logits,
    so a hard-coded number would be wrong for every configuration but one.
    """
    best = []
    for question in questions:
        _, _, scores = pipeline.retrieve(question)
        if len(scores):
            best.append(float(np.max(scores)))
    if not best:
        return None
    value = float(np.percentile(best, percentile))
    print(f"threshold at p{percentile}: {value:.3f}   "
          f"(min {min(best):.3f}, median {np.median(best):.3f}, max {max(best):.3f})")
    return value
