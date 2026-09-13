"""Test the paper's semantic claim on a laptop, with no GPU.

Section 5.3 reports that harder-to-detect hallucinations sit semantically
*closer* to the ground truth. The paper shows this by generating 50 candidates
per question and clustering them by bidirectional entailment (Table 3) -- that
generation step needs the GPUs we do not have.

But the published dataset already encodes the same signal. A row's difficulty
label *is* how many discriminator LLMs the hallucination fooled: easy = one,
medium = several, hard = all of them. So the claim becomes directly testable on
the 10k released rows:

    similarity(hard hallucination, ground truth)
        > similarity(easy hallucination, ground truth)

That needs no generation and no clustering -- just a similarity measure and a
significance test. Three measures are computed, cheapest first:

  rouge1      unigram overlap F1. Pure Python.
  tfidf_cos   cosine over TF-IDF vectors. sklearn only.
  embed_cos   cosine over sentence-transformer embeddings. Needs torch (CPU is
              fine, ~350 MB of wheels); skipped automatically if absent.

Usage:
  python src/semantics.py                       # all 1000 labeled rows
  python src/semantics.py --config pqa_artificial --limit 3000
  python src/semantics.py --embed               # add the neural measure
"""
import argparse
import textwrap
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

import data
from config import COL_DIFFICULTY, COL_HALLU, COL_TRUTH, RESULTS_DIR

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN.findall(str(text).lower())


def rouge1_f1(candidate, reference):
    """Unigram-overlap F1, the ROUGE-1 variant reported in the paper's Table 3."""
    cand, ref = tokenize(candidate), tokenize(reference)
    if not cand or not ref:
        return 0.0
    cand_counts, ref_counts = pd.Series(cand).value_counts(), pd.Series(ref).value_counts()
    shared = cand_counts.index.intersection(ref_counts.index)
    overlap = sum(min(cand_counts[w], ref_counts[w]) for w in shared)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(cand), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def tfidf_cosine(hallucinated, truth):
    """Row-wise cosine similarity between each pair, in one shared TF-IDF space."""
    vec = TfidfVectorizer(lowercase=True, stop_words="english")
    matrix = vec.fit_transform(list(hallucinated) + list(truth))
    n = len(hallucinated)
    left, right = matrix[:n], matrix[n:]
    # Rows are already L2-normalised by TfidfVectorizer, so the dot product
    # is the cosine directly.
    return np.asarray(left.multiply(right).sum(axis=1)).ravel()


def embed_cosine(hallucinated, truth, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer  # optional dependency
    model = SentenceTransformer(model_name, device="cpu")
    a = model.encode(list(hallucinated), normalize_embeddings=True,
                     batch_size=16, show_progress_bar=True)
    b = model.encode(list(truth), normalize_embeddings=True,
                     batch_size=16, show_progress_bar=True)
    return (a * b).sum(axis=1)


def compare(df, measure):
    """Hard vs easy, reported as means plus a Mann-Whitney U test.

    Mann-Whitney rather than a t-test: these similarity scores are bounded in
    [0, 1] and skewed, so a rank test is the safer call.

    The test is two-sided on purpose. A one-sided test in the direction the
    paper predicts would return p ~ 1.0 if the effect runs the other way, which
    reads as "no effect" when it is really "strong effect, opposite sign".
    """
    easy = df.loc[df[COL_DIFFICULTY] == "easy", measure].dropna()
    hard = df.loc[df[COL_DIFFICULTY] == "hard", measure].dropna()
    _, p_two_sided = stats.mannwhitneyu(hard, easy, alternative="two-sided")
    return {
        "measure": measure,
        "mean_easy": round(float(easy.mean()), 4),
        "mean_medium": round(float(df.loc[df[COL_DIFFICULTY] == "medium", measure].mean()), 4),
        "mean_hard": round(float(hard.mean()), 4),
        "delta_hard_minus_easy": round(float(hard.mean() - easy.mean()), 4),
        "p_two_sided": float(p_two_sided),
        "n_easy": int(len(easy)),
        "n_hard": int(len(hard)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pqa_labeled")
    ap.add_argument("--limit", type=int, default=0, help="0 = use every row")
    ap.add_argument("--embed", action="store_true",
                    help="also run sentence-transformer embeddings (needs torch)")
    args = ap.parse_args()

    df = data.load(args.config)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=0)
    df = df.reset_index(drop=True)

    hallucinated = df[COL_HALLU].astype(str)
    truth = df[COL_TRUTH].astype(str)

    df["rouge1"] = [rouge1_f1(h, t) for h, t in zip(hallucinated, truth)]
    df["tfidf_cos"] = tfidf_cosine(hallucinated, truth)
    measures = ["rouge1", "tfidf_cos"]

    if args.embed:
        try:
            df["embed_cos"] = embed_cosine(hallucinated, truth)
            measures.append("embed_cos")
        except ImportError:
            print("! sentence-transformers not installed; skipping embed_cos.\n"
                  "  see README for the CPU-only install command")

    rows = [compare(df, m) for m in measures]
    table = pd.DataFrame(rows)

    print(f"\nMedHallu {args.config}: {len(df)} rows")
    print("Claim under test (paper Sec 5.3): hard hallucinations are semantically")
    print("closer to the ground truth than easy ones.\n")
    print(table.to_string(index=False))

    print("\nverdict per measure  (hard closer to ground truth than easy?):")
    for row in rows:
        if row["p_two_sided"] >= 0.05:
            verdict = "no difference"
        elif row["delta_hard_minus_easy"] > 0:
            verdict = "YES, as predicted"
        else:
            verdict = "NO, reversed"
        print(f"  {row['measure']:<11}{verdict:<18} p={row['p_two_sided']:.2e}")

    print("\n  paper Table 3, fooled vs not-fooled clusters:"
          "  Rouge1-F1 0.358 vs 0.319 (p=0.002)")
    print(textwrap.dedent("""
      Reading a reversal here: it is not a refutation. Two things differ.
      (1) The paper compares clusters over 50 candidate generations per
          question; this compares the one hallucination that was released.
      (2) The generation pipeline's fallback rule picks, out of all failed
          candidates, the one with MAXIMUM cosine similarity to the ground
          truth -- and labels it easy (paper Sec 3, Algorithm 1 Phase 2).
          So easy rows are enriched with answers explicitly selected for
          being close to the truth, which pushes easy similarity up.
      Lexical overlap is also not semantic closeness. Re-run with --embed
      to test the same claim in embedding space.
    """).rstrip())

    out = RESULTS_DIR / f"semantics_{args.config}.csv"
    table.to_csv(out, index=False)
    df[[COL_DIFFICULTY] + measures].to_csv(
        RESULTS_DIR / f"semantics_{args.config}_per_row.csv", index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
