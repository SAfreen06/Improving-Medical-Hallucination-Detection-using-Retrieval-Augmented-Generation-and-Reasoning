"""Stage-by-stage check of medrag_pipeline on a small corpus slice.

Run this before committing to the full 61k encode. It uses 3,000 abstracts so
MedCPT finishes in minutes rather than an hour, which is enough to confirm the
stages behave differently from each other.

  python test_medrag.py            # 3,000 abstracts
  python test_medrag.py --full     # all 61,249 (45-90 min on CPU, once)
"""
import argparse

import numpy as np

import medrag_pipeline as mp

QUESTION = ("Do mitochondria play a role in remodelling lace plant leaves "
            "during programmed cell death?")


def show(label, pipeline, question, chars=260):
    evidence, sufficient, scores = pipeline.retrieve(question)
    print()
    print("=" * 78)
    print(label)
    print("  pipeline:", pipeline.name)
    if scores is not None and len(scores):
        print("  scores  :", np.round(np.asarray(scores, dtype=float), 2))
    print("  enough  :", sufficient)
    print("-" * 78)
    print(evidence[:chars])
    return evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--question", default=QUESTION)
    args = ap.parse_args()

    corpus = mp.load_pubmed_corpus(max_docs=0 if args.full else 3000)
    print(f"corpus: {len(corpus):,} PubMed abstracts")
    print(f"question: {args.question}")

    show("STAGE 1 - BM25 only",
         mp.MedRAGPipeline(corpus, use_medcpt=False, use_rerank=False, k=3),
         args.question)

    show("STAGE 2 - BM25 + MedCPT, fused with RRF",
         mp.MedRAGPipeline(corpus, use_medcpt=True, use_rerank=False, k=3),
         args.question)

    pipe3 = mp.MedRAGPipeline(corpus, use_medcpt=True, use_rerank=True, k=3)
    show("STAGE 3 - ...plus MedCPT cross-encoder reranking", pipe3, args.question)

    print()
    print("=" * 78)
    print("STAGE 4 - threshold calibration")
    print("=" * 78)
    import data_questions
    sample = data_questions.sample_questions(30)
    threshold = mp.calibrate_threshold(pipe3, sample, percentile=25)
    if threshold is not None:
        pipe4 = mp.MedRAGPipeline(corpus, use_medcpt=True, use_rerank=True,
                                  k=3, threshold=threshold)
        _, sufficient, _ = pipe4.retrieve(args.question)
        print(f"  this question passes the threshold: {sufficient}")
        if not sufficient:
            print("  -> judge would be told to prefer '2' (not sure)")


if __name__ == "__main__":
    main()
