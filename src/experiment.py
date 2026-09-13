"""The 2x2 that Li et al. set up and never run on medical data.

Their thesis (Mitigating Hallucination in LLMs, SS IV-V):

    RAG fixes KNOWLEDGE-based hallucinations   (wrong or missing facts)
    CoT fixes LOGIC-based hallucinations       (sound facts, broken reasoning)

MedHallu carries category labels that map onto that split (see
data.HALLUCINATION_TYPE -- that mapping is ours, and arguable). So the
prediction is directly testable:

                  knowledge slice      logic slice
    baseline           low                 low
    + RAG             HIGH                 low
    + CoT              low                HIGH
    + both            HIGH                HIGH

A clean diagonal supports the taxonomy. If RAG lifts both, or neither lifts
its own slice, the taxonomy does not transfer to medical text -- which is
equally worth reporting, and nobody has checked.

Runs four conditions and prints one table. Everything else is detect.py.

Usage
-----
  python src/experiment.py --backend ollama:qwen2.5:1.5b-instruct --limit 150
  python src/experiment.py --backend anthropic --limit 300 --config pqa_artificial
"""
import argparse
import json
import time

import pandas as pd

import data
import detect
from backends import get_backend
from config import CONFIGS, RESULTS_DIR

CONDITIONS = [
    ("baseline",   "none",   False),   # 1. detection as the paper measures it
    ("oracle",     "oracle", False),   # 2. the paper's with-knowledge ceiling
    ("rag",        "rag",    False),   # 3. real retrieval
    ("cot",        "none",   True),    # 4. reasoning, no extra knowledge
    ("rag+cot",    "rag",    True),    # 5. both
]

# Not part of the headline table. `oracle+rag` answers a different question --
# WHY rag trails oracle -- by keeping the correct passage and adding distractors
# on top. Enable with --diagnose.
DIAGNOSTIC = ("oracle+rag", "oracle+rag", False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="lexical")
    ap.add_argument("--config", default="pqa_artificial", choices=CONFIGS,
                    help="pqa_artificial by default: the labeled split has only 3 "
                         "Evidence-Fabrication rows, too few for the knowledge slice")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--not-sure", action="store_true")
    ap.add_argument("--self-consistency", type=int, default=0, metavar="N")
    ap.add_argument("--retriever", default="dense")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--corpus", default="textbooks",
                    help="external knowledge source: textbooks | statpearls | "
                         "pubmed | wikipedia, or 'self' for the ablation")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--diagnose", action="store_true",
                    help="also run oracle+rag, which splits rag's shortfall into "
                         "distractor cost vs missing-passage cost")
    args = ap.parse_args()

    full_df = data.load(args.config)
    df = full_df
    if args.limit and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=args.seed)
    df = df.reset_index(names="orig_index")
    pairs = data.to_detection_pairs(df)

    counts = pairs["hallu_type"].value_counts().to_dict()
    print(f"data    : {args.config}, {len(df)} rows -> {len(pairs)} judgements")
    print(f"slices  : knowledge={counts.get('knowledge', 0)}  logic={counts.get('logic', 0)}")
    if min(counts.get("knowledge", 0), counts.get("logic", 0)) < 60:
        print("! one slice is very small; raise --limit before believing the split")

    backend = get_backend(args.backend)
    if args.self_consistency > 1:
        from backends import SelfConsistency
        backend = SelfConsistency(backend, args.self_consistency)
    print(f"backend : {backend.name}")

    detect.attach_rag(df, pairs, args, full_df)

    column = {"none": None, "oracle": "knowledge", "rag": "knowledge_rag",
              "oracle+rag": "knowledge_oracle_rag"}
    conditions = CONDITIONS + ([DIAGNOSTIC] if args.diagnose else [])
    results = {}
    for label, mode, cot in conditions:
        started = time.time()
        report, raw = detect.evaluate(backend, pairs, column[mode], args.not_sure, label, cot)
        results[label] = report
        raw.to_csv(RESULTS_DIR / f"exp_{_slug(backend.name)}_{label}.csv", index=False)
        print(f"  {label:<10} done in {time.time() - started:.1f}s")

    def f1(label, slice_key):
        entry = results[label].get(slice_key)
        return entry["f1"] if entry else float("nan")

    print(f"\n{'condition':<12}{'overall':>10}{'knowledge':>12}{'logic':>10}")
    print("-" * 44)
    for label, _, _ in conditions:
        print(f"{label:<12}{f1(label, 'overall'):>10.3f}"
              f"{f1(label, 'type:knowledge'):>12.3f}{f1(label, 'type:logic'):>10.3f}")

    base_k = f1("baseline", "type:knowledge")
    base_l = f1("baseline", "type:logic")
    print(f"\n{'lift vs baseline':<12}{'':>10}{'knowledge':>12}{'logic':>10}")
    print("-" * 44)
    for label in ("rag", "cot", "rag+cot"):
        print(f"{label:<12}{'':>10}{f1(label, 'type:knowledge') - base_k:>+12.3f}"
              f"{f1(label, 'type:logic') - base_l:>+10.3f}")

    print("\nReading it:")
    print("  RAG should lift the knowledge column more than the logic column.")
    print("  CoT should lift the logic column more than the knowledge column.")
    print("  If both lift both, the taxonomy does not separate these categories.")
    print("  If neither lifts anything, your judge is too weak to show the effect.")

    out = RESULTS_DIR / f"experiment_{_slug(backend.name)}.json"
    out.write_text(json.dumps(
        {"backend": backend.name, "config": args.config, "rows": len(df),
         "retriever": args.retriever, "k": args.k, "slices": counts,
         "results": results}, indent=2))
    print(f"\nsaved -> {out}")


def _slug(text):
    return "".join(c if c.isalnum() else "-" for c in text).strip("-")


if __name__ == "__main__":
    main()
