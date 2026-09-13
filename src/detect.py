"""Run the MedHallu detection benchmark.

Reproduces the shape of the paper's headline results on a laptop:

  Table 2  overall / easy / medium / hard F1, by knowledge condition
  Table 4  effect of offering the model a "not sure" option

Three knowledge conditions, not the paper's two:

  none     the judge sees only the question and answer
  oracle   the judge gets the exact PubMed context the question came from.
           This is the paper's "with knowledge" setting -- perfect retrieval.
  rag      the judge gets the top-k passages an actual retriever fetched from
           an EXTERNAL corpus (medical textbooks). See rag.py.
  oracle+rag
           both: the correct passage AND the retrieved ones. Isolates the cost
           of distractors from the cost of missing the passage --
             oracle - (oracle+rag) = what the noise alone costs
             (oracle+rag) - rag    = what missing the passage costs

The paper measures none (~0.53) and oracle (~0.78) and leaves the gap between
them unexplored. Running all three tells you how much of that +0.25 survives
imperfect retrieval.

Examples
--------
  python src/detect.py --backend constant --limit 300
  python src/detect.py --backend lexical --limit 300 --knowledge-mode none,oracle
  python src/detect.py --backend ollama:qwen2.5:1.5b-instruct --limit 100 \
      --knowledge-mode none,oracle,rag --retriever tfidf --k 3
"""
import argparse
import json
import time

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

import data
from backends import get_backend
from config import CONFIGS, RESULTS_DIR

MODES = ("none", "oracle", "rag", "oracle+rag")


def score(preds, labels, allow_not_sure):
    """Precision / recall / F1 for the positive (hallucinated) class.

    With "not sure" enabled, abstentions are dropped before scoring and reported
    separately as `response_pct` -- the paper's Table 4 does the same, which is
    why precision can rise while coverage falls.
    """
    frame = pd.DataFrame({"pred": preds, "label": labels})
    total = len(frame)
    if allow_not_sure:
        frame = frame[frame["pred"] != 2]
    if frame.empty:
        return {"n": total, "response_pct": 0.0, "precision": float("nan"),
                "recall": float("nan"), "f1": float("nan")}
    return {
        "n": total,
        "response_pct": round(100 * len(frame) / total, 1),
        "precision": round(precision_score(frame["label"], frame["pred"], zero_division=0), 3),
        "recall": round(recall_score(frame["label"], frame["pred"], zero_division=0), 3),
        "f1": round(f1_score(frame["label"], frame["pred"], zero_division=0), 3),
    }


def evaluate(backend, pairs, knowledge_col, allow_not_sure, mode_label, cot=False):
    preds = []
    desc = f"{backend.name} | {mode_label} | not_sure={allow_not_sure}"
    knowledge = pairs[knowledge_col] if knowledge_col else None

    for i, row in enumerate(tqdm(pairs.itertuples(index=False), total=len(pairs), desc=desc)):
        preds.append(backend.judge(
            question=row.question,
            answer=row.answer,
            knowledge=knowledge.iat[i] if knowledge is not None else None,
            allow_not_sure=allow_not_sure,
            cot=cot,
        ))

    out = pairs.copy()
    out["pred"] = preds

    # Difficulty labels a source row, and both of that row's examples carry it.
    # Slicing on the label keeps each difficulty balanced 1:1 the same way the
    # full set is -- otherwise precision within a slice is not comparable to
    # overall precision, or to the paper's per-difficulty columns.
    report = {"overall": score(out["pred"], out["label"], allow_not_sure)}
    for level in ("easy", "medium", "hard"):
        slice_ = out[out["difficulty"] == level]
        report[level] = score(slice_["pred"], slice_["label"], allow_not_sure)

    # Li et al.'s taxonomy: RAG is predicted to help the knowledge slice, CoT
    # the logic slice. Both slices stay balanced because hallu_type, like
    # difficulty, is a property of the source row.
    for htype in ("knowledge", "logic"):
        slice_ = out[out["hallu_type"] == htype]
        report[f"type:{htype}"] = (score(slice_["pred"], slice_["label"], allow_not_sure)
                                   if len(slice_) else None)
    return report, out


def print_report(title, report):
    print(f"\n{title}")
    header = f"  {'slice':<15}{'n':>6}{'P':>8}{'R':>8}{'F1':>8}{'resp%':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in ("overall", "easy", "medium", "hard", "type:knowledge", "type:logic"):
        m = report.get(key)
        if m is None:
            continue
        print(f"  {key:<15}{m['n']:>6}{m['precision']:>8}{m['recall']:>8}"
              f"{m['f1']:>8}{m['response_pct']:>8}")


def attach_rag(df, pairs, args, full_df=None):
    """Add a `knowledge_rag` column: the top-k passages a retriever fetched.

    The corpus is built from the FULL config, never from the sampled subset.
    Retrieving 300 questions out of a 300-document corpus is trivially easy and
    would flatter the results; the retriever should face the whole collection
    however many queries you score.
    """
    import rag  # imported lazily so non-RAG runs need no retrieval deps

    full = df if full_df is None else full_df
    corpus = rag.get_corpus(args.corpus, full, args.max_docs)
    print(f"rag corpus: {args.corpus} ({len(corpus):,} passages)")
    if args.corpus == "self":
        print("  ! 'self' searches MedHallu's own Knowledge fields.")
        print("    The correct passage is present by construction, so this is")
        print("    an ablation, not a RAG result.")

    retriever = rag.get_retriever(args.retriever, corpus)
    passages, ranked = rag.retrieve_knowledge(df, retriever, args.k, corpus)

    # recall@k is only defined for the 'self' ablation, where the gold passage
    # is in the corpus. An external corpus has no labelled gold document.
    recall = None
    if args.corpus == "self":
        gold = df["orig_index"].to_numpy() if "orig_index" in df.columns else None
        recall = rag.recall_at_k(ranked, args.k, gold)
        print(f"rag retriever: {retriever.name}   k={args.k}   recall@{args.k}={recall:.3f}")
    else:
        print(f"rag retriever: {retriever.name}   k={args.k}")

    # pairs carry source_row, so the same retrieval serves both of a row's
    # examples without retrieving twice.
    by_row = dict(enumerate(passages))
    pairs["knowledge_rag"] = pairs["source_row"].map(by_row)

    # Oracle first, then the retrieved passages. Position matters -- models read
    # the head of a context best -- so putting oracle first is the GENEROUS
    # arrangement. If it still loses to plain oracle, distractors are doing real
    # damage; shuffle the order for a harsher test.
    separator = "\n\n"
    pairs["knowledge_oracle_rag"] = (
        pairs["knowledge"].astype(str) + separator + pairs["knowledge_rag"].astype(str))
    return recall


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="lexical",
                    help="constant | lexical | ollama:<model> | anthropic[:model] | openai[:model]")
    ap.add_argument("--config", default="pqa_labeled", choices=CONFIGS)
    ap.add_argument("--limit", type=int, default=200,
                    help="source rows to use; each yields 2 judgements")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--knowledge-mode", default="none,oracle",
                    help="comma list of: none, oracle, rag")
    ap.add_argument("--not-sure", action="store_true", help="offer the abstain option")
    ap.add_argument("--cot", action="store_true",
                    help="chain-of-thought prompting (Li et al. SS V)")
    ap.add_argument("--self-consistency", type=int, default=0, metavar="N",
                    help="sample the judge N times and majority-vote; needs --cot to be "
                         "meaningful, and costs N forward passes per judgement")
    ap.add_argument("--retriever", default="tfidf", help="rag only: tfidf | dense[:model]")
    ap.add_argument("--k", type=int, default=3, help="rag only: passages to retrieve")
    ap.add_argument("--corpus", default="textbooks",
                    help="rag only: textbooks | statpearls | pubmed | wikipedia | self")
    ap.add_argument("--max-docs", type=int, default=0,
                    help="rag only: cap the corpus size for a quick run")
    args = ap.parse_args()

    modes = [m.strip() for m in args.knowledge_mode.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise SystemExit(f"unknown knowledge mode(s) {bad}; choose from {list(MODES)}")

    full_df = data.load(args.config)
    df = full_df
    if args.limit and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=args.seed)
    df = df.reset_index(names="orig_index") if "orig_index" not in df.columns         else df.reset_index(drop=True)
    pairs = data.to_detection_pairs(df)

    backend = get_backend(args.backend)
    if args.self_consistency > 1:
        from backends import SelfConsistency
        backend = SelfConsistency(backend, args.self_consistency)
    if args.cot and args.backend in ("constant", "lexical"):
        print("! --cot has no effect on a non-LLM backend; it is accepted so the "
              "plumbing can be tested, but the numbers will be identical.")
    print(f"backend : {backend.name}   cot={args.cot}")
    print(f"data    : {args.config}, {len(df)} rows -> {len(pairs)} judgements per pass")

    recall = None
    if {"rag", "oracle+rag"} & set(modes):
        recall = attach_rag(df, pairs, args, full_df)

    column = {"none": None, "oracle": "knowledge", "rag": "knowledge_rag",
              "oracle+rag": "knowledge_oracle_rag"}
    results = {}
    for mode in modes:
        started = time.time()
        report, raw = evaluate(backend, pairs, column[mode], args.not_sure, mode, args.cot)
        elapsed = time.time() - started
        results[mode] = report
        print_report(f"{mode}  ({elapsed:.1f}s, {elapsed / max(len(pairs), 1):.2f}s/judgement)",
                     report)
        raw.to_csv(RESULTS_DIR / f"raw_{_slug(backend.name)}"
                   f"{'_cot' if args.cot else ''}_{mode}.csv", index=False)

    if len(modes) > 1:
        print(f"\n  {'condition':<10}{'F1':>8}{'vs none':>10}")
        print("  " + "-" * 28)
        base = results.get("none", {}).get("overall", {}).get("f1")
        for mode in modes:
            f1 = results[mode]["overall"]["f1"]
            delta = f"{f1 - base:+.3f}" if base is not None and mode != "none" else ""
            print(f"  {mode:<10}{f1:>8}{delta:>10}")
        if {"oracle", "rag", "oracle+rag"} <= set(modes) and base is not None:
            o = results["oracle"]["overall"]["f1"]
            orag = results["oracle+rag"]["overall"]["f1"]
            r = results["rag"]["overall"]["f1"]
            print()
            print("  decomposing why rag trails oracle:")
            print(f"    cost of distractors      {orag - o:+.3f}   "
                  "(oracle+rag vs oracle)")
            print(f"    cost of missing passage  {r - orag:+.3f}   "
                  "(rag vs oracle+rag)")

        if {"oracle", "rag"} <= set(modes) and base is not None:
            kept = results["rag"]["overall"]["f1"] - base
            total = results["oracle"]["overall"]["f1"] - base
            # The ratio only means anything when oracle knowledge actually
            # helped. If it hurt, the denominator is negative and the
            # percentage is worse than useless -- it looks like a result.
            if total > 0.01:
                print(f"\n  retrieval captured {100 * kept / total:.0f}% of the "
                      "oracle-knowledge gain")
            else:
                print("\n  oracle knowledge did not help this backend "
                      f"({total:+.3f} F1), so the retrieval ratio is undefined.")
                print("  Expected for the non-LLM baselines -- they cannot read "
                      "context. Use a real judge.")
        print("\n  paper: +0.251 F1 from oracle knowledge, averaged over general LLMs")

    tag = _slug(backend.name) + ("_cot" if args.cot else "")
    summary = RESULTS_DIR / f"summary_{tag}.json"
    payload = {"backend": backend.name, "config": args.config, "rows": len(df),
               "not_sure": args.not_sure, "cot": args.cot,
               "self_consistency": args.self_consistency,
               "modes": modes, "results": results}
    if "rag" in modes:
        payload["rag"] = {"retriever": args.retriever, "k": args.k,
                      "corpus": args.corpus, "recall_at_k": recall}
    summary.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved -> {summary}")


def _slug(text):
    return "".join(c if c.isalnum() else "-" for c in text).strip("-")


if __name__ == "__main__":
    main()
