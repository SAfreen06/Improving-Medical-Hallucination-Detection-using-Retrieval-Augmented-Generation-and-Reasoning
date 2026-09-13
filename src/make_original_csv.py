"""Convert the published MedHallu dataset into the CSV the ORIGINAL repo expects.

You need this because the official Detection script cannot read the official
published dataset. Their README tells you to load it from HuggingFace, but
`detection_vllm_notsurecase.py` reads different column names off a local CSV:

    script wants          HuggingFace release has
    --------------------  -----------------------
    question              Question
    ground_truth          Ground Truth
    least_similar_answer  Hallucinated Answer
    knowledge             Knowledge
    final_difficulty_level Difficulty Level

It also does `ast.literal_eval(df.loc[i, 'knowledge'])['contexts']`, so the
knowledge column has to be a *stringified dict* with a 'contexts' key, not the
plain list the release ships.

Run this on any machine (no GPU needed), copy the CSV to your GPU box, and
point `df_path` at it.

Usage:
  python src/make_original_csv.py --config pqa_labeled
  python src/make_original_csv.py --config pqa_artificial -o /path/to/medhallu.csv
"""
import argparse

import data
from config import (COL_CATEGORY, COL_DIFFICULTY, COL_HALLU, COL_KNOWLEDGE,
                    COL_QUESTION, COL_TRUTH, DATA_DIR)


def to_contexts_dict(value) -> str:
    """Wrap the knowledge list as the stringified dict the original code parses."""
    if isinstance(value, str):
        contexts = [value]
    else:
        contexts = [str(v) for v in value]
    return repr({"contexts": contexts})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pqa_labeled",
                    choices=("pqa_labeled", "pqa_artificial"))
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="keep only the first N rows; use ~20 for a smoke test")
    args = ap.parse_args()

    src = data.load(args.config)
    if args.limit:
        src = src.head(args.limit)
    out = src.rename(columns={
        COL_QUESTION: "question",
        COL_TRUTH: "ground_truth",
        COL_HALLU: "least_similar_answer",
        COL_DIFFICULTY: "final_difficulty_level",
        COL_CATEGORY: "category_of_hallucination",
    })
    out["knowledge"] = src[COL_KNOWLEDGE].map(to_contexts_dict)

    required = ["question", "ground_truth", "least_similar_answer",
                "knowledge", "final_difficulty_level"]
    out = out[required + ["category_of_hallucination"]]

    suffix = f"_first{args.limit}" if args.limit else ""
    path = args.output or (DATA_DIR / f"original_format_{args.config}{suffix}.csv")
    out.to_csv(path, index=False)

    print(f"wrote {len(out)} rows -> {path}")
    print("\ncolumns:", list(out.columns))
    print("\nPoint the original repo at it:")
    print(f'  detection_vllm_notsurecase.py line 293:  df_path = "{path}"')
    print('  detection_vllm_notsurecase.py line 294:  csv_path = "results.csv"')


if __name__ == "__main__":
    main()
