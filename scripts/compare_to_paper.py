"""Put your results next to MedHallu's published Table 2.

The point of using MedHallu as a base is that the baselines already exist --
you do not have to re-run them. This file holds Table 2 verbatim and joins your
own results against it, so the only question left is whether RAG / CoT / RAG+CoT
beat what the paper reports for the same model.

Usage (from the repo root):
  python scripts/compare_to_paper.py outputs/results.csv
  python scripts/compare_to_paper.py outputs/results.csv --model qwen2.5:7b-instruct
"""
import argparse
import sys

import pandas as pd

# --------------------------------------------------------------------------
# MedHallu paper, Table 2 (10,000 samples). Columns are
#   f1 / precision / easy_f1 / medium_f1 / hard_f1
# in the no-knowledge setting, then the same with oracle knowledge, then the
# knowledge delta. Transcribed from arXiv:2502.14302.
# --------------------------------------------------------------------------
COLUMNS = ["f1", "p", "easy_f1", "med_f1", "hard_f1",
           "k_f1", "k_p", "k_easy_f1", "k_med_f1", "k_hard_f1", "delta"]

TABLE2 = {
    "GPT-4o":                       [0.737, 0.723, 0.844, 0.758, 0.625, 0.877, 0.882, 0.947, 0.880, 0.811,  0.140],
    "GPT-4o-mini":                  [0.607, 0.772, 0.783, 0.603, 0.446, 0.841, 0.820, 0.914, 0.854, 0.761,  0.234],
    "Qwen2.5-14B-Instruct":         [0.619, 0.691, 0.773, 0.611, 0.483, 0.852, 0.857, 0.935, 0.856, 0.769,  0.233],
    "Gemma-2-9b-Instruct":          [0.515, 0.740, 0.693, 0.512, 0.347, 0.838, 0.809, 0.918, 0.848, 0.758,  0.323],
    "Llama-3.1-8B-Instruct":        [0.522, 0.791, 0.679, 0.515, 0.372, 0.797, 0.775, 0.880, 0.796, 0.722,  0.275],
    "DeepSeek-R1-Distill-Llama-8B": [0.514, 0.570, 0.589, 0.515, 0.444, 0.812, 0.864, 0.895, 0.794, 0.751,  0.298],
    "Qwen2.5-7B-Instruct":          [0.553, 0.745, 0.733, 0.528, 0.402, 0.839, 0.866, 0.923, 0.832, 0.770,  0.286],
    "Qwen2.5-3B-Instruct":          [0.606, 0.495, 0.667, 0.602, 0.556, 0.676, 0.514, 0.693, 0.677, 0.661,  0.070],
    "Llama-3.2-3B-Instruct":        [0.499, 0.696, 0.651, 0.467, 0.384, 0.734, 0.775, 0.822, 0.723, 0.664,  0.235],
    "Gemma-2-2b-Instruct":          [0.553, 0.620, 0.680, 0.524, 0.457, 0.715, 0.786, 0.812, 0.705, 0.631,  0.162],
    "OpenBioLLM-Llama3-8B":         [0.484, 0.490, 0.494, 0.474, 0.483, 0.424, 0.567, 0.438, 0.412, 0.423, -0.060],
    "BioMistral-7B":                [0.570, 0.518, 0.627, 0.563, 0.525, 0.648, 0.516, 0.652, 0.660, 0.634,  0.078],
    "Llama-3.1-8B-UltraMedical":    [0.619, 0.657, 0.747, 0.596, 0.524, 0.773, 0.679, 0.832, 0.777, 0.718,  0.153],
    "Llama3-Med42-8B":              [0.416, 0.829, 0.600, 0.379, 0.264, 0.797, 0.856, 0.898, 0.794, 0.707,  0.381],
}

# Ollama tag -> paper row. Ollama names differ from the HuggingFace ids the
# paper uses, so the join needs this.
OLLAMA_TO_PAPER = {
    "qwen2.5:1.5b-instruct": "Qwen2.5-3B-Instruct",
    "qwen2.5:1.5b": "Qwen2.5-3B-Instruct",
    "qwen2.5-1.5b-instruct": "Qwen2.5-3B-Instruct",
    "qwen2.5:3b-instruct": "Qwen2.5-3B-Instruct",
    "qwen2.5:3b": "Qwen2.5-3B-Instruct",
    "qwen2.5:7b-instruct": "Qwen2.5-7B-Instruct",
    "qwen2.5:7b": "Qwen2.5-7B-Instruct",
    "qwen2.5:14b-instruct": "Qwen2.5-14B-Instruct",
    "qwen2.5:14b": "Qwen2.5-14B-Instruct",
    "llama3.2:3b": "Llama-3.2-3B-Instruct",
    "llama3.2:3b-instruct": "Llama-3.2-3B-Instruct",
    "llama3.1:8b": "Llama-3.1-8B-Instruct",
    "llama3.1:8b-instruct": "Llama-3.1-8B-Instruct",
    "gemma2:2b": "Gemma-2-2b-Instruct",
    "gemma2:9b": "Gemma-2-9b-Instruct",
    "deepseek-r1:8b": "DeepSeek-R1-Distill-Llama-8B",
    "biomistral": "BioMistral-7B",
    "biomistral:7b": "BioMistral-7B",
    "openbiollm": "OpenBioLLM-Llama3-8B",
    "med42": "Llama3-Med42-8B",
    "ultramedical": "Llama-3.1-8B-UltraMedical",
}


def paper_row(model_name):
    """Resolve a model name -- ollama tag, HF id or paper name -- to Table 2."""
    key = str(model_name).strip()
    if key in TABLE2:
        return key, TABLE2[key]
    low = key.lower()
    if low in OLLAMA_TO_PAPER:
        name = OLLAMA_TO_PAPER[low]
        return name, TABLE2[name]
    short = key.split("/")[-1]          # HF id -> bare model name
    if short.lower() in OLLAMA_TO_PAPER:
        name = OLLAMA_TO_PAPER[short.lower()]
        return name, TABLE2[name]
    for name in TABLE2:
        if name.lower() == short.lower():
            return name, TABLE2[name]
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="the CSV written by detection_vllm_notsurecase.py")
    ap.add_argument("--model", default=None, help="restrict to one model")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    if "Condition" not in df.columns:
        print("! no 'Condition' column -- apply scripts/patch_original.py first, or this")
        print("  CSV came from an unpatched run.")
    if args.model:
        df = df[df["Model Name"].astype(str).str.contains(args.model, case=False)]
    if df.empty:
        sys.exit("no rows to compare")

    for model, group in df.groupby("Model Name"):
        name, base = paper_row(model)
        print(f"\n{'=' * 74}\n{model}")
        if base is None:
            print("  not in MedHallu Table 2 -- no published baseline to compare against")
            continue
        print(f"paper row: {name}\n{'=' * 74}")

        ref = dict(zip(COLUMNS, base))
        print(f"  {'condition':<12}{'F1':>8}{'P':>8}{'easy':>8}{'med':>8}{'hard':>8}"
              f"{'vs paper':>11}")
        print("  " + "-" * 63)

        # The paper's two rows are the reference, not results of yours.
        print(f"  {'PAPER none':<12}{ref['f1']:>8.3f}{ref['p']:>8.3f}"
              f"{ref['easy_f1']:>8.3f}{ref['med_f1']:>8.3f}{ref['hard_f1']:>8.3f}"
              f"{'baseline':>11}")
        print(f"  {'PAPER oracle':<12}{ref['k_f1']:>8.3f}{ref['k_p']:>8.3f}"
              f"{ref['k_easy_f1']:>8.3f}{ref['k_med_f1']:>8.3f}{ref['k_hard_f1']:>8.3f}"
              f"{ref['delta']:>+11.3f}")
        print("  " + "-" * 63)

        order = ["baseline", "oracle", "rag", "cot", "rag+cot"]
        rows = {str(r.get("Condition", "?")): r for _, r in group.iterrows()}
        for cond in order + [c for c in rows if c not in order]:
            r = rows.get(cond)
            if r is None:
                continue
            f1 = float(r.get("f1", float("nan")))
            # Compare like with like: knowledge-using conditions against the
            # paper's oracle row, knowledge-free ones against its no-knowledge
            # row. Comparing RAG to the no-knowledge baseline would overstate it.
            against = ref["k_f1"] if cond in ("oracle", "rag", "rag+cot") else ref["f1"]
            print(f"  {cond:<12}{f1:>8.3f}{float(r.get('precision', float('nan'))):>8.3f}"
                  f"{float(r.get('easy_f1', float('nan'))):>8.3f}"
                  f"{float(r.get('medium_f1', float('nan'))):>8.3f}"
                  f"{float(r.get('hard_f1', float('nan'))):>8.3f}"
                  f"{f1 - against:>+11.3f}")

        print("\n  Reading it:")
        print("   cot      vs PAPER none   -- does reasoning alone beat the baseline?")
        print("   rag      vs PAPER oracle -- how much of perfect knowledge does")
        print("                               retrieval recover?")
        print("   rag+cot  vs PAPER oracle -- do they add up, or overlap?")

    print(f"\n{'=' * 74}")
    print("Caveats to state when you report this:")
    print("  - The paper runs 10,000 samples at fp16. A few hundred rows of a")
    print("    Q4-quantised model is not the same measurement. Differences under")
    print("    roughly 0.05 F1 are noise at that sample size.")
    print("  - The paper samples ONE answer per row at random; check whether your")
    print("    run does the same before calling any gap a real effect.")


if __name__ == "__main__":
    main()
