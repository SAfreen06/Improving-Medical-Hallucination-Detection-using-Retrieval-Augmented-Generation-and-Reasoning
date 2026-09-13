"""Walk through the dataset-generation pipeline (paper Figure 2) on a few rows.

This is the part that genuinely needs the GPUs. At full scale the paper runs
Qwen2.5-14B as generator plus three discriminator LLMs plus TextGrad plus an NLI
model, for 26.5 hours on 4x A6000, to produce 10,000 rows.

The point of this script is not to reproduce that. It is to make the control
flow concrete: you watch one question go through generate -> quality vote ->
entailment check -> difficulty label, and the difficulty label stops being a
mystery column in a CSV.

Two modes:

  --dry-run   No model needed. Replays the pipeline over rows the authors
              already generated, using the released hallucination as the
              "candidate" and simulating the discriminator vote from the
              published difficulty label. Shows the shape of the pipeline in
              about a second. Start here.

  live        Needs a backend that can generate text (ollama / anthropic /
              openai). Actually generates a fresh hallucinated answer for each
              question, then puts it through a real discriminator vote. This is
              the honest small-scale version.

Examples:
  python src/generate_demo.py --dry-run --n 3
  python src/generate_demo.py --backend ollama:qwen2.5:1.5b-instruct --n 2
"""
import argparse
import textwrap

import data
from backends import get_backend
from config import COL_DIFFICULTY, COL_HALLU, COL_KNOWLEDGE, COL_QUESTION, COL_TRUTH
from semantics import rouge1_f1

# Condensed from Dataset Generation/Prompts/system_prompt.txt in the official repo.
GENERATOR_PROMPT = """You are a hallucinated answer generator for building a medical hallucination benchmark.

Given a #Question#, supporting #Knowledge#, and the #Ground Truth Answer#, write ONE hallucinated answer that is plausible and fluent but factually wrong. Use exactly one of these strategies:

1. Misinterpretation of Question - answer a subtly different question.
2. Incomplete Information - stay on topic but omit the essential detail.
3. Mechanism and Pathway Misattribution - attribute a false biological mechanism.
4. Methodological and Evidence Fabrication - invent methods, statistics or outcomes.

Constraints:
- Keep the length within 10% of the ground truth answer.
- Do not signal uncertainty, and do not mention that the answer is wrong.
- Return only the hallucinated answer text, nothing else.
"""

# The three-model ensemble the paper uses for the quality vote.
PAPER_DISCRIMINATORS = ["GPT-4o-mini", "Gemma2-9B", "Qwen2.5-7B"]


def difficulty_from_votes(n_fooled: int, n_judges: int) -> str:
    """The paper's labelling rule (Sec 3, step 2).

    A candidate survives only if it fools at least one judge; how many it fools
    sets the difficulty. Fooling every judge is 'hard'.
    """
    if n_fooled == 0:
        return "rejected"
    if n_fooled == n_judges:
        return "hard"
    if n_fooled == 1:
        return "easy"
    return "medium"


def votes_from_difficulty(difficulty: str, n_judges: int) -> int:
    """Inverse of the rule above, for --dry-run. Lossy for 'medium'."""
    return {"easy": 1, "medium": max(2, n_judges - 1), "hard": n_judges}.get(difficulty, 0)


def wrap(text, width=76, indent="      "):
    return textwrap.fill(str(text), width=width, initial_indent=indent,
                         subsequent_indent=indent)


def judge_which_is_correct(backend, question, candidate, truth):
    """One discriminator's quality vote.

    The paper shows a judge both answers WITHOUT the supporting knowledge and
    asks which is correct. 'Fooled' means it picked the hallucination. Note the
    judge never sees the knowledge here -- that is what makes fooling it
    possible, and it is why the same models score far better in the detection
    task when knowledge is supplied.
    """
    verdict = backend.judge(question=question, answer=candidate,
                            knowledge=None, allow_not_sure=False)
    return verdict == 0  # judged the hallucination "factual" => fooled


def run_row(row, idx, backend, dry_run, n_judges):
    print(f"\n{'=' * 78}\nROW {idx}\n{'=' * 78}")
    question = row[COL_QUESTION]
    truth = str(row[COL_TRUTH])
    knowledge = data._flatten_knowledge(row[COL_KNOWLEDGE])

    print("\n  QUESTION")
    print(wrap(question))
    print("\n  KNOWLEDGE (context given to the generator, withheld from judges)")
    print(wrap(knowledge[:400] + ("..." if len(knowledge) > 400 else "")))
    print("\n  GROUND TRUTH")
    print(wrap(truth[:400] + ("..." if len(truth) > 400 else "")))

    # ---- Step 1: candidate generation -------------------------------------
    print(f"\n  [1] CANDIDATE GENERATION")
    if dry_run:
        candidate = str(row[COL_HALLU])
        print("      (dry run: replaying the authors' released candidate)")
    else:
        candidate = generate_candidate(backend, question, knowledge, truth)
    print(wrap(candidate))

    # ---- Step 2: quality vote --------------------------------------------
    print(f"\n  [2] QUALITY VOTE  ({n_judges} judges, no knowledge provided)")
    if dry_run:
        n_fooled = votes_from_difficulty(str(row[COL_DIFFICULTY]), n_judges)
        print(f"      (dry run: inferred {n_fooled}/{n_judges} fooled from the "
              f"published label '{row[COL_DIFFICULTY]}')")
        print(f"      paper's real ensemble: {', '.join(PAPER_DISCRIMINATORS)}")
    else:
        n_fooled = 0
        for i in range(n_judges):
            fooled = judge_which_is_correct(backend, question, candidate, truth)
            n_fooled += fooled
            print(f"      judge {i + 1}: {'FOOLED' if fooled else 'caught it'}")
        print(f"      {n_fooled}/{n_judges} fooled")

    # ---- Step 3: correctness check ---------------------------------------
    # The paper uses bidirectional NLI entailment with deberta-large-mnli: if
    # each answer entails the other they mean the same thing, so the candidate
    # is not actually a hallucination and gets rejected. Unigram overlap is a
    # crude stand-in that needs no model; the real check is in semantics.py.
    overlap = rouge1_f1(candidate, truth)
    same_meaning = overlap > 0.75
    print(f"\n  [3] CORRECTNESS CHECK")
    print(f"      rouge1 vs ground truth: {overlap:.3f}"
          f"  ->  {'TOO SIMILAR, reject' if same_meaning else 'distinct enough, keep'}")
    print("      (paper uses bidirectional NLI entailment, threshold 0.75)")

    # ---- Step 4: label ---------------------------------------------------
    label = "rejected" if same_meaning else difficulty_from_votes(n_fooled, n_judges)
    print(f"\n  [4] LABEL -> {label.upper()}")
    if label == "rejected":
        print("      Paper's next move: refine via TextGrad, then regenerate,")
        print("      up to 5 attempts. If all fail, fall back to the candidate")
        print("      most similar to the ground truth and label it EASY.")
    if dry_run:
        print(f"      published label: {row[COL_DIFFICULTY]}")
    return label


def generate_candidate(backend, question, knowledge, truth):
    """Ask the backend for one hallucinated answer.

    Backends expose `judge` for the benchmark; generation needs raw completion,
    so reach past that here.
    """
    user = (f"#Question#: {question}\n#Knowledge#: {knowledge}\n"
            f"#Ground Truth Answer#: {truth}\n\n#Hallucinated Answer#:")

    if hasattr(backend, "client") and backend.name.startswith("anthropic"):
        resp = backend.client.messages.create(
            model=backend.model, max_tokens=400, temperature=0.7,
            system=GENERATOR_PROMPT, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    if hasattr(backend, "client"):  # openai
        resp = backend.client.chat.completions.create(
            model=backend.model, max_tokens=400, temperature=0.7,
            messages=[{"role": "system", "content": GENERATOR_PROMPT},
                      {"role": "user", "content": user}])
        return resp.choices[0].message.content.strip()

    if backend.name.startswith("ollama"):
        import requests
        r = requests.post(f"{backend.host}/api/chat", timeout=backend.timeout, json={
            "model": backend.model, "stream": False,
            "options": {"temperature": 0.7, "top_p": 0.95, "num_predict": 400},
            "messages": [{"role": "system", "content": GENERATOR_PROMPT},
                         {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    raise SystemExit(f"backend {backend.name!r} cannot generate text; use --dry-run")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="replay released rows, no model required")
    ap.add_argument("--backend", default="ollama:qwen2.5:1.5b-instruct")
    ap.add_argument("--n", type=int, default=3, help="rows to walk through")
    ap.add_argument("--judges", type=int, default=3,
                    help="discriminator votes per candidate (paper uses 3)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = data.load("pqa_labeled").sample(n=args.n, random_state=args.seed)
    backend = None if args.dry_run else get_backend(args.backend)

    print(f"mode    : {'dry run (no model)' if args.dry_run else backend.name}")
    print(f"rows    : {args.n}   judges: {args.judges}")

    labels = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        labels.append(run_row(row, i, backend, args.dry_run, args.judges))

    print(f"\n{'=' * 78}")
    print("labels produced:", ", ".join(labels))
    print(textwrap.dedent("""
        Scale check: this pipeline, run to 10,000 accepted rows with a 14B
        generator and three real judges, is the 26.5 GPU-hours on 4x A6000
        reported in the paper's Table 6. The released dataset is that output --
        which is why detect.py consumes it instead of rebuilding it.
    """).rstrip())


if __name__ == "__main__":
    main()
