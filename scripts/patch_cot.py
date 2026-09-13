"""Upgrade the CoT in the patched MedHallu script to Li et al. SS V-A properly.

`patch_original.py` added bare zero-shot CoT. SS V-A describes a family, and the
bare version is only the first member of it. This adds the rest:

  zeroshot   "reason step by step"           Kojima et al. [161]. The baseline.
  fewshot    worked exemplars first          Wei et al. [64]. The original CoT.
  verify     reason, then check each step    Ling et al. [163] Natural Program:
                                             "decomposes CoT reasoning into
                                             multiple sub-processes and performs
                                             strict logical self-verification at
                                             each step"
  plus SELF_CONSISTENCY_N: sample N chains, majority-vote   Wang et al. [164]

Not implemented, deliberately: symbolic CoT (SS V-C). It converts statements to
formal logic and applies rules. Medical claims like "nutritional deficiency
contributes to mortality" do not linearise into symbols without losing the
thing being judged.

The exemplars used by `fewshot` are taken from MedHallu's own Table 1 category
definitions, so they teach the task the benchmark actually poses rather than a
generic reasoning pattern.

Usage (from the repo root):
  python scripts/patch_cot.py /path/to/MedHallu
  python scripts/patch_cot.py /path/to/MedHallu --revert
"""
import argparse
import shutil
import sys
from pathlib import Path

TARGET = Path("Detection/detection_vllm_notsurecase.py")

CONFIG_ADDITION = '''
# CoT variant (Li et al. SS V-A):
#   "zeroshot"  reason step by step                      Kojima et al.
#   "fewshot"   worked exemplars first                   Wei et al.
#   "verify"    reason, then verify each step            Ling et al., Natural Program
COT_MODE = "zeroshot"

# Self-consistency (Wang et al.): sample N chains per judgement and majority-vote.
# 1 disables it. Costs N forward passes per judgement -- on a laptop CPU, 5 turns
# a 3-hour run into 15. Use it on one condition, not all five.
SELF_CONSISTENCY_N = 1
'''

COT_BLOCK = '''

# ---------------------------------------------------------------------------
# Chain-of-Thought variants (Li et al. SS V-A)
# ---------------------------------------------------------------------------

COT_ZEROSHOT = """

Before answering, reason step by step:
1. What exactly is the question asking?
2. What does the world knowledge (if given) actually establish?
3. Does the answer contradict, overreach beyond, or sidestep that?

Then end your reply with a final line in exactly this form:
FINAL: <digit>"""


# Ling et al. [163]: decompose the reasoning, then verify each sub-conclusion
# before committing. The verification pass is the point -- a plausible chain
# that nothing checks is how logic-based hallucinations survive.
COT_VERIFY = """

Answer in two passes.

PASS 1 - reason:
1. What exactly is the question asking?
2. What does the world knowledge (if given) actually establish?
3. What does the answer claim, broken into separate claims?

PASS 2 - verify each claim from step 3:
For each one, state VERIFIED if the world knowledge supports it, UNSUPPORTED if
the knowledge is silent on it, or CONTRADICTED if the knowledge says otherwise.
An answer with any CONTRADICTED claim is hallucinated. An answer built mainly on
UNSUPPORTED claims is probably hallucinated.

Then end your reply with a final line in exactly this form:
FINAL: <digit>"""


# Wei et al. [64]: worked exemplars. Both are drawn from MedHallu's own Table 1
# category definitions, so they demonstrate this task rather than generic
# reasoning. One hallucinated, one factual -- a single-polarity exemplar set
# biases the model toward that label.
COT_FEWSHOT_EXEMPLARS = """

Here are two worked examples.

EXAMPLE 1
Question: What is the primary mechanism of action of aspirin in reducing inflammation?
Answer: Aspirin primarily reduces inflammation by blocking calcium channels in
immune cells, which prevents histamine release and suppresses T-cell activation.
Reasoning:
1. The question asks for aspirin's primary anti-inflammatory mechanism.
2. Established pharmacology: aspirin irreversibly inhibits COX-1 and COX-2,
   reducing prostaglandin synthesis.
3. The answer instead asserts calcium-channel blockade and T-cell suppression.
   That is a different mechanism entirely, and contradicts established knowledge.
   This is Mechanism and Pathway Misattribution.
FINAL: 1

EXAMPLE 2
Question: Does high-dose vitamin C therapy improve survival in sepsis?
Answer: Trial evidence has been mixed, with several randomised trials showing no
significant mortality benefit from high-dose vitamin C in sepsis.
Reasoning:
1. The question asks whether vitamin C improves sepsis survival.
2. The answer reports mixed evidence with no significant mortality benefit.
3. It answers the question that was asked, does not invent a mechanism, and does
   not fabricate specific figures. It states a limitation rather than
   overreaching.
FINAL: 0

Now judge the following the same way."""


def build_cot_suffix(mode):
    """Assemble the CoT instructions for the configured variant."""
    if mode == "fewshot":
        return COT_FEWSHOT_EXEMPLARS + COT_ZEROSHOT
    if mode == "verify":
        return COT_VERIFY
    return COT_ZEROSHOT


def majority_vote(votes):
    """Self-consistency aggregation (Wang et al. [164]).

    Abstentions are excluded when any real verdict exists -- a model answering
    1,1,2 has twice committed to "hallucinated", and letting the abstention
    outvote that discards information. Ties go to "hallucinated": in a clinical
    setting, flagging a true answer for review costs less than passing a
    fabricated one.
    """
    committed = [v for v in votes if v != 2]
    if not committed:
        return 2
    return 1 if committed.count(1) * 2 >= len(committed) else 0

'''


def patch(repo: Path, revert: bool) -> int:
    path = repo / TARGET
    backup = path.with_suffix(".py.precot")

    if revert:
        if not backup.exists():
            print(f"no backup at {backup}")
            return 1
        shutil.copy2(backup, path)
        print(f"reverted from {backup}")
        return 0

    if not path.exists():
        return print(f"not found: {path}") or 1

    source = path.read_text(encoding="utf-8")
    if "COT_MODE" in source:
        print("already CoT-patched. Use --revert first.")
        return 1
    if "COT_SUFFIX" not in source:
        print("run scripts/patch_original.py first -- this builds on it.")
        return 1

    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"backup -> {backup}")

    # config knobs
    source = source.replace("RAG_K = 3", CONFIG_ADDITION + "\nRAG_K = 3", 1)

    # the variants, replacing the single hard-coded suffix
    start = source.index('COT_SUFFIX = """')
    end = source.index('"""', start + len('COT_SUFFIX = """')) + 3
    source = source[:start] + COT_BLOCK.strip() + source[end:]

    # use the configured variant
    source = source.replace(
        'sys_prompt = system_prompt + (COT_SUFFIX if use_cot else "")',
        'sys_prompt = system_prompt + (build_cot_suffix(COT_MODE) if use_cot else "")')

    # self-consistency: sample N and vote
    source = source.replace(
        '''        max_tok = 600 if use_cot else 8
        for chat_prompt in tqdm(prompts, desc=model_config['model_name']):
            try:
                response = client.chat.completions.create(
                    model=model_config['model_name'],
                    messages=chat_prompt,
                    max_tokens=max_tok,
                    n=1,
                    temperature=0.3,
                )
                llm_answers.append(response.choices[0].message.content.strip())''',
        '''        max_tok = 600 if use_cot else 8
        n_samples = max(1, SELF_CONSISTENCY_N)
        # Self-consistency only does anything above temperature 0 -- identical
        # samples cannot disagree.
        temp = 0.7 if n_samples > 1 else 0.3
        for chat_prompt in tqdm(prompts, desc=model_config['model_name']):
            try:
                response = client.chat.completions.create(
                    model=model_config['model_name'],
                    messages=chat_prompt,
                    max_tokens=max_tok,
                    n=n_samples,
                    temperature=temp,
                )
                if n_samples > 1:
                    votes = [parse_reply(c.message.content.strip(), use_cot)
                             for c in response.choices]
                    # Re-encode the voted verdict as text; calculate_metrics
                    # parses strings, so this keeps one code path.
                    llm_answers.append(str(majority_vote(votes)))
                else:
                    llm_answers.append(response.choices[0].message.content.strip())''')

    path.write_text(source, encoding="utf-8")
    print("  ok  CoT variants: zeroshot | fewshot | verify")
    print("  ok  self-consistency voting")
    print(f"\npatched -> {path}")
    print("\nSet COT_MODE and SELF_CONSISTENCY_N in the CONFIGURATION block.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    sys.exit(patch(Path(args.repo), args.revert))
