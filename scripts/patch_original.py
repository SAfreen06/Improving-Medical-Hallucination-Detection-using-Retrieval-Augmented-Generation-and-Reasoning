"""Patch the ORIGINAL MedHallu detection script to run on a laptop via API,
with RAG and CoT added.

Run this once, pointed at your clone of https://github.com/MedHallu/MedHallu.
It edits Detection/detection_vllm_notsurecase.py in place (after taking a
backup) and makes seven changes:

  1  vLLM import moved inside the HF branch.
     As shipped it is at module level, so the script cannot even be imported
     on a machine without vLLM. This is the single blocker for laptop use.

  2  OpenAI branch rewritten for the modern SDK.
     The original calls `openai.ChatCompletion.create`, removed in openai>=1.0.
     Replaced with a client that takes a base_url, so the same code path
     reaches OpenAI, Together, DeepInfra, Featherless or any other
     OpenAI-compatible provider -- including the open models MedHallu itself
     evaluates.

  3  Reply parser fixed.
     The original tests 'not' before 'not sure', so "not sure" scores as
     "hallucinated" and the not-sure branch is unreachable. 'non' also catches
     "non-hallucinated" and inverts it. Also handles CoT replies, where the
     verdict is the LAST digit, not the first.

  4  CoT prompting        (Li et al., "Mitigating Hallucination in LLMs", SS V)
  5  RAG knowledge source (Li et al., SS IV)
  6  A CONDITION switch so one run does none / oracle / rag / cot / rag+cot.
  7  Config block at the top for paths, models, provider and conditions.

Usage (from the repo root):
  python scripts/patch_original.py /path/to/MedHallu
  python scripts/patch_original.py /path/to/MedHallu --revert
"""
import argparse
import shutil
import sys
from pathlib import Path

TARGET = Path("Detection/detection_vllm_notsurecase.py")


# ------------------------------------------------------- 0. header imports
# `datasets` is imported and never used; `torch` only appears in GPU-only code
# paths. Both force heavyweight installs on a laptop that will only call an API.
OLD_HEADER = """import torch
import gc
import os
import random
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
"""

NEW_HEADER = """import gc
import os
import random
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# torch is needed only for the local-GPU path. Importing it unconditionally
# forces a ~250 MB dependency on a laptop that will only ever call an API.
try:
    import torch
except ImportError:
    torch = None

# `from datasets import Dataset` was here. It is never referenced anywhere in
# this file, so it is dropped rather than made optional.
"""


# ---------------------------------------------------------------- 1. imports
OLD_IMPORT = """# vLLM imports
from vllm import LLM, SamplingParams
import torch
import multiprocessing
import ast
"""

NEW_IMPORT = '''import multiprocessing
import ast
import re
import json

# vLLM is imported lazily inside the HF branch. At module level it makes the
# script unimportable on any machine without a CUDA GPU, which is the whole
# reason this patch exists.

# ===================== CONFIGURATION =====================
DF_PATH = "medhallu.csv"          # built by make_original_csv.py
CSV_PATH = "results.csv"

# Conditions to run. Each is (label, knowledge_source, use_cot):
#   knowledge_source: None = no knowledge, "oracle" = the row's own context,
#                     "rag" = top-k retrieved passages
CONDITIONS = [
    ("baseline", None,     False),   # 1. detection as the paper measures it
    ("oracle",   "oracle", False),   # 2. the paper's "with knowledge" ceiling
    ("rag",      "rag",    False),   # 3. real retrieval, external corpus
    ("cot",      None,     True),    # 4. reasoning, no extra knowledge
    ("rag+cot",  "rag",    True),    # 5. both
]
# Optional diagnostic, not a headline result. Add this row to split rag's
# shortfall into distractor cost vs missing-passage cost:
#     ("oracle+rag", "oracle+rag", False),

# Ollama serves an OpenAI-compatible endpoint, so these defaults run the
# models on THIS machine, free, with no account and no key. Ollama ignores the
# key but the SDK requires a non-empty string.
API_KEY = "ollama"
BASE_URL = "http://localhost:11434/v1"

# Rows to evaluate. The paper uses all 10,000; on a laptop CPU that is days.
# 200-300 is enough to see whether a mitigation moves the needle.
LIMIT = 200

RAG_K = 3                          # passages to retrieve
RAG_CORPUS_CSV = None              # defaults to DF_PATH; point elsewhere to add
                                   # distractors, without which retrieval is
                                   # near-perfect and rag collapses onto oracle
# =========================================================
'''


# ------------------------------------------------------------- 2. API client
OLD_OPENAI = """    else:
        # Example for OpenAI calls (if needed)
        import openai
        for chat_prompt in prompts:
            response = openai.ChatCompletion.create(
                model=model_config['model_name'],
                messages=chat_prompt,
                max_tokens=4,
                n=1,
                temperature=0.3,
            )
            content = response.choices[0].message.content.strip()
            llm_answers.append(content)
"""

NEW_OPENAI = '''    else:
        # Modern OpenAI SDK (>=1.0). `openai.ChatCompletion.create` was removed;
        # the original code predates that and raises AttributeError.
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        # CoT needs room to reason; the terse setting would truncate mid-chain.
        max_tok = 600 if use_cot else 8
        for chat_prompt in tqdm(prompts, desc=model_config['model_name']):
            try:
                response = client.chat.completions.create(
                    model=model_config['model_name'],
                    messages=chat_prompt,
                    max_tokens=max_tok,
                    n=1,
                    temperature=0.3,
                )
                llm_answers.append(response.choices[0].message.content.strip())
            except Exception as exc:
                # One bad call should not lose the other 999. An empty reply
                # parses to "not sure", which is the honest thing to record.
                print(f"  api error: {exc}")
                llm_answers.append("")
'''


# ---------------------------------------------------------------- 3. parser
OLD_PARSE = """    llm_answers_int = []
    for i in llm_answers:
        i_lower = i.lower()
        if any(x in i_lower for x in ['1', 'not', 'non']):
            llm_answers_int.append(1)
        elif any(x in i_lower for x in ['not sure', 'pass', 'skip', '2']):
            llm_answers_int.append(2)
        else:
            llm_answers_int.append(0)
"""

NEW_PARSE = '''    llm_answers_int = [parse_reply(i, use_cot) for i in llm_answers]
'''

PARSE_FN = '''

FINAL_RE = re.compile(r"FINAL\\s*:?\\s*([012])", re.IGNORECASE)


def parse_reply(text, use_cot=False):
    """Turn a model reply into 0 (factual) / 1 (hallucinated) / 2 (not sure).

    Replaces the original substring test, which had two inversions:
      - 'not' was checked before 'not sure', so "not sure" scored as
        hallucinated and the not-sure branch was unreachable
      - 'non' caught "non-hallucinated" and scored it hallucinated

    With CoT the verdict is the FINAL: line, or failing that the LAST digit.
    Taking the first digit -- right for terse replies -- is wrong here, because
    the reasoning is full of digits ("step 1", "type 2 diabetes").
    """
    text = (text or "").strip()

    if use_cot:
        m = FINAL_RE.search(text)
        if m:
            return int(m.group(1))
        digits = re.findall(r"[012]", text)
        if digits:
            return int(digits[-1])
    else:
        m = re.search(r"[012]", text)
        if m:
            return int(m.group())

    low = text.lower()
    if any(x in low for x in ["not sure", "unsure", "pass", "skip"]):
        return 2
    if any(x in low for x in ["not hallucinat", "non-hallucinat",
                              "not a hallucinat", "factual", "is correct"]):
        return 0
    if "hallucinat" in low:
        return 1
    return 2
'''


# ------------------------------------------------------------- 4/5. prompts
COT_SUFFIX = '''

Before answering, reason step by step:
1. What exactly is the question asking?
2. What does the world knowledge (if given) actually establish?
3. Does the answer contradict, overreach beyond, or sidestep that?

Then end your reply with a final line in exactly this form:
FINAL: <digit>'''

PROMPT_FNS = '''

COT_SUFFIX = """%s"""


def create_prompt_cot(question, option1, knowledge=None):
    """Chain-of-Thought variant (Li et al. SS V-A, zero-shot CoT)."""
    head = f"World Knowledge: {knowledge}\\n" if knowledge else ""
    return f"""
{head}Question: {question}
Answer: {option1}

Reason step by step, then end with 'FINAL: <digit>' -- '0' if factual, '1' if
hallucinated, '2' if unsure.
Your Judgement:
"""


# ----------------------------- RAG -----------------------------
_RAG_CACHE = {}


def build_rag_index(corpus_csv):
    """TF-IDF index over the knowledge passages. Sparse, but it needs no torch,
    which matters on a laptop. Swap in embeddings if you have the budget.
    """
    if corpus_csv in _RAG_CACHE:
        return _RAG_CACHE[corpus_csv]
    from sklearn.feature_extraction.text import TfidfVectorizer

    frame = pd.read_csv(corpus_csv)
    passages = []
    for value in frame['knowledge']:
        try:
            passages.append(" ".join(ast.literal_eval(value)['contexts']))
        except Exception:
            passages.append(str(value))
    vec = TfidfVectorizer(lowercase=True, stop_words='english', sublinear_tf=True)
    matrix = vec.fit_transform(passages)
    _RAG_CACHE[corpus_csv] = (vec, matrix, passages)
    return _RAG_CACHE[corpus_csv]


def retrieve(question, corpus_csv, k):
    """Top-k passages for a question, joined in rank order."""
    vec, matrix, passages = build_rag_index(corpus_csv)
    scores = (vec.transform([question]) @ matrix.T).toarray()[0]
    top = scores.argsort()[::-1][:k]
    return "\\n\\n".join(passages[j] for j in top)
''' % COT_SUFFIX


def patch(repo: Path, revert: bool) -> int:
    path = repo / TARGET
    backup = path.with_suffix(".py.orig")

    if revert:
        if not backup.exists():
            print(f"no backup at {backup}")
            return 1
        shutil.copy2(backup, path)
        print(f"reverted {path} from {backup}")
        return 0

    if not path.exists():
        print(f"not found: {path}\nPass the root of your MedHallu clone.")
        return 1

    source = path.read_text(encoding="utf-8")
    if "===================== CONFIGURATION" in source:
        print("already patched. Use --revert first if you want to redo it.")
        return 1

    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"backup -> {backup}")

    steps = [
        ("0 drop dead/GPU-only header imports", OLD_HEADER, NEW_HEADER),
        ("1 lazy vLLM import + config block", OLD_IMPORT, NEW_IMPORT),
        ("2 modern OpenAI client", OLD_OPENAI, NEW_OPENAI),
        ("3 reply parser", OLD_PARSE, NEW_PARSE),
    ]
    for label, old, new in steps:
        if old not in source:
            print(f"FAILED at step {label}: anchor text not found.")
            print("The upstream file has changed. Patch it by hand, or open an issue.")
            return 1
        source = source.replace(old, new, 1)
        print(f"  ok  {label}")

    # clear_gpu_memory must tolerate torch being absent
    source = source.replace(
        "def clear_gpu_memory():\n    if torch.cuda.is_available():",
        "def clear_gpu_memory():\n"
        "    if torch is None:   # API-only run; nothing on a GPU to free\n"
        "        return\n"
        "    if torch.cuda.is_available():", 1)

    # vLLM import, now inside the branch that uses it
    source = source.replace(
        "    if model_config['type'] == 'hf':\n        # Initialize vLLM model",
        "    if model_config['type'] == 'hf':\n"
        "        from vllm import LLM, SamplingParams  # local GPU only\n"
        "        # Initialize vLLM model", 1)

    # helper functions after the prompt builders
    source = source.replace(
        "# ---------------------\n# GPU MEMORY CLEARING",
        PARSE_FN + PROMPT_FNS + "\n\n# ---------------------\n# GPU MEMORY CLEARING", 1)

    # thread use_cot through the functions that need it
    source = source.replace(
        "def calculate_metrics(answer_list, llm_answers, df, model_config, use_knowledge):",
        "def calculate_metrics(answer_list, llm_answers, df, model_config, use_knowledge, use_cot=False):")
    source = source.replace(
        "def run_evaluation(model_config, df, use_knowledge=False):",
        "def run_evaluation(model_config, df, use_knowledge=False, use_cot=False):")
    source = source.replace(
        "    result_df = calculate_metrics(answer_list, llm_answers, df, model_config, use_knowledge)",
        "    result_df = calculate_metrics(answer_list, llm_answers, df, model_config, use_knowledge, use_cot)")

    # knowledge source: none / oracle / rag, and the CoT prompt
    source = source.replace(
        """        if use_knowledge:
            # Assuming the 'knowledge' column is stored as a string representation of a dict.
            try:
                knowledge = ast.literal_eval(df.loc[i, 'knowledge'])['contexts']
            except Exception as e:
                knowledge = ""
        else:
            knowledge = None""",
        """        if use_knowledge == "oracle":
            try:
                knowledge = ast.literal_eval(df.loc[i, 'knowledge'])['contexts']
            except Exception:
                knowledge = ""
        elif use_knowledge == "rag":
            knowledge = retrieve(question, RAG_CORPUS_CSV or DF_PATH, RAG_K)
        else:
            knowledge = None""")

    source = source.replace(
        """        if use_knowledge:
            user_prompt = create_prompt_withknowledge(question, chosen, knowledge)
        else:
            user_prompt = create_prompt(question, chosen)""",
        """        if use_cot:
            user_prompt = create_prompt_cot(question, chosen, knowledge)
        elif knowledge:
            user_prompt = create_prompt_withknowledge(question, chosen, knowledge)
        else:
            user_prompt = create_prompt(question, chosen)""")

    source = source.replace(
        '        prompt_chat = [\n            # {"role": "system", "content": system_prompt},\n'
        '            {"role": "user", "content": f"{system_prompt} {user_prompt}"},\n        ]',
        '        sys_prompt = system_prompt + (COT_SUFFIX if use_cot else "")\n'
        '        prompt_chat = [\n'
        '            {"role": "user", "content": f"{sys_prompt} {user_prompt}"},\n        ]')

    # subprocess wrapper carries the condition
    source = source.replace(
        "def evaluate_model_subprocess(model_config, use_knowledge, df_path, csv_path):",
        "def evaluate_model_subprocess(model_config, use_knowledge, df_path, csv_path, use_cot=False, label=''):")
    source = source.replace(
        "        result = run_evaluation(model_config, df, use_knowledge)",
        "        result = run_evaluation(model_config, df, use_knowledge, use_cot)\n"
        "        result['Condition'] = label")

    # main loop over conditions
    source = source.replace(
        """    for model_config in models:
        for use_knowledge in [False, True]:
            proc = multiprocessing.Process(
                target=evaluate_model_subprocess,
                args=(model_config, use_knowledge, df_path, csv_path)
            )""",
        """    for model_config in models:
        for label, knowledge_source, use_cot in CONDITIONS:
            print(f"\\n=== {model_config['model_name']} | {label} ===")
            proc = multiprocessing.Process(
                target=evaluate_model_subprocess,
                args=(model_config, knowledge_source, df_path, csv_path, use_cot, label)
            )""")
    source = source.replace(
        '            print(f"Completed {model_config[\'model_name\']} with knowledge = {use_knowledge}\\n")',
        '            print(f"Completed {model_config[\'model_name\']} | {label}\\n")')

    # wire the config block into main()
    source = source.replace('    df_path = " "\n    csv_path = " "',
                            '    df_path = DF_PATH\n    csv_path = CSV_PATH')

    path.write_text(source, encoding="utf-8")
    print(f"  ok  4-7 CoT, RAG, conditions, config")
    print(f"\npatched -> {path}")
    print("\nNext:")
    print("  1. edit the CONFIGURATION block at the top (paths, API key, base_url)")
    print("  2. set the `models` list to the 2-3 models you are testing")
    print("  3. build the CSV:  python make_original_csv.py --config pqa_labeled")
    print("  4. run:            python Detection/detection_vllm_notsurecase.py")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="root of your MedHallu clone")
    ap.add_argument("--revert", action="store_true", help="restore from the .orig backup")
    args = ap.parse_args()
    sys.exit(patch(Path(args.repo), args.revert))
