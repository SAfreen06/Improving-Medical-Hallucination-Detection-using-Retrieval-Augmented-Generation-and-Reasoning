"""Shared plumbing for the Streamlit frontend.

The project's own code lives in src/ and scripts/ as bare scripts (no
package, no __init__.py, imports like `import data`). Rather than touch
that layout, we put both directories on sys.path once, here, and every
page imports this module first.
"""
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
OUTPUTS = ROOT / "outputs"

for p in (SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import data  # noqa: E402
import backends  # noqa: E402
from config import CONFIGS  # noqa: E402
from compare_to_paper import TABLE2, paper_row  # noqa: E402

VERDICT_LABEL = {0: "factual", 1: "hallucinated", 2: "not sure"}
VERDICT_COLOR = {0: "green", 1: "red", 2: "orange"}


@st.cache_data(show_spinner="Downloading MedHallu (~11 MB, cached after first run)...")
def load_dataset(config: str):
    return data.load(config)


@st.cache_data
def load_outputs_csv(name: str):
    import pandas as pd
    path = OUTPUTS / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def ollama_status(host: str = "http://localhost:11434"):
    """Return (reachable, [model names]). Never raises."""
    import requests
    try:
        r = requests.get(f"{host}/api/tags", timeout=2)
        r.raise_for_status()
        return True, [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return False, []


def anthropic_available():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def openai_available():
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


def judge_verbose(backend, question, answer, knowledge, allow_not_sure, cot):
    """Like backend.judge(), but also returns the raw model text when the
    backend actually calls a model -- useful to show the CoT reasoning or to
    audit a bare verdict. constant/lexical have no raw text to show.
    """
    import requests

    if isinstance(backend, backends.OllamaBackend):
        payload = {
            "model": backend.model,
            "messages": [
                {"role": "system", "content": backends.build_system_prompt(allow_not_sure, cot)},
                {"role": "user", "content": backends.build_user_prompt(
                    question, answer, knowledge, allow_not_sure, cot)},
            ],
            "stream": False,
            "options": {"temperature": backend.temperature, "num_predict": 400 if cot else 8},
        }
        r = requests.post(f"{backend.host}/api/chat", json=payload, timeout=backend.timeout)
        r.raise_for_status()
        text = r.json()["message"]["content"]
        return backends.parse_judgement(text, allow_not_sure, cot), text

    if isinstance(backend, backends.AnthropicBackend):
        resp = backend.client.messages.create(
            model=backend.model, max_tokens=600 if cot else 8, temperature=backend.temperature,
            system=backends.build_system_prompt(allow_not_sure, cot),
            messages=[{"role": "user", "content": backends.build_user_prompt(
                question, answer, knowledge, allow_not_sure, cot)}])
        text = "".join(b.text for b in resp.content if b.type == "text")
        return backends.parse_judgement(text, allow_not_sure, cot), text

    if isinstance(backend, backends.OpenAIBackend):
        resp = backend.client.chat.completions.create(
            model=backend.model, max_tokens=600 if cot else 8, temperature=backend.temperature,
            messages=[
                {"role": "system", "content": backends.build_system_prompt(allow_not_sure, cot)},
                {"role": "user", "content": backends.build_user_prompt(
                    question, answer, knowledge, allow_not_sure, cot)},
            ])
        text = resp.choices[0].message.content
        return backends.parse_judgement(text, allow_not_sure, cot), text

    return backend.judge(question, answer, knowledge, allow_not_sure, cot), None


def verdict_badge(verdict: int) -> str:
    label = VERDICT_LABEL.get(verdict, "?")
    color = VERDICT_COLOR.get(verdict, "gray")
    return f":{color}[**{verdict} -- {label}**]"


def expected_f1(model_tag: str, condition: str):
    """Look up the measured F1 for this model/condition in outputs/results_mitigations.csv,
    so the Live Judge page can tell you *before* you run it how often to expect a 'wrong'
    verdict -- instead of a wrong verdict looking like a bug.
    """
    df = load_outputs_csv("results_mitigations.csv")
    if df is None:
        return None
    match = df[(df["Model Name"].astype(str).str.lower() == model_tag.lower())
              & (df["Condition"].astype(str) == condition)]
    if match.empty:
        return None
    return float(match.iloc[0]["f1"])


def condition_for(know_choice: str, cot: bool) -> str | None:
    """Map a Live Judge knowledge-mode choice + CoT toggle to one of this
    project's five named conditions, or None if there's no matching row
    (e.g. 'Custom text' knowledge isn't one of the five)."""
    if know_choice.startswith("None"):
        return "cot" if cot else "baseline"
    if know_choice.startswith("Oracle"):
        return "oracle"
    if know_choice.startswith("RAG"):
        return "rag+cot" if cot else "rag"
    return None


@st.cache_resource(show_spinner=False)
def get_corpus_and_retriever(corpus_name: str, config_for_self: str = "pqa_labeled",
                             max_docs: int = 0):
    """Build (or load the cached) corpus + a TF-IDF retriever over it.

    Shared by the Live Judge (RAG knowledge mode) and RAG Inspector pages so a
    corpus is only ever downloaded/indexed once per session, regardless of
    which page asked for it first. 'self' ignores max_docs -- it's built from
    the dataset's own knowledge column, not downloaded.
    """
    import rag
    corpus_df = load_dataset(config_for_self) if corpus_name == "self" else None
    corpus = rag.get_corpus(corpus_name, corpus_df, max_docs)
    retriever = rag.get_retriever("tfidf", corpus)
    return corpus, retriever
