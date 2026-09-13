"""Paths and environment setup.

Everything is pinned to E: on purpose: C: has very little free space, and the
HuggingFace cache (models) will happily eat several GB if left at its default
location under %USERPROFILE%.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"

for d in (DATA_DIR, RESULTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Redirect model downloads off C: before transformers/sentence-transformers load.
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(CACHE_DIR / "sentence-transformers"))

# The paper's dataset, published by the authors. Public, no token required.
HF_REPO = "UTAustin-AIHealth/MedHallu"
CONFIGS = ("pqa_labeled", "pqa_artificial")

# Column names as published on the Hub.
COL_QUESTION = "Question"
COL_KNOWLEDGE = "Knowledge"
COL_TRUTH = "Ground Truth"
COL_HALLU = "Hallucinated Answer"
COL_DIFFICULTY = "Difficulty Level"
COL_CATEGORY = "Category of Hallucination"
