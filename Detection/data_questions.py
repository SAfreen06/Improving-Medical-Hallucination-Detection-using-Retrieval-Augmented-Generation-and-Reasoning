"""Pull MedHallu questions, for calibrating retrieval thresholds.

Thresholds have to be set from the queries you will actually run, not from a
handful typed by hand -- retrieval scores depend on question phrasing, and
MedHallu questions are paper titles, which score differently from natural
questions.
"""
import functools
import io

import pandas as pd
import requests

HF_REPO = "UTAustin-AIHealth/MedHallu"


@functools.lru_cache(maxsize=4)
def load_split(config="pqa_labeled"):
    index = requests.get(
        f"https://huggingface.co/api/datasets/{HF_REPO}/parquet", timeout=120).json()
    urls = index[config]["train"]
    frames = []
    for url in urls:
        resp = requests.get(url, timeout=600)
        resp.raise_for_status()
        frames.append(pd.read_parquet(io.BytesIO(resp.content)))
    return pd.concat(frames, ignore_index=True)


def sample_questions(n=30, config="pqa_labeled", seed=0):
    df = load_split(config)
    if n and n < len(df):
        df = df.sample(n=n, random_state=seed)
    return df["Question"].astype(str).tolist()
