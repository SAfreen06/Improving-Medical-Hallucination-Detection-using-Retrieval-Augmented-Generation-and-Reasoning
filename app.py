"""MedHallu mitigation -- Streamlit frontend.

Run with:
    streamlit run app.py
"""
import streamlit as st

from app_lib.common import ROOT, load_outputs_csv, ollama_status

st.set_page_config(page_title="MedHallu Mitigation", layout="wide")

st.title("MedHallu Mitigation")
st.caption("Does RAG, Chain-of-Thought, or both improve hallucination *detection* in medical QA?")

st.markdown(
    """
MedHallu benchmarks how well LLMs spot fabricated medical answers, and finds they are poor at it.
This project adds retrieval (RAG) and reasoning (CoT) mitigations from Li et al. and measures
whether either one actually helps the judge tell a hallucinated answer from a factual one.

**Use the pages in the sidebar:**

- **Results Dashboard** -- the five-condition comparison (baseline / oracle / rag / cot / rag+cot),
  sliced by difficulty and by knowledge-vs-logic hallucination type, next to the paper's published numbers.
- **Dataset Explorer** -- browse MedHallu itself: questions, ground truth, hallucinated answers,
  difficulty and category.
- **Live Judge** -- run a real judge backend (constant / lexical / a local Ollama model) on a
  question and answer, with or without knowledge, with or without chain-of-thought, and see the
  verdict and reasoning.
- **RAG Inspector** -- pick a question and see what a retriever actually pulls back from an
  external medical corpus, next to the oracle passage MedHallu ships.
"""
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Environment")
    reachable, models = ollama_status()
    if reachable:
        st.success(f"Ollama reachable -- {len(models)} model(s) pulled")
        if models:
            st.caption(", ".join(models))
    else:
        st.warning("Ollama not reachable at localhost:11434 -- the `constant` and `lexical` "
                   "backends still work everywhere.")

with col2:
    st.subheader("Cached data")
    parquet = list((ROOT / "data").glob("*.parquet"))
    if parquet:
        st.success(f"{len(parquet)} dataset/corpus file(s) cached locally")
        for p in parquet:
            st.caption(p.name)
    else:
        st.info("MedHallu not downloaded yet -- the Dataset Explorer, Live Judge and RAG "
                "Inspector pages fetch it on first use (~11 MB).")

with col3:
    st.subheader("Latest results")
    df = load_outputs_csv("results_mitigations.csv")
    if df is not None:
        st.success(f"{df['Model Name'].nunique()} model(s), {len(df)} rows in "
                   "outputs/results_mitigations.csv")
    else:
        st.info("No outputs/results_mitigations.csv yet -- run src/experiment.py "
                "and scripts/build_mitigation_csv.py first.")

st.divider()
st.caption(
    "Baselines and oracle numbers come from the MedHallu paper "
    "([arXiv:2502.14302](https://arxiv.org/abs/2502.14302)); RAG/CoT conditions are this "
    "project's contribution. See the README for the full write-up and known limitations."
)
