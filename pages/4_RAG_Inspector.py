import streamlit as st

from app_lib.common import CONFIGS, data, get_corpus_and_retriever, load_dataset
from config import COL_KNOWLEDGE, COL_QUESTION

st.set_page_config(page_title="RAG Inspector", layout="wide")
st.title("RAG Inspector")
st.caption(
    "See what a retriever actually pulls back for a question, next to the oracle passage "
    "MedHallu ships."
)
st.info(
    "**Set expectations first:** RAG here searches an external corpus the benchmark knows "
    "nothing about -- unlike the 'self' ablation, the specific finding a question needs may "
    "simply not be in the corpus. Retrieved passages that are topically related but don't "
    "actually answer the question are the **expected, realistic result** -- that gap between "
    "RAG and oracle is what the project measures, not a bug in this page. See the README."
)

CORPORA = {
    "self (ablation -- fast, no download, always finds the answer)": "self",
    "textbooks (real RAG corpus, ~126k docs, ~101 MB first download)": "textbooks",
}
# 'statpearls' is deliberately not offered here: MedRAG's statpearls repo on
# HuggingFace ships no data files at all (StatPearls' licence forbids
# redistributing the content), so it can never be downloaded this way --
# see src/rag.py's load_external_corpus.

# ---------------------------------------------------------------------------
# Question
# ---------------------------------------------------------------------------
mode = st.radio("Question source", ["From dataset", "Custom"], horizontal=True)

oracle_knowledge = None
config = "pqa_labeled"
if mode == "From dataset":
    config = st.selectbox("Config", CONFIGS)
    df = load_dataset(config)
    row_idx = st.selectbox("Row #", list(df.index),
                           format_func=lambda i: f"{i}: {str(df.loc[i, COL_QUESTION])[:80]}")
    question = str(df.loc[row_idx, COL_QUESTION])
    oracle_knowledge = data._flatten_knowledge(df.loc[row_idx, COL_KNOWLEDGE])
else:
    df = None
    question = st.text_area("Question", height=80)

st.text_area("Query sent to the retriever", question, height=80, disabled=True,
            label_visibility="collapsed")

# ---------------------------------------------------------------------------
# Corpus + retriever
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    options = dict(CORPORA)
    if mode != "From dataset":
        options.pop("self (ablation -- fast, no download, always finds the answer)", None)
    corpus_label = st.selectbox("Corpus", list(options))
    corpus_name = options[corpus_label]
with col2:
    k = st.slider("Passages to retrieve (k)", 1, 10, 3)
with col3:
    max_docs = st.number_input(
        "Cap corpus size (0 = full)", min_value=0,
        value=0 if corpus_name == "self" else 20000, step=5000,
        disabled=(corpus_name == "self"),
        help="Lower = faster first download/index; 0 uses the full corpus.")

if corpus_name == "self" and df is None:
    st.warning("The 'self' ablation needs a dataset to build its corpus from -- pick "
              "'From dataset' above, or choose statpearls/textbooks instead.")
    st.stop()

if corpus_name != "self":
    st.caption("First use downloads and caches the corpus under data/ -- subsequent runs "
              "and app restarts reuse the cached file, so only the first click is slow.")

run = st.button("Retrieve", type="primary", disabled=not question.strip())

if run:
    try:
        with st.spinner(f"Building/loading the {corpus_name} corpus..."):
            corpus, retriever = get_corpus_and_retriever(
                corpus_name, config, 0 if corpus_name == "self" else max_docs)
    except Exception as e:
        st.error(
            f"Could not build the {corpus_name} corpus -- likely a network issue downloading "
            f"MedRAG data, or the build timed out. Try 'statpearls' (much smaller) or a lower "
            f"size cap first. Details: {e}"
        )
        st.stop()

    st.success(f"Corpus: {corpus_name} ({len(corpus):,} passages)  |  retriever: {retriever.name}")

    ranked = retriever.search([question], k)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Retrieved")
        for rank, j in enumerate(ranked, 1):
            with st.container(border=True):
                st.markdown(f"**[{rank}]**")
                st.write(corpus[int(j)][:1200])
    with col2:
        st.subheader("Oracle (MedHallu's own source passage)")
        if oracle_knowledge:
            with st.container(border=True):
                st.write(oracle_knowledge[:1200])
        else:
            st.info("No oracle passage for a custom question -- only dataset rows have one.")
