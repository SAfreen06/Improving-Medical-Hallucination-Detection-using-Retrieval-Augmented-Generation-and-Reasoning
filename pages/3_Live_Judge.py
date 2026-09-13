import streamlit as st

from app_lib.common import (
    CONFIGS, anthropic_available, backends, condition_for, data, expected_f1,
    get_corpus_and_retriever, judge_verbose, load_dataset, ollama_status,
    openai_available, verdict_badge,
)
from config import COL_CATEGORY, COL_DIFFICULTY, COL_HALLU, COL_KNOWLEDGE, COL_QUESTION, COL_TRUTH

st.set_page_config(page_title="Live Judge", layout="wide")
st.title("Live Judge")
st.caption("Run a real detection backend on a question/answer pair and see its verdict.")

# ---------------------------------------------------------------------------
# 1. Input: from the dataset, or typed by hand
# ---------------------------------------------------------------------------
st.header("1. Pick a question and answer")
mode = st.radio("Input", ["From dataset", "Custom"], horizontal=True,
                help="'From dataset' lets you judge a real MedHallu row and see whether the "
                     "verdict matches the true label. 'Custom' lets you type your own.")

question = answer = ""
true_label = None
oracle_knowledge = ""

if mode == "From dataset":
    config = st.selectbox("Config", CONFIGS)
    df = load_dataset(config)
    row_idx = st.selectbox("Row #", list(df.index),
                           format_func=lambda i: f"{i}: {str(df.loc[i, COL_QUESTION])[:80]}")
    row = df.loc[row_idx]
    which = st.radio("Answer to judge", ["Ground truth (should be factual)",
                                         "Hallucinated (should be flagged)"], horizontal=True)
    question = str(row[COL_QUESTION])
    answer = str(row[COL_TRUTH]) if which.startswith("Ground") else str(row[COL_HALLU])
    true_label = 0 if which.startswith("Ground") else 1
    st.info(f"**Question:** {question}")
    st.write(f"**Answer being judged** ({'ground truth' if true_label == 0 else 'hallucinated'}):")
    st.code(answer, language=None)
    st.caption(f"Difficulty: {row[COL_DIFFICULTY]}  |  Category: {row[COL_CATEGORY]}  |  "
              f"True label: {true_label}")
    oracle_knowledge = data._flatten_knowledge(row[COL_KNOWLEDGE])
else:
    df = None
    question = st.text_area("Question", height=80)
    answer = st.text_area("Answer to judge", height=100)

st.divider()

# ---------------------------------------------------------------------------
# 2. Knowledge to hand the judge
# ---------------------------------------------------------------------------
st.header("2. Give the judge knowledge (optional)")
st.caption(
    "This is the difference between the `baseline`, `oracle` and `rag` conditions in the "
    "Results Dashboard -- none vs. the exact source passage vs. what a real retriever finds."
)

know_options = ["None (baseline)"]
if mode == "From dataset":
    know_options.append("Oracle (dataset's own source passage)")
know_options += ["RAG (auto-retrieve from a corpus)", "Custom text"]

know_choice = st.radio("Mode", know_options, horizontal=True)

knowledge = None

if know_choice.startswith("Oracle"):
    knowledge = oracle_knowledge
    st.text_area("oracle knowledge", knowledge, height=120, disabled=True, label_visibility="collapsed")

elif know_choice.startswith("RAG"):
    st.caption(
        "Retrieves the top-k passages from an external medical corpus and hands them to the "
        "judge as its knowledge -- this is exactly the `rag` condition. **Expect it to sometimes "
        "fall short of oracle knowledge**: the corpus may simply not contain the specific finding "
        "the question is about (see README). That is the realistic case, not a malfunction."
    )
    corpus_options = {"textbooks (real RAG corpus, ~126k docs, ~101 MB first download)": "textbooks"}
    if mode == "From dataset":
        corpus_options = {"self (ablation -- searches the answer key, fast, no download)": "self",
                          **corpus_options}

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        corpus_label = st.selectbox("Corpus", list(corpus_options))
        corpus_name = corpus_options[corpus_label]
    with rc2:
        k = st.slider("Passages (k)", 1, 5, 3)
    with rc3:
        max_docs = st.number_input(
            "Cap corpus size (0 = full)", min_value=0, value=0 if corpus_name == "self" else 20000,
            step=5000, disabled=(corpus_name == "self"),
            help="Lower = faster first download/index. 0 uses the full corpus (slower, more thorough).")

    if not question.strip():
        st.info("Type or pick a question above first.")
    else:
        try:
            with st.spinner(f"Building/loading the {corpus_name} corpus "
                            f"(cached after the first time)..."):
                corpus, retriever = get_corpus_and_retriever(
                    corpus_name, config if mode == "From dataset" else "pqa_labeled",
                    0 if corpus_name == "self" else max_docs)
            ranked = retriever.search([question], k)[0]
            passages = [corpus[int(j)] for j in ranked]
            knowledge = "\n\n".join(passages)
            with st.expander(f"Retrieved {len(passages)} passage(s) from {corpus_name}", expanded=True):
                for i, p in enumerate(passages, 1):
                    st.markdown(f"**[{i}]** {p[:500]}{'...' if len(p) > 500 else ''}")
        except Exception as e:
            st.error(
                f"Retrieval failed -- likely a network issue downloading the {corpus_name} "
                f"corpus, or a first-time index build that timed out. Try 'statpearls' (smaller) "
                f"or a lower size cap. Details: {e}"
            )

elif know_choice.startswith("Custom"):
    knowledge = st.text_area("Paste retrieved / custom context", height=120)

st.divider()

# ---------------------------------------------------------------------------
# 3. Backend
# ---------------------------------------------------------------------------
st.header("3. Choose a backend")

reachable, ollama_models = ollama_status()
real_options = []
if reachable:
    real_options.append("ollama")
if anthropic_available():
    real_options.append("anthropic")
if openai_available():
    real_options.append("openai")

backend_kind = None
model_name = None

if real_options:
    backend_kind = st.selectbox(
        "Backend", real_options,
        help="A real model judging the answer. Ollama runs free and local; Anthropic/OpenAI "
             "need an API key set in the environment.")
    if backend_kind == "ollama":
        model_name = (st.selectbox("Ollama model", ollama_models) if ollama_models else
                      st.text_input("Ollama model tag (none pulled locally)",
                                    "qwen2.5:1.5b-instruct"))
    elif backend_kind == "anthropic":
        model_name = st.text_input("Model", "claude-sonnet-5")
    elif backend_kind == "openai":
        model_name = st.text_input("Model", "gpt-4o-mini")

    with st.expander("Debug baselines (constant / lexical) -- not real judges"):
        st.caption(
            "`constant` always answers 'hallucinated' -- it will look wrong on every ground-truth "
            "answer, on purpose. `lexical` is a crude word-overlap heuristic with no real "
            "understanding -- its verdicts will look inconsistent, on purpose. Both exist to "
            "smoke-test the plumbing without a model; ignore their verdicts as judgements. "
            "See the README for why these are in the codebase at all."
        )
        if st.checkbox("Use a debug baseline instead of the real backend above"):
            backend_kind = st.selectbox("Debug backend", ["constant", "lexical"])
            model_name = None
else:
    st.warning(
        "No real judge is available: Ollama isn't reachable at localhost:11434, and neither "
        "ANTHROPIC_API_KEY nor OPENAI_API_KEY is set. Falling back to a debug baseline below -- "
        "**its verdicts are not meaningful**. Start Ollama and `ollama pull qwen2.5:7b-instruct` "
        "for a real demo."
    )
    backend_kind = st.selectbox("Debug backend", ["constant", "lexical"])

st.divider()

# ---------------------------------------------------------------------------
# 4. Options + run
# ---------------------------------------------------------------------------
st.header("4. Options")
c1, c2 = st.columns(2)
with c1:
    cot = st.checkbox("Chain-of-thought (reason step by step)")
with c2:
    allow_not_sure = st.checkbox("Allow 'not sure'")

with st.expander("Prompt sent to the model"):
    st.code(backends.build_system_prompt(allow_not_sure, cot), language=None)
    st.code(backends.build_user_prompt(question or "<question>", answer or "<answer>",
                                       knowledge, allow_not_sure, cot), language=None)

condition = condition_for(know_choice, cot)
f1 = expected_f1(model_name, condition) if model_name and condition else None
if f1 is not None:
    st.caption(
        f"Measured accuracy for **{model_name}** in the **{condition}** condition: "
        f"**F1 = {f1}** (see Results Dashboard). That means roughly "
        f"**{round((1 - f1) * 100)}%** of verdicts like this one are expected to be wrong -- "
        f"a wrong answer below isn't necessarily a bug."
    )
elif condition == "baseline":
    st.caption(
        "No knowledge given: this is the hardest condition for any model. The MedHallu paper's "
        "own finding is that models are genuinely bad here (~0.5-0.6 F1) -- try Oracle or RAG "
        "knowledge below to see accuracy improve."
    )

run = st.button("Judge", type="primary", disabled=not (question and answer))

if run:
    spec = backend_kind if not model_name else f"{backend_kind}:{model_name}"
    try:
        backend = backends.get_backend(spec)
    except SystemExit as e:
        st.error(str(e))
        st.stop()

    with st.spinner(f"Asking {backend.name}..."):
        try:
            verdict, raw_text = judge_verbose(backend, question, answer, knowledge, allow_not_sure, cot)
        except Exception as e:
            st.error(f"Backend call failed: {e}")
            st.stop()

    st.subheader("Verdict")
    st.markdown(verdict_badge(verdict))
    if mode == "From dataset" and true_label is not None:
        correct = verdict == true_label
        st.caption(f"True label was {true_label} ({'hallucinated' if true_label else 'factual'}) -- "
                  f"{'correct' if correct else 'incorrect'} (verdict 2 = abstained).")
        if not correct and backend_kind not in ("constant", "lexical"):
            st.caption(
                "A wrong verdict from a real backend is expected sometimes -- this project's own "
                "results put F1 around 0.5-0.85 depending on condition and model, matching the "
                "paper's finding that LLMs are genuinely bad at this task without help. See the "
                "Results Dashboard page for the measured accuracy."
            )
    if raw_text:
        with st.expander("Raw model output", expanded=cot):
            st.text(raw_text)
