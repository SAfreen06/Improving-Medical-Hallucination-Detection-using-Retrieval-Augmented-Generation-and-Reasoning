import altair as alt
import pandas as pd
import streamlit as st

from app_lib.common import load_outputs_csv, paper_row

st.set_page_config(page_title="Results Dashboard", layout="wide")
st.title("Results Dashboard")

mitigations = load_outputs_csv("results_mitigations.csv")
baseline = load_outputs_csv("results_baseline.csv")

if mitigations is None and baseline is None:
    st.warning(
        "No results yet. Run `src/experiment.py` (or `src/detect.py`) and "
        "`scripts/build_mitigation_csv.py`, then reload this page."
    )
    st.stop()

CONDITION_ORDER = ["baseline", "oracle", "rag", "cot", "rag+cot", "oracle+rag"]

# ---------------------------------------------------------------------------
# Five-condition comparison
# ---------------------------------------------------------------------------
if mitigations is not None:
    st.header("Five conditions, by model")
    st.caption(
        "baseline = question + answer only  |  oracle = + the exact source abstract  |  "
        "rag = + retrieved textbook passages  |  cot = step-by-step reasoning, no extra text  |  "
        "rag+cot = both"
    )

    models = sorted(mitigations["Model Name"].unique())
    picked = st.multiselect("Model(s)", models, default=models)
    view = mitigations[mitigations["Model Name"].isin(picked)].copy()
    view["Condition"] = pd.Categorical(view["Condition"], categories=CONDITION_ORDER, ordered=True)
    view = view.sort_values(["Model Name", "Condition"])

    metric = st.radio("Metric", ["f1", "precision", "recall"], horizontal=True)

    chart = (
        alt.Chart(view)
        .mark_bar()
        .encode(
            x=alt.X("Condition:N", sort=CONDITION_ORDER, title=None),
            y=alt.Y(f"{metric}:Q", title=metric.upper()),
            color=alt.Color("Model Name:N"),
            xOffset="Model Name:N",
            tooltip=["Model Name", "Condition", "f1", "precision", "recall", "n_rows", "corpus"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.divider()

    # -----------------------------------------------------------------------
    # Difficulty breakdown
    # -----------------------------------------------------------------------
    st.header("By difficulty")
    diff_cols = ["easy_f1", "medium_f1", "hard_f1"]
    if set(diff_cols) <= set(view.columns):
        long = view.melt(
            id_vars=["Model Name", "Condition"], value_vars=diff_cols,
            var_name="difficulty", value_name="score",
        )
        long["difficulty"] = long["difficulty"].str.replace("_f1", "", regex=False)
        chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X("Condition:N", sort=CONDITION_ORDER, title=None),
                y=alt.Y("score:Q", title="F1"),
                color=alt.Color("difficulty:N", sort=["easy", "medium", "hard"]),
                column=alt.Column("Model Name:N"),
                xOffset="difficulty:N",
                tooltip=["Model Name", "Condition", "difficulty", "score"],
            )
            .properties(height=300, width=220)
        )
        st.altair_chart(chart, use_container_width=False)

    st.divider()

    # -----------------------------------------------------------------------
    # Knowledge vs logic -- the Li et al. taxonomy check
    # -----------------------------------------------------------------------
    st.header("Knowledge vs logic slice -- does the taxonomy hold?")
    st.markdown(
        """
Li et al. predict **RAG fixes knowledge-based hallucinations, CoT fixes logic-based ones.**
A clean diagonal below supports that; if RAG lifts both slices equally (or CoT does), the
taxonomy doesn't separate cleanly on medical text.
"""
    )
    know_cols = ["knowledge_f1", "logic_f1"]
    if set(know_cols) <= set(view.columns):
        long = view.melt(
            id_vars=["Model Name", "Condition"], value_vars=know_cols,
            var_name="slice", value_name="score",
        )
        long["slice"] = long["slice"].str.replace("_f1", "", regex=False)
        chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X("Condition:N", sort=CONDITION_ORDER, title=None),
                y=alt.Y("score:Q", title="F1"),
                color=alt.Color("slice:N", scale=alt.Scale(domain=["knowledge", "logic"])),
                column=alt.Column("Model Name:N"),
                xOffset="slice:N",
                tooltip=["Model Name", "Condition", "slice", "score"],
            )
            .properties(height=300, width=220)
        )
        st.altair_chart(chart, use_container_width=False)

        st.markdown("**Lift vs baseline** (positive = mitigation helped that slice)")
        pivot = view.pivot_table(index=["Model Name", "Condition"], values=know_cols).reset_index()
        lift_rows = []
        for model, g in pivot.groupby("Model Name"):
            base = g[g["Condition"] == "baseline"]
            if base.empty:
                continue
            base = base.iloc[0]
            for _, row in g[g["Condition"].isin(["rag", "cot", "rag+cot"])].iterrows():
                lift_rows.append({
                    "Model Name": model,
                    "Condition": row["Condition"],
                    "knowledge lift": round(row["knowledge_f1"] - base["knowledge_f1"], 3),
                    "logic lift": round(row["logic_f1"] - base["logic_f1"], 3),
                })
        if lift_rows:
            st.dataframe(pd.DataFrame(lift_rows), use_container_width=True, hide_index=True)

    st.divider()

    # -----------------------------------------------------------------------
    # vs paper
    # -----------------------------------------------------------------------
    st.header("vs. the MedHallu paper")
    st.caption(
        "Comparing like with like: knowledge-using conditions (oracle/rag/rag+cot) against the "
        "paper's *with-knowledge* row, knowledge-free ones (baseline/cot) against its *without* row."
    )
    rows = []
    for _, r in view.iterrows():
        name, ref = paper_row(r["Model Name"])
        if ref is None:
            continue
        ref = dict(zip(
            ["f1", "p", "easy_f1", "med_f1", "hard_f1",
             "k_f1", "k_p", "k_easy_f1", "k_med_f1", "k_hard_f1", "delta"], ref))
        against = ref["k_f1"] if r["Condition"] in ("oracle", "rag", "rag+cot", "oracle+rag") else ref["f1"]
        rows.append({
            "Model Name": r["Model Name"],
            "paper row": name,
            "Condition": r["Condition"],
            "F1": r["f1"],
            "paper F1 (matched setting)": against,
            "delta": round(r["f1"] - against, 3),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("None of the selected models match a row in the paper's Table 2.")

# ---------------------------------------------------------------------------
# Baseline validation (Step 1) table
# ---------------------------------------------------------------------------
if baseline is not None:
    st.divider()
    st.header("Baseline validation (Step 1)")
    st.caption(
        "Reproducing the paper's own baseline/oracle numbers, as a sanity check that the "
        "setup is sound before adding mitigations."
    )
    st.dataframe(baseline, use_container_width=True, hide_index=True)
