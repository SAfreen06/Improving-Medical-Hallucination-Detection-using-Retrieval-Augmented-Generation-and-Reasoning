import altair as alt
import streamlit as st

from app_lib.common import CONFIGS, data, load_dataset
from config import COL_CATEGORY, COL_DIFFICULTY, COL_HALLU, COL_KNOWLEDGE, COL_QUESTION, COL_TRUTH

st.set_page_config(page_title="Dataset Explorer", layout="wide")
st.title("Dataset Explorer")
st.caption("Browse MedHallu directly: questions, ground truth, hallucinated answers, difficulty and category.")

config = st.sidebar.selectbox("Config", CONFIGS, index=0,
                              help="pqa_labeled: human-labeled, smaller. pqa_artificial: larger, "
                                   "needed for any per-category claim (see README).")
df = load_dataset(config)

difficulties = sorted(df[COL_DIFFICULTY].dropna().unique())
categories = sorted(df[COL_CATEGORY].dropna().unique())

pick_diff = st.sidebar.multiselect("Difficulty", difficulties, default=difficulties)
pick_cat = st.sidebar.multiselect("Category", categories, default=categories)
search = st.sidebar.text_input("Search question text")

filtered = df[df[COL_DIFFICULTY].isin(pick_diff) & df[COL_CATEGORY].isin(pick_cat)]
if search:
    filtered = filtered[filtered[COL_QUESTION].astype(str).str.contains(search, case=False, na=False)]

st.subheader(f"{len(filtered):,} of {len(df):,} rows")

col1, col2 = st.columns(2)
with col1:
    chart = (
        alt.Chart(filtered)
        .mark_bar()
        .encode(x=alt.X(f"{COL_DIFFICULTY}:N", sort=difficulties, title="Difficulty"),
                y=alt.Y("count():Q", title="rows"),
                tooltip=["count()"])
        .properties(height=250, title="Difficulty distribution")
    )
    st.altair_chart(chart, use_container_width=True)
with col2:
    chart = (
        alt.Chart(filtered)
        .mark_bar()
        .encode(x=alt.X("count():Q", title="rows"),
                y=alt.Y(f"{COL_CATEGORY}:N", sort="-x", title=None),
                tooltip=["count()"])
        .properties(height=250, title="Category distribution")
    )
    st.altair_chart(chart, use_container_width=True)

st.divider()

preview = filtered[[COL_QUESTION, COL_DIFFICULTY, COL_CATEGORY]].copy()
preview[COL_QUESTION] = preview[COL_QUESTION].astype(str).str.slice(0, 140)
preview.insert(0, "row #", filtered.index)
st.dataframe(preview, use_container_width=True, hide_index=True, height=320)

if filtered.empty:
    st.stop()

st.divider()
st.subheader("Row detail")
row_idx = st.selectbox("Pick a row #", list(filtered.index),
                       format_func=lambda i: f"{i}: {str(df.loc[i, COL_QUESTION])[:80]}")
row = df.loc[row_idx]

st.markdown(f"**Difficulty:** {row[COL_DIFFICULTY]}  |  **Category:** {row[COL_CATEGORY]}  |  "
           f"**Hallucination type (this project's mapping):** {data.hallucination_type(row[COL_CATEGORY])}")

st.markdown("**Question**")
st.info(str(row[COL_QUESTION]))

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Ground truth answer**")
    st.success(str(row[COL_TRUTH]))
with c2:
    st.markdown("**Hallucinated answer**")
    st.error(str(row[COL_HALLU]))

st.markdown("**Knowledge (oracle context, MedHallu's exact source passage)**")
st.text_area("knowledge", data._flatten_knowledge(row[COL_KNOWLEDGE]), height=180,
            label_visibility="collapsed")
