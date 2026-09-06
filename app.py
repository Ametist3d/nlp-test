from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from dataframe_bot import DataFrameBot


SAMPLE_CSV = Path(__file__).parent / "sample_data" / "physical_exam_study.csv"


st.set_page_config(
    page_title="DataFrame Analysis Bot",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def get_bot() -> DataFrameBot:
    return DataFrameBot.from_env()


def load_dataframe() -> tuple[pd.DataFrame, str, str]:
    source = st.sidebar.radio(
        "Data source",
        ["Sample dataset", "Upload CSV"],
    )

    if source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is None:
            st.info("Upload a CSV file to start.")
            st.stop()
        dataset_id = f"upload:{uploaded_file.name}:{uploaded_file.size}"
        return pd.read_csv(uploaded_file), uploaded_file.name, dataset_id

    return pd.read_csv(SAMPLE_CSV), SAMPLE_CSV.name, "sample"


st.title("DataFrame Analysis Bot")
st.caption("Ask natural-language questions about a CSV. Calculations are executed with Pandas.")

try:
    df, dataset_name, dataset_id = load_dataframe()
except Exception as exc:
    st.error(f"Could not load CSV: {exc}")
    st.stop()

try:
    bot = get_bot()
except Exception as exc:
    st.error(f"LLM configuration error: {exc}")
    st.stop()

st.sidebar.divider()
st.sidebar.write(f"**Dataset:** {dataset_name}")
st.sidebar.write(f"**Rows:** {len(df):,}")
st.sidebar.write(f"**Columns:** {len(df.columns)}")

with st.expander("Preview dataset", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.write("**Column types**")
    st.code(df.dtypes.to_string())

st.markdown(
    "**Example questions:**  "
    "Mean and standard deviation of intensity for men · "
    "Min/max pain score for women · "
    "Pearson correlation between intensity and pain score"
)

if st.session_state.get("dataset_id") != dataset_id:
    st.session_state.dataset_id = dataset_id
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            st.write(message["summary"])
            if message["result"] is not None:
                st.json(message["result"])
            st.caption(f"Completed in {message['elapsed']:.2f}s")

question = st.chat_input("Ask a question about the current dataset")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing dataframe..."):
                start = time.perf_counter()
                answer = bot.ask(question, df)
                elapsed = time.perf_counter() - start

            st.write(answer.summary)
            if answer.result is not None:
                st.json(answer.result)
            st.caption(f"Completed in {elapsed:.2f}s")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "summary": answer.summary,
                    "result": answer.result,
                    "elapsed": elapsed,
                }
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
