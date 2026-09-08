from __future__ import annotations

import time
from pathlib import Path
import hashlib

import pandas as pd
import streamlit as st
import traceback

from dataframe_bot import DataFrameBot


SAMPLE_CSV = Path(__file__).parent / "sample_data" / "physical_exam_study.csv"

UPLOAD_DIR = Path("/tmp/dataframe_bot_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="DataFrame Analysis Bot",
    layout="wide",
)


@st.cache_resource
def get_bot() -> DataFrameBot:
    return DataFrameBot.from_env()


@st.cache_data(show_spinner=False)
def read_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_dataframe() -> tuple[pd.DataFrame, str, str]:
    source = st.sidebar.radio(
        "Data source",
        ["Sample dataset", "Upload CSV"],
    )

    if source == "Sample dataset":
        path = SAMPLE_CSV
        return (
            read_csv_cached(str(path)),
            path.name,
            "sample",
        )

    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="csv_upload",
        max_upload_size=500,
    )

    if uploaded_file is not None:
        dataset_id = hashlib.md5(
            f"{uploaded_file.name}:{uploaded_file.size}".encode()
        ).hexdigest()

        path = UPLOAD_DIR / f"{dataset_id}.csv"

        if not path.exists():
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.session_state.dataset_path = str(path)
        st.session_state.dataset_name = uploaded_file.name
        st.session_state.dataset_id = dataset_id

    dataset_path = st.session_state.get("dataset_path")

    if not dataset_path:
        st.info("Upload a CSV file to start.")
        st.stop()

    path = Path(dataset_path)

    if not path.exists():
        st.session_state.pop("dataset_path", None)
        st.session_state.pop("dataset_name", None)
        st.session_state.pop("dataset_id", None)

        st.warning("Uploaded dataset is no longer available. Please upload it again.")
        st.stop()

    return (
        read_csv_cached(str(path)),
        st.session_state["dataset_name"],
        st.session_state["dataset_id"],
    )


st.title("DataFrame Analysis Bot")
st.caption("Ask natural-language questions about a CSV. Calculations are executed with Pandas.")

df, dataset_name, dataset_id = load_dataframe()
bot = get_bot()

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
    "Analyze this dataset and show statistics · "
    "Which column is most representative? · "
    "What is median value in each column?"
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
                with st.expander("Show raw result", expanded=False):
                    st.json(message["result"])
            st.caption(f"Completed in {message['elapsed']:.2f}s")

question = st.chat_input("Ask a question about the current dataset")

if question:
    history = []
    for message in st.session_state.messages[-4:]:
        content = message.get("content", message.get("summary", ""))
        history.append({"role": message["role"], "content": content})

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        try:
            answer = bot.ask(question, df, history=history)
            elapsed = time.perf_counter() - start

            st.write(answer.summary)
            if answer.result is not None:
                with st.expander("Show raw result", expanded=False):
                    st.json(answer.result)
            st.caption(f"Completed in {elapsed:.2f}s")

            st.session_state.messages.append({
                "role": "assistant",
                "summary": answer.summary,
                "result": answer.result,
                "elapsed": elapsed,
            })
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
