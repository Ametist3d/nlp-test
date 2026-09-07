import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from dataframe_bot import DataFrameContext, create_pandas_dataframe_agent

load_dotenv()

SAMPLE_CSV = "./sample_data/physical_exam_study.csv"

llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model=os.environ["GROQ_MODEL"],
    temperature=0,
)

agent = create_pandas_dataframe_agent(llm)

st.title("DataFrame Bot")

uploaded = st.file_uploader("Upload CSV", type="csv")

df = pd.read_csv(uploaded if uploaded else SAMPLE_CSV)

st.caption(f"{len(df):,} rows × {len(df.columns)} columns")
st.dataframe(df.head(), width="stretch")

question = st.chat_input("Ask about the data")

if question:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        context=DataFrameContext(data=df),
    )

    st.write(result["messages"][-1].content)
