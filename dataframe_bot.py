from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Optional
from typing_extensions import NotRequired

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_experimental.tools import PythonAstREPLTool
from langchain_groq import ChatGroq
from langgraph.types import Command
from pydantic import BaseModel


ScalarValue = int | float | str | bool | None


class DataAnalysisAnswer(BaseModel):
    summary: str
    result: Optional[dict[str, ScalarValue]] = None


@dataclass
class DataFrameContext:
    data: pd.DataFrame


class DataFrameState(AgentState):
    result: NotRequired[dict[str, ScalarValue]]


@tool
def execute_dataframe_code(
    query: str,
    runtime: ToolRuntime[DataFrameContext, DataFrameState],
) -> Command:
    """Execute Python/Pandas code against the current dataframe named ``df``."""

    repl = PythonAstREPLTool(
        locals={
            "df": runtime.context.data,
            "pd": pd,
        }
    )

    output = repl.invoke(query)

    # Preferred path: the model creates a structured result dictionary.
    result = repl.locals.get("result")

    if isinstance(result, dict):
        result = {
            str(key): value.item() if hasattr(value, "item") else value
            for key, value in result.items()
        }
    else:
        # Fallback for valid scalar expressions such as df["x"].mean().
        try:
            value = ast.literal_eval(str(output).strip())
        except (ValueError, SyntaxError):
            value = str(output).strip()

        if hasattr(value, "item"):
            value = value.item()

        result = {"value": value}

    return Command(
        update={
            "result": result,
            "messages": [
                ToolMessage(
                    content=str(result),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


AGENT_PROMPT = """
You are a data analysis agent working with a pandas DataFrame named `df`.

You MUST use the execute_dataframe_code tool for every dataframe calculation.
Never estimate numeric answers yourself.

IMPORTANT:
- Prefer exactly ONE tool call per user question.
- Calculate ALL requested values in that single call.
- Assign all requested values to a dictionary named `result` with clear keys.
- Do not call the tool separately for each statistic.
- Do not repeat a calculation that has already succeeded.
- Use standard Python values with float() or int().

Example:

values = df["temperature"]
result = {
    "minimum_temperature": float(values.min()),
    "maximum_temperature": float(values.max()),
}
"""


class DataFrameBot:
    """LLM data-analysis agent whose DataFrame is supplied per invocation."""

    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.agent = create_agent(
            model=llm,
            tools=[execute_dataframe_code],
            system_prompt=AGENT_PROMPT,
            context_schema=DataFrameContext,
            state_schema=DataFrameState,
        )

    @classmethod
    def from_env(cls) -> "DataFrameBot":
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("GROQ_MODEL")

        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        if not model:
            raise RuntimeError("GROQ_MODEL is not set.")

        llm = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=0,
            max_tokens=512,
            max_retries=2,
        )
        return cls(llm)

    def dataframe_context(self, data: pd.DataFrame) -> str:
        preview = data.head(3).copy()

        for column in preview.columns:
            if pd.api.types.is_string_dtype(preview[column].dtype):
                values = preview[column].astype(str)

                if values.str.len().max() > 200:
                    preview[column] = "<long text omitted>"
                else:
                    preview[column] = values.str[:80]

        return f"""
    DataFrame shape: {data.shape}

    Columns:
    {data.dtypes.to_string()}

    Sample rows:
    {preview.to_string(index=False)}
    """
        
    def ask(self, question: str, data: pd.DataFrame) -> DataAnalysisAnswer:
        if data.empty:
            raise ValueError("The dataframe is empty.")
        if not question.strip():
            raise ValueError("Question must not be empty.")

        context = DataFrameContext(data=data)

        dataframe_info = f"""
        {self.dataframe_context(data)}

        Question:
        {question}
        """

        agent_result = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": dataframe_info,
                    }
                ]
            },
            context=context,
        )

        computed_result = agent_result.get("result")

        if computed_result is None:
            messages = agent_result["messages"]

            tool_was_used = any(
                isinstance(message, ToolMessage)
                for message in messages
            )

            if tool_was_used:
                raise ValueError(
                    "The dataframe tool was called but did not produce a valid result."
                )

            # General / non-dataframe question
            final_message = messages[-1]

            return DataAnalysisAnswer(
                summary=str(final_message.content),
                result=None,
            )

        summary_response = self.llm.invoke(
            f"""
        Question:
        {question}

        Computed result:
        {computed_result}

        Explain the result briefly.
        Use only the supplied values.
        """
        )

        summary = summary_response.content
        if not isinstance(summary, str):
            summary = str(summary)

        return DataAnalysisAnswer(
            summary=summary,
            result=computed_result,
        )
