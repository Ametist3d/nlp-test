from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Any
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

DECIMAL_PLACES = 4
MAX_COLUMNS = 24
MAX_CELL_CHARS = 80
MAX_CONTEXT_CHARS = 7000
MAX_TOOL_CHARS = 3000


class DataAnalysisAnswer(BaseModel):
    summary: str
    result: dict[str, Any] | None = None


@dataclass
class DataFrameContext:
    data: pd.DataFrame


class DataFrameState(AgentState):
    result: NotRequired[dict[str, Any]]


def normalize_value(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): normalize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]
    return value


@tool
def execute_dataframe_code(
    query: str,
    runtime: ToolRuntime[DataFrameContext, DataFrameState],
) -> Command:
    """Execute Python/Pandas code against the current dataframe named df."""

    repl = PythonAstREPLTool(locals={"df": runtime.context.data, "pd": pd})
    output = repl.invoke(query)
    result = repl.locals.get("result")

    if isinstance(result, dict):
        result = normalize_value(result)
    else:
        try:
            value = ast.literal_eval(str(output).strip())
        except (ValueError, SyntaxError):
            value = str(output).strip()
        result = {"value": normalize_value(value)}

    content = str(result)[:MAX_TOOL_CHARS]
    return Command(update={
        "result": result,
        "messages": [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)],
    })

AGENT_PROMPT = f"""
You are a data analysis agent working with a pandas DataFrame named `df`.

Use execute_dataframe_code for every question that requires information
from the dataframe.

Rules:
- Use ONLY columns present in the current dataframe schema.
- Never assume columns from previous datasets or examples.
- Keep generated Python code short and concise.
- Do not import libraries unless absolutely necessary.
- Prefer exactly ONE tool call.
- Calculate all requested values in that call.
- Store results in a dictionary named `result`.
- Use standard Python values with float() or int().
- Do not repeat successful calculations.
- Use recent conversation history to interpret follow-up questions.
- If the user refers to a previous statistic, preserve that context.


For vague requests such as "give me some stats":
- Use column names in the summary to clarify what is being reported.
- Return a small useful summary rather than every possible statistic.
- Include row count and simple statistics for available numeric columns.
- Do not claim the dataset measures something it does not contain.
- For ambiguous wording, prefer the closest valid dataframe analysis
  or clearly state the limitation

Do not unnecessarily round numeric results.
Preserve at least {DECIMAL_PLACES} decimal places for floating-point values.
"""


class DataFrameBot:
    def __init__(self, llm: ChatGroq):
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

        if not api_key or not model:
            raise RuntimeError("GROQ_API_KEY and GROQ_MODEL must be set.")

        return cls(ChatGroq(
            api_key=api_key,
            model=model,
            temperature=0,
            max_tokens=1024,
            reasoning_effort="low",
            reasoning_format="hidden",
            max_retries=2,
        ))

    def dataframe_context(self, data: pd.DataFrame) -> str:
        columns = list(data.columns)
        shown = columns if len(columns) <= MAX_COLUMNS else columns[:12] + columns[-6:]
        schema = ", ".join(f"{c} ({data[c].dtype})" for c in shown)

        if len(columns) > len(shown):
            schema += f", ... +{len(columns) - len(shown)} more"

        sample = ""
        if len(columns) <= MAX_COLUMNS:
            preview = data[shown].head(2).astype(str)
            preview = preview.apply(lambda col: col.str[:MAX_CELL_CHARS])
            sample = f"\nSample:\n{preview.to_string(index=False)}"

        context = f"""Shape: {data.shape}
Columns: {schema}{sample}
The Python tool has access to the full dataframe. Inspect df.columns if needed."""
        return context[:MAX_CONTEXT_CHARS]

    def ask(
        self,
        question: str,
        data: pd.DataFrame,
        history: list[dict] | None = None,
    ) -> DataAnalysisAnswer:
        if data.empty:
            raise ValueError("The dataframe is empty.")
        if not question.strip():
            raise ValueError("Question must not be empty.")

        messages = [
            {"role": m["role"], "content": str(m["content"])[:800]}
            for m in (history or [])[-4:]
        ]
        messages.append({
            "role": "user",
            "content": f"{self.dataframe_context(data)}\n\nQuestion: {question}",
        })

        agent_result = self.agent.invoke(
            {"messages": messages},
            context=DataFrameContext(data=data),
        )

        result = agent_result.get("result")
        final_message = agent_result["messages"][-1]
        summary = final_message.content
        if not isinstance(summary, str):
            summary = str(summary)

        if result is None:
            tool_used = any(isinstance(m, ToolMessage) for m in agent_result["messages"])
            if tool_used:
                raise ValueError("The dataframe tool did not produce a valid result.")

        return DataAnalysisAnswer(summary=summary, result=result)
    