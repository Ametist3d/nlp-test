from dataclasses import dataclass

import pandas as pd
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_experimental.tools import PythonAstREPLTool

DECIMAL_PLACES = 4


@dataclass
class DataFrameContext:
    data: pd.DataFrame


@tool
def execute_dataframe_code(
    code: str,
    runtime: ToolRuntime[DataFrameContext],
) -> str:
    """Execute Python/Pandas code against the current dataframe named `df`."""
    repl = PythonAstREPLTool(locals={"df": runtime.context.data, "pd": pd})
    return str(repl.invoke(code))


def create_pandas_dataframe_agent(llm):
    return create_agent(
        model=llm,
        tools=[execute_dataframe_code],
        context_schema=DataFrameContext,
        system_prompt=f"""
    You are a pandas data analysis agent.
    The current dataframe is available as `df`.

    Use execute_dataframe_code for every dataframe calculation.
    Keep generated Python short and use one tool call when possible.
    Make the last Python expression the value or dictionary to return.
    Use only values returned by the tool in your answer.

    Do not assume categorical values such as gender labels.
    Use the actual values in the dataframe, preferably with groupby when appropriate.

    Do not unnecessarily round numeric results.
    Preserve at least {DECIMAL_PLACES} decimal places for floating-point values.
    """,
    )
