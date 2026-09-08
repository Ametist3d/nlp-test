# Artidis DataFrame Analysis Bot

Small Streamlit demo for querying CSV data with an LLM-driven Pandas agent.

## Architecture

```text
Streamlit UI
    ↓
DataFrameBot
    ↓
LangChain agent + Groq
    ↓
execute_dataframe_code
    ↓
Pandas / runtime DataFrame
```

The agent is initialized without a DataFrame. The current DataFrame is injected through LangChain runtime context for each `ask(question, data)` invocation. LLM-generated Pandas code is executed by the tool, and exact computed values are stored in agent state. The LLM is then used only to summarize those computed values.

## Configuration

Copy the environment template:

```bash
cp .env.example .env
```

Set:

```env
GROQ_API_KEY=...
GROQ_MODEL=...
```

The selected Groq model must support tool calling.

## Run locally

Python 3.13 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run with Docker

```bash
docker compose up --build -d
```

Open:

```text
http://<server-ip>:8501
```

The app supports the bundled `physical_exam_study.csv` and arbitrary CSV uploads.

## Trade-offs

- Pandas is appropriate for the assignment-sized datasets and keeps the solution simple.
- The DataFrame is passed per invocation rather than bound to the agent at initialization.
- Numeric results come from executed Pandas code, not from LLM-generated prose.
- `PythonAstREPLTool` executes generated Python code. Docker provides process isolation from the host, but it is not a complete security sandbox. A production deployment should use stronger execution isolation and authentication.
- Remote storage such as Google Drive, databases, n8n, or MCP is intentionally excluded from this demo and can be added later through a data-source abstraction.
