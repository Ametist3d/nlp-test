# DataFrame Bot

LLM-powered CSV analysis built with LangChain and Pandas.
---
The project contains two implementations:

- `dataframe_bot/` — minimal implementation focused on the assignment requirement: create the agent once and provide the DataFrame at runtime.

*this version can be tested on* [dataframe_bot](http://46.225.185.220:8501)

- `dataframe_bot_advanced/` — extended version for wider/larger datasets with compact DataFrame context, bounded tool output, structured results, and short conversation history.

*this version can be tested on* [dataframe_bot_advanced](http://46.225.185.220:8502)
---

## Project structure

```text
nlp-test/
├── dataframe_bot/
├── dataframe_bot_advanced/
├── sample_data/
├── requirements.txt
└── .env
```

## Setup

Python 3.13+ is recommended.

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

## Run

Minimal version:

```bash
streamlit run dataframe_bot/app.py
```

Advanced version:

```bash
streamlit run dataframe_bot_advanced/app.py
```

The sample CSV is loaded by default. Uploading another CSV replaces it for the current session.

## Docker

Minimal version:

```bash
docker compose -f dataframe_bot/compose.yaml up --build -d
```

Advanced version:

```bash
docker compose -f dataframe_bot_advanced/compose.yaml up --build -d
```

## Core idea

Unlike the standard Pandas agent, the DataFrame is not bound when the agent is created:

```python
agent = create_pandas_dataframe_agent(llm)

result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    context=DataFrameContext(data=df),
)
```

The same agent can therefore be reused with different DataFrames at runtime.

## Example questions

- What is the mean and standard deviation of intensity for men?
- What are the minimum and maximum pain scores for women?
- What is the Pearson correlation between intensity and pain score overall and by gender?

## Note

`PythonAstREPLTool` executes model-generated Python code. The demo should be treated as a trusted/local environment rather than a public code-execution service.
