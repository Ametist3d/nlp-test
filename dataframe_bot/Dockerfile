FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY dataframe_bot/app.py dataframe_bot/dataframe_bot.py ./
COPY sample_data ./sample_data

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8505

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8505/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8505", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
