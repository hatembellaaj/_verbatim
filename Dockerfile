FROM python:3.10-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY requirements.txt ./
COPY .env .env
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m textblob.download_corpora


COPY . .
COPY .streamlit/config.toml /root/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "verbatim_analyzer/main.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.headless=true"]
