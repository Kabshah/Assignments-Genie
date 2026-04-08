FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN pip install --no-cache-dir \
    fastapi==0.135.3 \
    uvicorn==0.42.0 \
    langgraph==1.1.4 \
    langchain-core==1.2.24 \
    langchain-community==0.4.1 \
    langchain-huggingface==1.2.1 \
    langchain-groq==1.1.2 \
    langchain-tavily==0.2.17 \
    langchain-pinecone==0.2.13 \
    sentence-transformers==3.0.1 \
    python-dotenv==1.2.2 \
    requests==2.33.1 \
    pinecone==7.0.1 \
    streamlit==1.56.0 \
    pypdf==4.0.1

COPY backend/ ./backend/
COPY frontend/ ./frontend/

RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/bin/bash", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port=8501 --server.headless=true --server.enableCORS=false && wait"]
