FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REPOLENS_CACHE_DIR=/data/cache \
    # Reach an Ollama server running on the Docker host by default.
    REPOLENS_LLM_BASE_URL=http://host.docker.internal:11434/v1 \
    REPOLENS_EMBED_BASE_URL=http://host.docker.internal:11434/v1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY repolens ./repolens
RUN pip install ".[app]"
COPY app.py ./
COPY .streamlit ./.streamlit

RUN useradd --create-home repolens && mkdir -p /data/cache && chown -R repolens /data
USER repolens
VOLUME ["/data"]
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
