FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# ocrmypdf + tesseract add a text layer to scanned notes so they can be indexed;
# ghostscript is ocrmypdf's PDF engine and osd holds the page-orientation model.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ocrmypdf \
        ghostscript \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-osd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first for layer caching. The default 30s is not enough for the
# larger wheels (torch-sized downloads time out mid-extraction on a home line).
ENV UV_HTTP_TIMEOUT=180
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uvicorn", "vtu_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
