# scikit-learn ≥1.8 needs Python ≥3.11
FROM python:3.12-slim

# Must match the scikit-learn version used to train the model (sklearn.__version__ in TP_01)
ARG SKLEARN_VERSION=1.9.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt "scikit-learn==${SKLEARN_VERSION}"

# Code and model stay owned by root: the API process can read them but not modify them
COPY ./app /app
RUN useradd --system --uid 10001 --no-create-home api
USER api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
