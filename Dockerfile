# The web service of SPEC section 17, built from this repository.
# Hugging Face Spaces runs the container as user 1000 and expects port 7860.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 1000 app
WORKDIR /home/app

COPY --chown=app:app pyproject.toml README.md LICENSE ./
COPY --chown=app:app src ./src
RUN pip install ".[api]"

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/v1/health', timeout=4)"
CMD ["python", "-m", "mlguard.api"]
