# Dockerfile of the Hugging Face Space: installs a released version from PyPI.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 1000 app && pip install "ml-data-guard[api]==0.2.0"

USER app
WORKDIR /home/app
EXPOSE 7860
CMD ["python", "-m", "mlguard.api"]
