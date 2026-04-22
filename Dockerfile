# Minimal CPU image for ebrm-system.
# Use this for CI, reproducible benchmarks, and lightweight serving.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build deps for sympy/numpy wheels, then clean up.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip \
 && pip install .

# Non-root user
RUN useradd --create-home --shell /bin/bash ebrm
USER ebrm

ENTRYPOINT ["ebrm-system"]
CMD ["--help"]
