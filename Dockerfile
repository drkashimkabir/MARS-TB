# CPU image for running the tests and the CLI on synthetic data. NOT BUILT OR TESTED in the environment where
# this repository was prepared (no Docker daemon was used). For GPU work install the CUDA build of PyTorch.
FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -e ".[dev]"
COPY . .
RUN useradd --create-home runner && chown -R runner /app
USER runner
CMD ["python", "-m", "pytest", "-m", "not slow"]
