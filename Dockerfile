FROM python:3.12-slim

WORKDIR /app

# libgomp required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Install dependencies (exclude Windows-only polars-runtime-32)
COPY requirements.txt .
RUN grep -v "polars-runtime-32\|colorama" requirements.txt > requirements-linux.txt \
    && pip install --no-cache-dir -r requirements-linux.txt

COPY api/       api/
COPY scheduler/ scheduler/
COPY ui/        ui/
