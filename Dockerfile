FROM python:3.12-slim

WORKDIR /app

# Install dependencies (exclude Windows-only polars-runtime-32)
COPY requirements.txt .
RUN grep -v "polars-runtime-32\|colorama" requirements.txt > requirements-linux.txt \
    && pip install --no-cache-dir -r requirements-linux.txt

COPY api/       api/
COPY scheduler/ scheduler/
COPY ui/        ui/
