# Server B - Shaman
# Ищет смысл в своих данных, используя ТОЛЬКО априорные знания
# НЕ получает никаких данных от Server A!

FROM python:3.11-slim

LABEL maintainer="shaman-experiment"
LABEL description="Server B - Shaman for distributed experiment"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared knowledge base (meanings.py - the ONLY shared thing!)
COPY shared/ /app/shared/

# Copy NOBS module
COPY nobs/ /app/nobs/

# Copy server code
COPY server.py /app/

# Create directories
RUN mkdir -p /data /output

# Default command
CMD ["python", "server.py", "--meaning", "bullish_trend", "--data", "/data/BTC_USDT_USDT-4h-futures.feather", "--output", "/output"]
