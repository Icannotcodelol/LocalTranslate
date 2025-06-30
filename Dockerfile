FROM python:3.11-slim

# Install system packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirement definitions
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download models into the image so the container can run completely offline
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV TRANSFORMERS_CACHE=/models

RUN python - <<EOF
import os
from huggingface_hub import snapshot_download

token = os.getenv("HF_TOKEN")
if not token:
    raise RuntimeError("HF_TOKEN environment variable must be provided at build time")

# Whisper Medium
snapshot_download(repo_id="openai/whisper-medium", cache_dir="/models", use_auth_token=token)
# M2M100 418M
snapshot_download(repo_id="facebook/m2m100_418M", cache_dir="/models", use_auth_token=token)
EOF

# Copy the application source
COPY . /app

# Expose port and launch the server
EXPOSE 3000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"] 