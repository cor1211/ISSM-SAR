# =============================================================
#  ISSM_SAR Production Inference — Docker Image
# =============================================================
#  Build:
#    docker build -t issm-sar-infer .
#
#  Run (mount data + weights at runtime):
#    docker run --gpus all --rm \
#      -v /path/to/weights:/app/weights:ro \
#      -v /path/to/input/A.tif:/app/data/input/s1t1.tif:ro \
#      -v /path/to/input/A.tif:/app/data/input/s1t1.tif:ro \
#      -v /path/to/output/:/app/data/output \
#      issm-sar-infer \
# =============================================================

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies for rasterio (GDAL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies
COPY requirements_prod.txt .
RUN pip install --no-cache-dir -r requirements_prod.txt \
    && pip install --no-cache-dir rasterio

# Copy source code
COPY src/ src/
COPY config/ config/
COPY infer_production.py .

# Create data directories (will be overridden by volume mounts)
RUN mkdir -p weights data/input data/output

# Default entrypoint
ENTRYPOINT ["python", "infer_production.py"]
CMD ["--config", "config/infer_config.yaml"]
