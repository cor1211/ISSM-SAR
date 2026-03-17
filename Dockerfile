FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System packages required by rasterio/GDAL and compiled Python deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements_prod.txt .
RUN pip install --no-cache-dir -r requirements_prod.txt

# Copy only runtime code needed for the STAC->composite->infer pipeline.
COPY src/ src/
COPY config/ config/
COPY infer_production.py .
COPY query_stac_download.py .
COPY sar_pipeline.py .
COPY gee_compare_download.py .
COPY gee_trainlike_download.py .

# Default runtime directories. Real data/weights are expected to be mounted in.
RUN mkdir -p weights data/input data/output runs geojson

# Default entrypoint exposes the full production pipeline.
ENTRYPOINT ["python", "sar_pipeline.py"]
CMD ["--help"]
