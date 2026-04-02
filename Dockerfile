ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    tini \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_HOME}

COPY requirements_runtime_stac.txt ./
RUN pip install --no-cache-dir -r requirements_runtime_stac.txt

COPY sar_pipeline.py ./
COPY query_stac_download.py ./
COPY infer_production.py ./
COPY db_aoi_source.py ./
COPY runtime_logging.py ./
COPY runtime_env_overrides.py ./
COPY sr_workflow.py ./
COPY sr_publish.py ./
COPY config/ config/
COPY src/model.py src/model.py
COPY tools/publish_sr_outputs.py tools/publish_sr_outputs.py
COPY docker/entrypoint.sh /usr/local/bin/issm-sar-entrypoint

RUN chmod +x /usr/local/bin/issm-sar-entrypoint \
    && mkdir -p /app/weights /app/runs /app/geojson /app/tmp

RUN python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')
import sar_pipeline  # noqa: F401
import query_stac_download  # noqa: F401
import infer_production  # noqa: F401
import db_aoi_source  # noqa: F401
import sr_workflow  # noqa: F401
import sr_publish  # noqa: F401
print('runtime import smoke passed')    
PY

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/issm-sar-entrypoint"]
CMD ["pipeline", "--help"]
