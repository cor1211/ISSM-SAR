# ISSM-SAR

STAC-first runtime for Sentinel-1 monthly SAR super-resolution.

Current recommended focus:
- `STAC + S3` input
- local output packaging
- optional publish to STAC/S3 target
- Docker runtime for server execution

Legacy GEE helpers still exist in the repo, but they are now treated as
optional and are no longer part of the default local/runtime dependency path.

## 📘 Dependency Files

Use the smallest file that matches your use case:

- `requirements_runtime_stac.txt`
  - canonical STAC runtime base
  - used by the Docker image
- `requirements_runtime_local.txt`
  - recommended local install for STAC runtime outside Docker
  - `requirements_runtime_stac.txt` + `torch`
- `requirements_gee.txt`
  - optional add-on only for legacy GEE workflows/tools
- `requirements.txt`
  - broader local development environment
  - includes local runtime + training/evaluation/notebook extras

## ⚙️ Local Run

### 1. Create Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_runtime_local.txt
```

If you still need legacy GEE tooling:

```bash
pip install -r requirements_gee.txt
```

### 2. Run Pipeline Locally

Example for one DB AOI and one month:

```bash
python sar_pipeline.py \
  --config config/pipeline_config_stac_runtime.yaml \
  --db-aoi-id <AOI_UUID> \
  --target-month 2026-01
```

### 3. Publish Locally

Preflight only:

```bash
python sr_publish.py \
  --item-json /abs/path/to/output/<ITEM_ID>.json
```

Publish thật:

```bash
python sr_publish.py \
  --item-json /abs/path/to/output/<ITEM_ID>.json \
  --execute
```

Notes:
- Local nên ưu tiên:
  - `sar_pipeline.py` để chạy pipeline
  - `sr_publish.py` để preflight hoặc publish
- `sr_workflow.py` phù hợp hơn với Docker/container mode.

## 🚀 Docker Run

### 1. Build Image

```bash
docker build -t issm-sar-stac-runtime:local .
```

### 2. Run Pipeline In Container

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  --env-file /abs/path/to/.env \
  -v /abs/path/to/weights:/app/weights:ro \
  -v /abs/path/to/runs:/app/runs \
  issm-sar-stac-runtime:local \
  pipeline \
  --db-aoi-id <AOI_UUID> \
  --target-month 2026-01
```

### 3. Run One-Shot Workflow In Container

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  --env-file /abs/path/to/.env \
  -v /abs/path/to/weights:/app/weights:ro \
  -v /abs/path/to/runs:/app/runs \
  issm-sar-stac-runtime:local \
  workflow \
  --db-aoi-id <AOI_UUID> \
  --target-month 2026-01
```

Useful publish toggles:

```env
WORKFLOW_PUBLISH_ENABLED=false
WORKFLOW_PUBLISH_EXECUTE=false
WORKFLOW_PUBLISH_OVERWRITE=false
```

Recommended safe progression:
1. `WORKFLOW_PUBLISH_ENABLED=false`
2. `WORKFLOW_PUBLISH_ENABLED=true` and `WORKFLOW_PUBLISH_EXECUTE=false`
3. `WORKFLOW_PUBLISH_ENABLED=true` and `WORKFLOW_PUBLISH_EXECUTE=true`

## 📚 Main Files

- `sar_pipeline.py`
  - main runtime pipeline
- `sr_workflow.py`
  - one-shot pipeline then publish wrapper
- `sr_publish.py`
  - publish contract and preflight checks
- `config/pipeline_config_stac_runtime.yaml`
  - runtime STAC pipeline config
- `.env`
  - runtime connection + tuning overrides

## 📝 Notes

- Docker runtime currently installs only `requirements_runtime_stac.txt`.
- Local STAC runtime should use `requirements_runtime_local.txt`.
- Legacy GEE code is still present in the repo, but no longer belongs to the
  default dependency path.
