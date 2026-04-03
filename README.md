# ISSM-SAR

STAC-first runtime for Sentinel-1 monthly SAR super-resolution.

Current recommended focus:
- `STAC + S3` input
- local output packaging
- optional publish to STAC/S3 target
- Docker runtime for server execution

Legacy GEE helpers still exist in the repo, but they are now treated as
optional and are no longer part of the default local/runtime dependency path.

## Dependency files

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
- `requirements_prod.txt`
  - compatibility alias to `requirements_runtime_local.txt`

## Local run

### 1. Create environment

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

### 2. Prepare environment variables

The pipeline reads runtime settings from `.env`.

Minimum STAC/S3 source settings:

```env
STAC_API_URL=
S3_ENDPOINT=
S3_ACCESS_KEY=
S3_SECRET_KEY=
```

If you want publish target overrides, set:

```env
SR_S3_BUCKET=
SR_S3_PREFIX_MONTHLY=
SR_COLLECTION_ID_MONTHLY=
SR_S3_ENDPOINT=
SR_S3_ACCESS_KEY=
SR_S3_SECRET_KEY=
SR_STAC_ROOT_URL=
```

If the `SR_* endpoint/credential/root-url` values are blank, target publish can
fall back to the source connection settings. The target identity keys should
still be set explicitly:

- `SR_S3_BUCKET`
- `SR_S3_PREFIX_MONTHLY`
- `SR_COLLECTION_ID_MONTHLY`

### 3. Run pipeline locally

Example for one DB AOI and one month:

```bash
python sar_pipeline.py \
  --config config/pipeline_config_stac_runtime.yaml \
  --db-aoi-id <AOI_UUID> \
  --target-month 2026-01
```

This is the recommended local entry point.

Notes:
- `sar_pipeline.py` runs cleanly on a normal machine with the local STAC requirements.
- `sr_workflow.py` is primarily intended for the container layout (`/app/...`) and is
  best used through Docker unless you deliberately mirror that layout locally.

## Docker run

### 1. Build image

```bash
docker build -t issm-sar-stac-runtime:local .
```

### 2. Run pipeline in container

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

### 3. Run one-shot workflow in container

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

Useful publish toggles in `.env`:

```env
WORKFLOW_PUBLISH_ENABLED=false
WORKFLOW_PUBLISH_EXECUTE=false
WORKFLOW_PUBLISH_OVERWRITE=false
```

Recommended safe progression:
1. `WORKFLOW_PUBLISH_ENABLED=false`
2. `WORKFLOW_PUBLISH_ENABLED=true` and `WORKFLOW_PUBLISH_EXECUTE=false`
3. `WORKFLOW_PUBLISH_ENABLED=true` and `WORKFLOW_PUBLISH_EXECUTE=true`

## Recommended files to keep in mind

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

## Notes

- Docker runtime currently installs only `requirements_runtime_stac.txt`.
- Local STAC runtime should use `requirements_runtime_local.txt`.
- Legacy GEE code is still present in the repo, but no longer belongs to the
  default dependency path.
