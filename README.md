# ISSM-SAR

Runtime STAC-first để tạo sản phẩm SAR Super-Resolution theo tháng từ dữ liệu Sentinel-1.

Phạm vi khuyến nghị hiện tại:
- đầu vào `STAC + S3`
- chạy pipeline cục bộ hoặc bằng Docker
- đóng gói output local
- tùy chọn publish lên STAC/S3 đích

Các helper GEE cũ vẫn còn trong repo, nhưng không còn nằm trong đường cài đặt mặc định nữa.

## 📘 Các File Requirements

Chọn file nhỏ nhất đúng với nhu cầu của bạn:

- `requirements_runtime_stac.txt`
  - bộ dependency runtime STAC chuẩn
  - Docker đang dùng file này
- `requirements_runtime_local.txt`
  - bộ dependency khuyến nghị khi chạy local ngoài Docker
  - gồm `requirements_runtime_stac.txt` + `torch`
- `requirements_gee.txt`
  - dependency tùy chọn cho các tool/flow GEE cũ
- `requirements.txt`
  - bộ local rộng hơn cho train / eval / dev

## ⚙️ Chạy Local

### 1. Tạo môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_runtime_local.txt
```

Nếu vẫn cần các tool GEE cũ:

```bash
pip install -r requirements_gee.txt
```

### 2. Chuẩn bị `.env`

```bash
cp .env.example .env
```

Sau đó điền các giá trị cần thiết trong `.env`:
- STAC source
- S3 source
- DB nếu chạy bằng `--db-aoi-id`
- target publish nếu cần publish

### 3. Chạy pipeline local

Ví dụ chạy một AOI từ database:

```bash
python sar_pipeline.py \
  --config config/pipeline_config_stac_runtime.yaml \
  --db-aoi-id <AOI_UUID> \
  --target-month 2026-01
```

### 4. Publish local

Preflight:

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

Gợi ý dùng local:
- dùng `sar_pipeline.py` để chạy pipeline
- dùng `sr_publish.py` để preflight hoặc publish
- `sr_workflow.py` phù hợp hơn với container mode

## 🚀 Chạy Docker

### 1. Build image

```bash
docker build -t issm-sar-stac-runtime:local .
```

### 2. Chạy pipeline trong container

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

### 3. Chạy one-shot workflow trong container

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

Các cờ publish hay dùng trong `.env`:

```env
WORKFLOW_PUBLISH_ENABLED=false
WORKFLOW_PUBLISH_EXECUTE=false
WORKFLOW_PUBLISH_OVERWRITE=false
```

Thứ tự an toàn nên đi:
1. `WORKFLOW_PUBLISH_ENABLED=false`
2. `WORKFLOW_PUBLISH_ENABLED=true` và `WORKFLOW_PUBLISH_EXECUTE=false`
3. `WORKFLOW_PUBLISH_ENABLED=true` và `WORKFLOW_PUBLISH_EXECUTE=true`

## 📚 File Chính

- `sar_pipeline.py`
  - pipeline runtime chính
- `sr_workflow.py`
  - wrapper one-shot `pipeline -> publish`
- `sr_publish.py`
  - preflight và publish contract
- `config/pipeline_config_stac_runtime.yaml`
  - config runtime STAC mặc định
- `.env`
  - cấu hình kết nối và tuning runtime

## 📝 Ghi Chú

- Docker hiện chỉ cài `requirements_runtime_stac.txt`
- chạy local theo hướng STAC nên dùng `requirements_runtime_local.txt`
- code GEE cũ vẫn còn trong repo, nhưng không còn là đường mặc định
