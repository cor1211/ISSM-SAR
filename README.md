# ISSM-SAR

Runtime STAC-first để tạo sản phẩm SAR Super-Resolution theo tháng từ dữ liệu Sentinel-1.

Phạm vi khuyến nghị hiện tại:
- đầu vào `STAC + S3`
- chạy pipeline cục bộ hoặc bằng Docker
- đóng gói output local
- tùy chọn publish lên STAC/S3 đích

Các helper GEE cũ vẫn còn trong repo, nhưng không còn nằm trong đường cài đặt mặc định nữa.

Ma trận runtime chuẩn hiện tại:
- `stac + whole_aoi`
- `stac + componentized_parent_mosaic`
- `gee + whole_aoi`
- `gee + componentized_parent_mosaic`

Core runtime hiện được tổ chức theo đúng 4 luồng chuẩn ở trên; các mode cũ không còn nằm trên đường chạy chính nữa.

## 🗂️ Cấu Trúc Thư Mục

Các thư mục và file đáng quan tâm nhất khi vận hành repo này:

```
ISSM-SAR/
├── config/
│   ├── pipeline_config_stac_runtime.yaml
│   │   └─ config chuẩn cho backend STAC
│   ├── pipeline_config_gee_runtime.yaml
│   │   └─ config chuẩn cho backend GEE
│   └─ chứa các file cấu hình runtime, infer và các recipe YAML
├── docker/
│   └─ entrypoint và các thành phần hỗ trợ container runtime
├── docs/
│   └─ tài liệu vận hành, reference và các ghi chú phân tích
├── src/
│   └─ code training/architecture/data của model, không phải entrypoint runtime chính
├── sar_pipeline.py
│   └─ entrypoint pipeline runtime chính
├── sr_workflow.py
│   └─ wrapper one-shot `pipeline -> publish`
├── sr_publish.py
│   └─ preflight/publish SR item lên STAC/S3 đích
├── query_stac_download.py
│   └─ query STAC, lọc inventory và download AOI subset
├── db_aoi_source.py
│   └─ đọc AOI từ database và materialize thành GeoJSON tạm cho pipeline
├── runtime_env_overrides.py
│   └─ map env var vào pipeline/infer config runtime
├── runtime_logging.py
│   └─ logging helpers dùng chung cho runtime
├── Dockerfile
│   └─ image runtime để chạy bằng Docker
└── .env.example
    └─ template env để khởi tạo `.env` local
```

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

Khuyến nghị dùng `conda` để tạo môi trường, sau đó cài dependency bằng `pip`
đúng theo file requirements của repo:

```bash
conda create -n issm-sar python=3.11 pip -y
conda activate issm-sar
pip install --upgrade pip
pip install -r requirements_runtime_local.txt
```

Nếu bạn thích `venv`, vẫn có thể dùng:

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

Nếu bạn cần cả training / eval / notebook:

```bash
pip install -r requirements.txt
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

Ví dụ chạy cùng recipe chuẩn nhưng với backend GEE:

```bash
python sar_pipeline.py \
  --config config/pipeline_config_gee_runtime.yaml \
  --db-aoi-id <AOI_UUID> \
  --target-month 2026-01
```

Gợi ý:
- `config/pipeline_config_stac_runtime.yaml` là config chuẩn cho backend STAC
- `config/pipeline_config_gee_runtime.yaml` là config chuẩn cho backend GEE
- `trainlike.componentize_seed_intersections=false` sẽ chuyển về `whole_aoi`
- `trainlike.componentize_seed_intersections=true` giữ `componentized_parent_mosaic`
- nên truyền `--config` tường minh thay vì dựa vào config mặc định cũ

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

Nếu bạn không muốn publish khi chạy `workflow`, hãy set tường minh trong `.env`:

```env
WORKFLOW_PUBLISH_ENABLED=false
WORKFLOW_PUBLISH_EXECUTE=false
```

Lưu ý:
- workflow defaults hiện không còn thiên về safe no-publish nữa
- vì vậy nên set rõ các cờ publish trong `.env` thay vì dựa vào mặc định

Thứ tự vận hành an toàn nên đi:
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
  - config runtime chuẩn cho `stac + componentized_parent_mosaic`
- `config/pipeline_config_gee_runtime.yaml`
  - config runtime chuẩn cho `gee + componentized_parent_mosaic`
- `.env`
  - cấu hình kết nối và tuning runtime

## 📝 Ghi Chú

- Docker hiện chỉ cài `requirements_runtime_stac.txt`
- chạy local theo hướng STAC nên dùng `requirements_runtime_local.txt`
- code GEE vẫn được giữ như backend chuẩn song song
- core runtime đã được thu gọn về ma trận canonical `backend + representative_calendar_period + spatial_strategy`
