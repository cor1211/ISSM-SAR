# `sar_pipeline.py` Reference

Tai lieu nay mo ta core runtime hien tai theo kien truc da duoc chot:

- `backend`: `stac` | `gee`
- `selection_strategy`: `representative_calendar_period`
- `spatial_strategy`: `componentized_parent_mosaic`

2 to hop canonical:

- `stac + componentized_parent_mosaic`
- `gee + componentized_parent_mosaic`

File nay chi mo ta duong chay canonical. Cac benchmark hoac tai lieu phan tich lich su co the van ton tai o nhung file docs khac, nhung khong con duoc xem la runtime contract chinh.

## 1. Vai tro cua `sar_pipeline.py`

`sar_pipeline.py` phu trach:

1. nap config va env overrides
2. chuan hoa AOI geometry
3. chon inventory dau vao theo backend
4. tach period theo calendar month
5. chon pools `pre/post`
6. xu ly theo spatial strategy
7. infer SR
8. ghi summary, manifest va SR item local

No khong bao gom publish that len he thong. Publish duoc xu ly boi:

- `sr_publish.py`
- `sr_workflow.py`

## 2. Selection Strategy Canonical

Selection strategy duy nhat cua core runtime hien tai la:

- `representative_calendar_period`

Semantics:

- period duoc chia theo thang lich
- anchor cua period la midpoint cua thang
- `pre/S1T2 = [period_start, anchor)`
- `post/S1T1 = [anchor, period_end)`

STAC va GEE deu dung chung logic period va selection semantics nay.

## 3. Spatial Strategy Canonical

### 3.1 `componentized_parent_mosaic`

Parent AOI duoc tach thanh cac child intersection regions:

- tao child candidates tu `AOI ∩ footprint(item)`
- chon `pre/post` pools cho tung child
- infer tren tung child
- mosaic lai theo parent geometry

Duong nay giu nguyen logic:

- child suppression
- largest-first parent mosaic
- output contract `VV/VH/item_json`

## 4. Backend Matrix

### 4.1 STAC

Workflow mode:

- `workflow.mode = stac_trainlike_composite`

Canonical STAC wrappers:

- `run_stac_representative_componentized_pipeline(...)`

Dispatcher STAC:

- `run_stac_trainlike_pipeline(...)`

### 4.2 GEE

Workflow mode:

- `workflow.mode = gee_trainlike_composite`

Canonical GEE wrappers:

- `run_gee_representative_componentized_pipeline(...)`

Dispatcher GEE:

- `run_gee_trainlike_pipeline(...)`

## 5. Dispatcher Top-Level

Core dispatcher hien tai chi cho phep 2 workflow modes:

- `stac_trainlike_composite`
- `gee_trainlike_composite`

Top-level orchestration:

- `run_pipeline(...)`
- `run_representative_composite_pipeline(...)`

Muc tieu cua lop orchestration la:

- tach ro `backend`
- tach ro `spatial_strategy`
- khong rewrite thuat toan representative pipeline goc

## 6. Dau ra va contract

Nhung phan khong thay doi trong refactor nay:

- logic chon pools `pre/post`
- child selection/suppression
- parent mosaic
- `VV/VH/item_json`
- publish plan / fallback source-target
- cleanup sau publish

Representative run summaries van ghi:

- `workflow_mode`
- `selection_strategy`
- `period_results`
- runtime settings
- output paths

Job-level runtime summary gio co them:

- `pipeline_profile`

de phan biet ro:

- backend
- runtime family
- spatial strategy

## 7. Config khuyen nghi

Config runtime chuan:

- `config/pipeline_config_stac_runtime.yaml`
- `config/pipeline_config_gee_runtime.yaml`

CLI mac dinh nen di theo STAC runtime config.

Neu muon chay GEE:

```bash
python sar_pipeline.py \
  --config config/pipeline_config_gee_runtime.yaml \
  --db-aoi-id <AOI_UUID> \
  --target-month 2026-01
```

## 8. Safety Notes

Refactor hien tai co chu dich:

- giu nguyen 4 luong canonical
- loai bo cac duong chay legacy khoi core runtime
- khong doi output/public contract
- khong dong vao publish/system behavior

Neu can benchmark lich su hoac doi chieu analysis cu, nen tach no ra khoi runtime guide va tooling production-facing.
