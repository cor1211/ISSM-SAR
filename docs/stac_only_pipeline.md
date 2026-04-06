# STAC Backend Guide

## 1. STAC backend hien nay dang o vai tro nao

`stac_trainlike_composite` van duoc giu va da duoc canh chinh theo cung representative-month semantics voi GEE. Tuy nhien inventory STAC test hien tai van ngheo item, nen backend nay hien phu hop hon cho:

- parity testing
- spatial logic validation
- benchmark/QA
- chuyen lai production sau nay khi inventory du day

## 2. Dieu bat buoc giu giong GEE

Khi STAC du item, ket qua phai giong GEE o cac diem sau:

- cung period generation theo `calendar month`
- cung `period_anchor_datetime = midpoint that cua period`
- cung `pre/S1T2`, `post/S1T1`
- cung geometry-based AOI coverage
- cung representative relaxation ladder
- cung witness support pair semantics
- cung `EPSG:3857`, `10m`
- cung focal config
- cung `model(S1T1=later, S1T2=earlier)`

Khac biet backend:

- STAC: download subsets + local composite
- GEE: query/export composite trong Earth Engine

## 3. STAC co nhung che do nao

### 3.1 Representative-month mode

Bat khi:

- `trainlike.selection_strategy = representative_calendar_period`

No la che do can parity voi GEE.

## 4. Luong xu ly representative-month tren STAC

1. query STAC items theo AOI + datetime
2. hard filter item (`VV/VH`, `IW`, `GRD`, asset hop le)
3. annotate geometry coverage cho tung item
4. chia range thanh `calendar month`
5. voi moi thang:
   - cat `pre` va `post`
   - dedupe scene
   - chon pools bang relaxation ladder
   - chon witness support pair
6. download AOI subsets cua scene duoc chon
7. align local ve target grid canonical
8. local `nanmedian` composite cho `VV` va `VH`
9. ap focal median neu bat
10. infer SR

## 5. Spatial semantics tren STAC

Metric chinh:

- `coverage = area(intersection(AOI_geometry, item_geometry)) / area(AOI_geometry)`

Hard gate:

- `coverage > min_aoi_coverage`

Khong con hard gate nao dang yeu cau:

- `AOI bbox coverage = 1.0`
- `bbox_overlap >= nguong`

`bbox_overlap` va `aoi_bbox_coverage_*` van duoc giu de diagnostic.

## 6. Khi nao STAC representative-month se skip

Mot period co the bi skip khi:

- range query khong chua thang day du trong khi `allow_partial_periods=false`
- khong co item o nua dau hoac nua sau
- co item nhung khong level nao trong ladder hop le
- inventory ngheo den muc khong tao duoc pre/post pool can bang

## 7. Trang thai inventory STAC hien tai

Theo suite AOI hien tai:

- 3 AOI Hanoi co `2 items`
- 3 AOI con lai chi co `1 item`
- 1 AOI coastal mixed co `0 item`

Y nghia:

- STAC hien hop de debug coverage/intersection logic
- STAC hien chua hop de ket luan production quality cho representative-month

## 8. Khi nao co the quay lai STAC cho production

Nen quay lai STAC khi inventory da dap ung toi thieu:

- co item o ca 2 nua thang tren cac AOI production
- representative ladder thuong xuyen thanh cong o it nhat level 1 hoac 2
- geometry coverage / union coverage cua pools khong qua thieu
- output distribution va visual QA gan voi GEE tren nhung AOI doi chung

## 9. Checklist parity voi GEE

Khi so STAC voi GEE, nen check:

- cung `period_id`
- cung split `pre/post`
- `selected_relaxation_name` co hop ly / tuong dong
- pre/post scene counts co cung cap logic
- union coverage diagnostics khong lech bat thuong
- witness pair co y nghia tuong dong
- output SR khong bi shift distribution vo ly

## 10. Lenh debug STAC huu ich

### 10.1 Representative-month pipeline

```bash
python sar_pipeline.py \
  --mode stac_trainlike_composite \
  --geojson geojson/generated_aoi/aoi_suite_hanoi_urban_square.geojson \
  --config config/pipeline_config.yaml \
  --datetime 2025-01-01/2025-12-31 \
  --output-dir runs/customer_jobs_stac
```
