# `sar_pipeline.py` Guide

## 1. Muc tieu

`sar_pipeline.py` la wrapper end-to-end cho 2 workflow:

- `exact_pair`
- `stac_trainlike_composite`

Workflow production hien tai duoc uu tien la:

- `stac_trainlike_composite`

vi no bam sat hon domain ma model hien tai can:

- timeline multi-scene
- local median composite
- local focal median
- output `s1t1/s1t2` 2-band roi moi infer

## 2. Hai workflow hien co

### 2.1 `exact_pair`

Luong cu:

1. query STAC
2. chon 1 cap exact `T1/T2`
3. download `4 file 1-band`
4. align
5. infer

Workflow nay van duoc giu de:

- debug
- so sanh
- kiem tra pair selection

Nhung no khong phai workflow toi uu cho model hien tai.

### 2.2 `stac_trainlike_composite`

Luong production moi:

1. query STAC timeline
2. de xuat `anchor`
3. tao `pre/post windows`
4. download nhieu STAC scenes trong 2 windows
5. align ve cung grid `EPSG:3857`, `10m`, `AOI bbox`
6. local `median()` per polarization per window
7. local `focal_median(15m)`
8. tao:
   - `s1t1_<anchor_id>.tif`
   - `s1t2_<anchor_id>.tif`
9. infer

Day la workflow mac dinh moi trong
[pipeline_config.yaml](/mnt/data1tb/vinh/ISSM-SAR/config/pipeline_config.yaml).

## 3. Dau vao

- `AOI geojson`
- `config/pipeline_config.yaml`
- `config/infer_config.yaml`

Vi du AOI:

- [ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson](/mnt/data1tb/vinh/ISSM-SAR/geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson)

## 4. Logic STAC-only production

### 4.1 Query timeline

Pipeline goi lai tang query tu
[query_stac_download.py](/mnt/data1tb/vinh/ISSM-SAR/query_stac_download.py):

- `intersects` theo AOI
- hard filters:
  - `IW`
  - `GRD`
  - du `VV,VH`
  - asset raster doc duoc
  - `AOI bbox coverage = 1.0`

### 4.2 Chon `anchor`

Pipeline khong can user dua san `system_t1/system_t2`.

No tu:

1. sinh `support pairs` tren STAC timeline
2. lay `midpoint(t1, t2)` lam `anchor candidate`
3. tim `latest_input_datetime` cua tung candidate, tuong ung scene moi nhat trong `post window`
4. chon duy nhat 1 `anchor` uu tien moi nhat

Ranking uu tien:

1. `latest_input_datetime` moi nhat
2. `anchor_datetime` moi nhat
3. `support_t2` moi nhat
4. `pre_latest_scene_datetime` moi nhat
5. `support pair gap` nho hon

`scene count` khong con la tieu chi rank chinh.
No chi la dieu kien toi thieu de candidate duoc xem la hop le.

### 4.3 Tao window

Mac dinh production hien tai:

- `window_before_days = 30`
- `window_after_days = 30`

Theo quy uoc model:

- `S1T2 = [anchor - 30d, anchor]`
- `S1T1 = [anchor, anchor + 30d]`

### 4.4 Download scenes

Pipeline download tat ca scene STAC thuoc `pre/post windows`:

- theo tung polarization `VV`, `VH`
- subset theo AOI bbox
- luu vao:
  - `window_raw/pre`
  - `window_raw/post`

Luu y:

- pipeline khong tai full item scene trong mode production nay
- moi asset duoc cat truoc theo `bbox` chua AOI roi moi luu local
- subset window duoc snap outward theo pixel grid de dam bao bao kin AOI

Scene duplicate cung acquisition duoc dedupe de tranh 1 timestamp dong gop nhieu lan.

### 4.5 Align + composite

Tat ca raster duoc warp ve 1 grid chuan:

- `CRS = EPSG:3857`
- `resolution = 10m`
- extent = `AOI bbox`

Sau do:

- `nanmedian()` qua scene stack cho moi polarization
- `focal_median(15m)` voi circular footprint

Ket qua:

- `composite/s1t1_<anchor_id>.tif`
- `composite/s1t2_<anchor_id>.tif`

Moi file co:

- 2 bands
- `S1_VV`
- `S1_VH`

### 4.6 Infer

Pipeline dua 2 file composite vao
[infer_production.py](/mnt/data1tb/vinh/ISSM-SAR/infer_production.py)
qua nhanh `multiband`.

Output cuoi:

- `output/<aoi_stem>__<anchor_id>_SR_x2.tif`

## 5. Config chinh

Config tai:

- [pipeline_config.yaml](/mnt/data1tb/vinh/ISSM-SAR/config/pipeline_config.yaml)

Cac section quan trong:

### 5.1 `workflow`

- `mode`: `exact_pair` hoac `stac_trainlike_composite`

### 5.2 `stac`

- `url`
- `collection`
- `datetime`
- `limit`

### 5.3 `pairing`

Dung cho hard filters va support-pair generation:

- `pols`
- `min_aoi_coverage`
- `min_delta_hours`
- `max_delta_days`
- `same_orbit_direction`

### 5.4 `trainlike`

- `window_before_days`
- `window_after_days`
- `min_scenes_per_window`
- `auto_relax_min_scenes`
- `anchor_pick_index`
- `same_orbit_direction`
- `anchor_min_delta_hours`
- `target_crs`
- `target_resolution`
- `resampling`
- `focal_median_radius_m`
- `window_raw_dir_name`
- `composite_dir_name`

### 5.5 `compatibility`

Mac dinh hien tai:

```yaml
compatibility:
  trained_input_profile: gee_s1_db
  current_download_profile: stac_trainlike_composite_db
  allow_domain_mismatch: false
```

Y nghia:

- exact raw STAC va train-like STAC khong con bi danh dong nham la cung 1 profile
- workflow moi duoc xem la gan hon domain model hon exact raw pair

## 6. Cach chay

### 6.1 Workflow production mac dinh

```bash
python sar_pipeline.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --datetime 2025-07-01/2025-09-10
```

### 6.2 Override window

```bash
python sar_pipeline.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --mode stac_trainlike_composite \
  --datetime 2025-07-01/2025-09-10 \
  --window-before-days 30 \
  --window-after-days 30 \
  --min-scenes-per-window 1
```

### 6.3 Quay lai workflow exact pair

```bash
python sar_pipeline.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --mode exact_pair
```

## 7. Cau truc output

Voi `stac_trainlike_composite`, moi run duoc luu theo dang:

- `runs/pipeline/<aoi_stem>/<run_id>/manifest.json`
- `runs/pipeline/<aoi_stem>/<run_id>/run_summary.json`
- `runs/pipeline/<aoi_stem>/<run_id>/run_summary.md`
- `runs/pipeline/<aoi_stem>/<run_id>/window_raw/pre/*.tif`
- `runs/pipeline/<aoi_stem>/<run_id>/window_raw/post/*.tif`
- `runs/pipeline/<aoi_stem>/<run_id>/composite/s1t1_<anchor_id>.tif`
- `runs/pipeline/<aoi_stem>/<run_id>/composite/s1t2_<anchor_id>.tif`
- `runs/pipeline/<aoi_stem>/<run_id>/output/<aoi_stem>__<anchor_id>_SR_x2.tif`

## 8. Smoke test da chay

Mình da chay end-to-end voi workflow moi.

Co the xem 1 run hien tai tai:

- [run_summary.md](/mnt/data1tb/vinh/ISSM-SAR/runs/customer_jobs/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2/20260317T152309/run_summary.md)

Ket qua:

- query STAC thanh cong
- chon duoc `anchor = 2025-08-04T04:58:57.472844Z`
- download `window_raw/pre/post`
- tao duoc 2 file composite
- infer thanh cong
- ghi duoc output SR hop le

Luu y:

- bo STAC hien tai cho AOI nay van sparse, nen smoke test chi co `pre=1`, `post=1`
- do user da xac nhan STAC se duoc cap day du hon ve sau, workflow duoc thiet ke cho timeline phong phu hon

## 9. Kien nghi hien tai

Voi model hien tai, workflow nen uu tien la:

1. `mode = stac_trainlike_composite`
2. `window_before_days = 30`
3. `window_after_days = 30`
4. `min_scenes_per_window = 1`
5. `target_crs = EPSG:3857`
6. `target_resolution = 10`
7. `focal_median_radius_m = 15`

Neu can debug timeline/anchor logic:

- dung `query_stac_download.py suggest-anchor`
- xem [query_stac_download_module.md](/mnt/data1tb/vinh/ISSM-SAR/docs/query_stac_download_module.md)
