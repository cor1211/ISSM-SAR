# sar_pipeline.py Guide (Code-Accurate)

## 1. Muc tieu

sar_pipeline.py la wrapper end-to-end cho 2 workflow:

- exact_pair
- stac_trainlike_composite

Gia tri mac dinh trong config hien tai:

- workflow.mode = stac_trainlike_composite (xem config/pipeline_config.yaml)

Luon tham chieu code chinh:

- sar_pipeline.py
- query_stac_download.py

## 2. Tong quan 2 workflow

### 2.1 exact_pair

Muc tieu: tim 1 cap T1/T2 hop le de infer truc tiep.

Luong xu ly:

1. Query STAC + hard filter item
2. Tim pair theo rang buoc thoi gian, coverage, orbit (tuy chon)
3. Chon pair top-1 theo ranking recency
4. Download 4 file 1-band: t1_vv, t1_vh, t2_vv, t2_vh
5. Align/staging theo infer_production
6. Infer

### 2.2 stac_trainlike_composite

Muc tieu: tao input train-like tu STAC timeline (multi-scene compositing), sau do infer.

Luong xu ly:

1. Query STAC + hard filter item
2. Sinh support pairs de de xuat anchor
3. Chon 1 anchor theo ranking recency
4. Tao 2 cua so thoi gian quanh anchor: pre/post
5. Download nhieu scene trong moi window
6. Warp ve 1 grid chung
7. Composite per-pol = nanmedian(scene stack)
8. Focal median smoothing
9. Ghi 2-band t1/t2 composite
10. Infer multiband

Luu y quan trong:

- Mode stac_trainlike_composite KHONG goi nhanh select_best_pair cua exact_pair.
- No tuong tac truc tiep voi suggest_trainlike_anchors.

## 3. Hard filters va cong thuc co ban

## 3.1 Hard filter item (ap dung truoc khi pairing)

Hard filters duoc ap trong query_stac_download.py:

- instrument_mode = IW
- product_type = GRD
- item co du polarization bat buoc (mac dinh VV,VH)
- item co asset href hop le cho cac polarization can dung
- neu config co orbit/rel_orbit thi phai match

## 3.2 Cong thuc coverage va overlap

Cho 2 bbox:

- inter = dien tich giao nhau
- area(b) = dien tich bbox

Cong thuc:

- AOI coverage:
  C = inter(AOI_bbox, item_bbox) / area(AOI_bbox)
- Pair overlap (diagnostic):
  O = inter(bbox_t1, bbox_t2) / min(area(bbox_t1), area(bbox_t2))

Luu y:

- min_aoi_coverage la hard filter
- min_overlap hien khong phai hard filter trong find_pairs, chi duoc luu de report/ranking

## 4. Exact pair: dieu kien, ranking, fallback

## 4.1 Dieu kien pair hop le

Cap (t1, t2) voi t2 muon hon t1 duoc giu neu:

- min_delta_hours <= delta_time <= max_delta_days
- coverage_t1 >= min_aoi_coverage
- coverage_t2 >= min_aoi_coverage
- neu same_orbit_direction = true thi orbit_state(t1) phai bang orbit_state(t2)

Quy doi thoi gian trong code:

- min_delta_sec = min_delta_hours * 3600
- max_delta_sec = max_delta_days * 86400

## 4.2 Ranking exact pair

pair_rank_key:

1. t2_datetime moi nhat
2. t1_datetime moi nhat
3. delta_seconds nho hon
4. bbox_overlap lon hon
5. t1_id, t2_id de on dinh thu tu

## 4.3 Auto relax

Neu khong tim thay strict pair va pairing.auto_relax = true:

- thu lai voi max_delta_days = 30 (balanced)
- neu van khong co, thu max_delta_days = 90 (loose)

Neu auto_relax = false:

- fail ngay voi thong bao chan doan reason tu diagnose_no_pair

## 5. Train-like composite: anchor logic dung theo code

## 5.1 Support pair cho anchor

Trong suggest_trainlike_anchors:

- support pair van phai dat anchor_min_delta_hours (fallback pairing.min_delta_hours), min_aoi_coverage, same_orbit_direction (neu bat)
- Gioi han support gap:
  delta_sec <= (window_before_days + window_after_days) * 86400

Luu y:

- O mode nay, pairing.max_delta_days KHONG duoc dung de cat support pair.
- Gioi han duoc rang buoc boi tong do rong cua 2 windows quanh anchor.

## 5.2 Anchor candidate

Voi moi support pair hop le:

- anchor = midpoint(t1_datetime, t2_datetime)
- pre window = [anchor - window_before_days, anchor)
- post window = [anchor, anchor + window_after_days]

Chi giu scene trong window neu:

- item_bbox hop le
- coverage(item, AOI) >= min_aoi_coverage

Sau do dedupe scene theo key:

- (datetime, platform, orbit_state, relative_orbit, slice_number)

Candidate hop le khi:

- so scene unique pre >= min_scenes_per_window
- so scene unique post >= min_scenes_per_window

## 5.3 Ranking anchor

anchor_rank_key uu tien:

1. post_latest_scene_datetime moi nhat (latest input)
2. anchor_datetime moi nhat
3. support_t2_datetime moi nhat
4. pre_latest_scene_datetime moi nhat
5. support_pair_delta_seconds nho hon
6. support_t1_id, support_t2_id de on dinh

## 5.4 Auto-relax scene count

choose_anchor_candidate chay required_count giam dan:

- bat dau tu trainlike.min_scenes_per_window
- neu khong co candidate va auto_relax_min_scenes = true thi giam den 1
- neu auto_relax_min_scenes = false thi dung ngay o nguong ban dau

## 6. Download, grid, composite

## 6.1 Download window assets

Moi scene trong pre/post window:

- download theo pol VV, VH
- subset theo AOI geometry
- luu vao window_raw/pre hoac window_raw/post

## 6.2 Build grid dich

Grid tao tu AOI bbox voi target_crs, target_resolution:

- left = floor(minx / xres) * xres
- right = ceil(maxx / xres) * xres
- bottom = floor(miny / yres) * yres
- top = ceil(maxy / yres) * yres
- width = round((right - left) / xres)
- height = round((top - bottom) / yres)

Transform:

- Affine(xres, 0, left, 0, -yres, top)

## 6.3 Composite per window

Cho moi polarization (VV, VH):

1. Reproject moi scene ve cung grid
2. Stack scene
3. composite = nanmedian(stack)
4. Neu con NaN cuc bo: fill bang median cua phan finite (neu rong thi 0)
5. focal median voi ban kinh:
   radius_px = focal_median_radius_m / resolution_m

Sau do ghi GeoTIFF 2-band voi descriptions:

- S1_VV
- S1_VH

## 7. Inference

- exact_pair: run_pair_from_single_band_files
- stac_trainlike_composite: run_pair_from_multiband_files

Output cuoi:

- output/<aoi_stem>__<pair_or_anchor_id>_SR_x2.tif

## 8. Config chinh (theo config/pipeline_config.yaml)

### 8.1 workflow

- mode: exact_pair | stac_trainlike_composite

### 8.2 stac

- url
- collection
- datetime
- limit

### 8.3 pairing

- pols
- orbit
- rel_orbit
- min_overlap (diagnostic)
- min_aoi_coverage
- min_delta_hours
- max_delta_days (dung trong exact_pair)
- same_orbit_direction
- strict_slice (co truyen vao ham pair; hien tai find_pairs khong su dung de hard-filter)
- auto_relax

### 8.4 trainlike

- window_before_days
- window_after_days
- min_scenes_per_window
- auto_relax_min_scenes
- anchor_pick_index
- same_orbit_direction
- anchor_min_delta_hours
- target_crs
- target_resolution
- resampling
- focal_median_radius_m
- window_raw_dir_name
- composite_dir_name

### 8.5 compatibility

Mac dinh:

- trained_input_profile: gee_s1_db
- current_download_profile: stac_trainlike_composite_db
- allow_domain_mismatch: false

Y nghia:

- exact raw STAC va train-like composite duoc xem la 2 profile khac nhau

## 9. CLI override trong sar_pipeline.py

Cac tham so hay dung:

- --mode
- --datetime
- --min-delta-hours
- --max-delta-days
- --min-aoi-coverage
- --same-orbit-direction
- --auto-relax
- --window-before-days
- --window-after-days
- --min-scenes-per-window
- --target-crs
- --target-resolution
- --focal-median-radius-m
- --device
- --output-dir
- --cache-staging

## 10. Vi du lenh chay

### 10.1 Production train-like (khuyen nghi)

```bash
python sar_pipeline.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --datetime 2025-07-01/2025-09-10
```

### 10.2 Train-like voi override windows

```bash
python sar_pipeline.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --mode stac_trainlike_composite \
  --datetime 2025-07-01/2025-09-10 \
  --window-before-days 30 \
  --window-after-days 30 \
  --min-scenes-per-window 1
```

### 10.3 Exact pair debug

```bash
python sar_pipeline.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --mode exact_pair
```

## 11. Cau truc output

Moi run tao:

- runs/pipeline/<aoi_stem>/<run_id>/manifest.json
- runs/pipeline/<aoi_stem>/<run_id>/run_summary.json
- runs/pipeline/<aoi_stem>/<run_id>/run_summary.md

Rieng train-like:

- runs/pipeline/<aoi_stem>/<run_id>/window_raw/pre/*.tif
- runs/pipeline/<aoi_stem>/<run_id>/window_raw/post/*.tif
- runs/pipeline/<aoi_stem>/<run_id>/composite/s1t1_<pair_id>.tif
- runs/pipeline/<aoi_stem>/<run_id>/composite/s1t2_<pair_id>.tif

