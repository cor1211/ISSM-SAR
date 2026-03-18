# gee_trainlike_download.py Guide (Code-Accurate)

## 1. Muc tieu

gee_trainlike_download.py tao cap train-like tren GEE dua theo system pair STAC.

No mo phong tien xu ly train/test:

- tao cua so thoi gian quanh anchor
- median composite
- focal median smoothing
- export 2-band TIFF tuong thich infer_production

Day la tool benchmark/doi chieu, khong phai production STAC-only chinh.

## 2. Dau vao va nguon system reference

Script nhan:

- --geojson
- --pipeline-config
- --gee-config
- --manifest (optional)

Neu co --manifest:

- dung pair/anchor trong manifest

Neu khong co --manifest:

- goi load_system_manifest -> select_best_pair trong sar_pipeline.py

## 3. Anchor strategies

Ham resolve_anchor ho tro:

- midpoint
- t1
- t2
- anchor / precomputed / manifest

Cong thuc midpoint:

- anchor = t1 + (t2 - t1) / 2

Co uu tien lay anchor precomputed trong manifest neu strategy la anchor/manifest.

## 4. Window definition

Ham resolve_window:

- start = anchor + start_day
- end = anchor + end_day
- yeu cau end > start

Mac dinh tu config/gee_compare_config.yaml:

- t1_window_days = [0, 7]
- t2_window_days = [-7, 0]

Neu manifest trainlike co window_before/after thi script map thanh:

- t1_window = [0, window_after_days]
- t2_window = [-window_before_days, 0]

Co the override bang CLI:

- --t1-window-days START END
- --t2-window-days START END

## 5. Build GEE collection

Moi nhanh T1/T2 dung build_collection:

- filterBounds(filter_geom)
- filterDate(start, end)
- listContains(VV)
- listContains(VH)
- instrumentMode = IW
- neu orbit_pass != BOTH thi filter orbitProperties_pass

Co the chon orbit pass:

- BOTH (mac dinh)
- ASCENDING
- DESCENDING

## 6. Composite va smoothing

build_trainlike_image:

1. image = collection.select([VV, VH]).median()
2. neu focal_median_radius_m > 0:
   image = image.focal_median(radius_m, circle, meters)
3. clip(image, clip_geom)

clip_mode:

- bbox
- geometry

Mac dinh: bbox.

## 7. Grid va export

Grid tao tu system_manifest.aoi_bbox:

- CRS default: EPSG:3857
- scale default: 10m
- dimensions + transform snap theo build_target_grid

Export:

- GEO_TIFF
- filePerBand false
- bands VV,VH
- sau do rewrite voi descriptions S1_VV,S1_VH

## 8. Validation va inference tuy chon

Sau export, validate_pair kiem:

- t1/t2 cung grid
- dung expected grid
- dung band descriptions
- infer scan pair OK

Neu --run-infer:

- script goi infer_production voi mode multiband
- output vao infer_output/<pair_id>_SR_x2.tif

## 9. Report outputs

Moi run tao:

- system_reference_manifest.json
- s1t1_<pair_id>.tif
- s1t2_<pair_id>.tif
- gee_trainlike_report.json
- gee_trainlike_report.md
- infer_output/*.tif (neu --run-infer)

Report chua:

- thong tin system reference
- anchor strategy + anchor datetime
- T1/T2 windows
- so luong scene dong gop moi window
- export grid
- validation
- inference output (neu co)

## 10. Lenh chay

### 10.1 Mac dinh

```bash
python gee_trainlike_download.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --pipeline-config config/pipeline_config.yaml \
  --gee-config config/gee_compare_config.yaml
```

### 10.2 Override anchor/window

```bash
python gee_trainlike_download.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --anchor-strategy midpoint \
  --t1-window-days 0 14 \
  --t2-window-days -14 0 \
  --orbit-pass BOTH \
  --focal-median-radius-m 15
```

### 10.3 Chay kem inference

```bash
python gee_trainlike_download.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --run-infer \
  --infer-config config/infer_config.yaml
```

## 11. Cac loi thuong gap

- No GEE images found for train-like T1/T2 window:
  - mo rong window
  - doi orbit_pass ve BOTH
- Missing GEE project:
  - truyen --gee-project hoac cap nhat config
- Authentication fail:
  - dung --authenticate

