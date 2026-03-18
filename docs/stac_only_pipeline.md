# STAC-Only Pipeline (Production Notes)

## 1. Dinh huong production

He thong production hien tai uu tien STAC-only, cu the:

- workflow.mode = stac_trainlike_composite

Muc tieu:

- giam phu thuoc GEE cho luong van hanh
- xu ly du lieu dau vao on-site qua STAC + S3
- tao input train-like bang local compositing

## 2. Luong STAC-only end-to-end

1. Query STAC items theo AOI + datetime
2. Hard filter item (IW, GRD, VV/VH, asset hop le)
3. Sinh support pairs va de xuat anchor
4. Chon anchor theo ranking latest_input_datetime
5. Tao pre/post windows quanh anchor
6. Download scene trong windows (subset AOI)
7. Warp tat ca ve grid chung
8. nanmedian per-pol per-window
9. focal median smoothing
10. Tao s1t1/s1t2 composite 2-band
11. Infer multiband

## 3. Cac cong thuc quan trong

## 3.1 Coverage

coverage = area(intersection(AOI_bbox, item_bbox)) / area(AOI_bbox)

Mac dinh min_aoi_coverage = 1.0.

## 3.2 Support pair time gate cho anchor

Trong train-like mode:

- delta >= anchor_min_delta_hours (neu khong set thi fallback pairing.min_delta_hours)
- delta <= window_before_days + window_after_days

Luu y:

- day la logic theo suggest_trainlike_anchors
- pairing.max_delta_days la gate cua exact_pair, khong la tran chinh trong support pair cua train-like

## 3.3 Window convention

Voi anchor A:

- pre (S1T2): [A - before, A)
- post (S1T1): [A, A + after]

## 3.4 Composite

Cho moi polarization:

- C = nanmedian(scene_stack_aligned)
- C_smooth = focal_median(C, radius_m)

## 4. Vi sao STAC-only van train-like

So voi exact raw pair:

- dung nhieu scene thay vi 1 scene
- co trung vi theo window de on dinh noise
- co focal median de giam speckle
- dau ra 2-band giu dung expectation infer_production

## 5. So sanh voi nhom GEE tools

GEE tools:

- gee_compare_download.py
- gee_trainlike_download.py

Vai tro:

- benchmark, doi chieu, phan tich domain

Khong phai luong production chinh do:

- phu thuoc EE project/auth
- khong phai data path van hanh chinh tai he thong STAC noi bo

## 6. Config production de xuat

Tu config hien tai:

- mode: stac_trainlike_composite
- window_before_days: 30
- window_after_days: 30
- min_scenes_per_window: 1
- target_crs: EPSG:3857
- target_resolution: 10
- focal_median_radius_m: 15
- min_aoi_coverage: 1.0
- min_delta_hours: 24

## 7. Checklist truoc khi run

1. STAC service online, collection dung
2. datetime range du rong
3. AOI hop le
4. S3 download truy cap duoc
5. output dir co quyen ghi
6. infer dependencies da cai

## 8. Monitoring sau khi run

Xem run_summary.json/md de check:

- anchor datetime
- latest_input_datetime
- pre/post scene counts
- support pair delta
- output path

Neu fail anchor:

- mo rong datetime
- giam min_scenes_per_window
- xem lai same_orbit_direction

## 9. Lenh production mau

```bash
python sar_pipeline.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --mode stac_trainlike_composite \
  --datetime 2025-07-01/2025-09-10
```

