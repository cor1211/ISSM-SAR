# How To Use ISSM-SAR Pipeline (Production + Debug)

## 1. Muc tieu tai lieu

Tai lieu nay huong dan cach chay he thong theo dung luong production hien tai, dong thoi mo ta ro cac mode debug va cong cu doi chieu.

Script chinh:

- sar_pipeline.py

Script ho tro:

- query_stac_download.py
- gee_compare_download.py
- gee_trainlike_download.py

## 2. Chon workflow nao

## 2.1 Production khuyen nghi

Dung:

- stac_trainlike_composite

Ly do:

- phu hop domain input model hien tai hon exact raw pair
- co compositing tren timeline (giam nhieu, on dinh hon)

## 2.2 Khi nao dung exact_pair

Dung de:

- debug pair selection
- benchmark voi luong cu
- test nhanh khong can composite

## 3. Chay nhanh production

Lenh co ban:

```bash
python sar_pipeline.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --datetime 2025-07-01/2025-09-10
```

Giai thich:

- mode duoc lay tu config (mac dinh stac_trainlike_composite)
- datetime nen de du rong de co du support pair va scene trong windows

## 4. Chay voi override tham so

## 4.1 Train-like windows

```bash
python sar_pipeline.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --mode stac_trainlike_composite \
  --datetime 2025-07-01/2025-09-10 \
  --window-before-days 30 \
  --window-after-days 30 \
  --min-scenes-per-window 1
```

## 4.2 Pair constraints (chu yeu cho exact_pair)

```bash
python sar_pipeline.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --mode exact_pair \
  --min-delta-hours 24 \
  --max-delta-days 10 \
  --min-aoi-coverage 1.0 \
  --same-orbit-direction
```

## 5. Cong thuc va dieu kien can nho

## 5.1 AOI coverage

coverage = area(intersection(AOI_bbox, item_bbox)) / area(AOI_bbox)

Mac dinh:

- min_aoi_coverage = 1.0

## 5.2 Exact pair time gate

Dieu kien:

- min_delta_hours <= delta <= max_delta_days

Trong code:

- delta tinh theo giay
- min_delta_hours * 3600
- max_delta_days * 86400

## 5.3 Train-like support pair time gate

Trong suggest_trainlike_anchors:

- delta >= anchor_min_delta_hours (hoac pairing.min_delta_hours)
- delta <= window_before_days + window_after_days

Y nghia:

- support pair duoc rang buoc boi do rong timeline ma windows can cover

## 5.4 Window convention

Voi anchor A:

- pre (S1T2): [A - before, A)
- post (S1T1): [A, A + after]

## 5.5 Composite

Moi polarization:

- warp scene ve 1 grid chung
- nanmedian(scene stack)
- focal_median(radius_m)

## 6. Cac file output

Moi run tao:

- runs/pipeline/<aoi_stem>/<run_id>/manifest.json
- runs/pipeline/<aoi_stem>/<run_id>/run_summary.json
- runs/pipeline/<aoi_stem>/<run_id>/run_summary.md
- runs/pipeline/<aoi_stem>/<run_id>/output/*.tif

Rieng train-like:

- window_raw/pre/*.tif
- window_raw/post/*.tif
- composite/s1t1_*.tif
- composite/s1t2_*.tif

## 7. Cach doc run_summary

## 7.1 Neu mode exact_pair

Kiem tra:

- selected_pair.latest_input_datetime
- selected_pair.delta_days
- selected_pair.aoi_bbox_coverage_t1/t2
- selection_profile (strict/balanced/loose)

## 7.2 Neu mode stac_trainlike_composite

Kiem tra:

- anchor.anchor_datetime
- anchor.latest_input_datetime
- anchor.pre_scene_count / post_scene_count
- composite.pre/post.scene_counts
- run_config.window_before_days / window_after_days

## 8. Troubleshooting nhanh

## 8.1 Loi no valid pair found

Huong xu ly:

1. Mo rong --datetime
2. Giam --min-aoi-coverage neu AOI kho
3. Giam --min-delta-hours
4. Bat --auto-relax (exact_pair)
5. Neu train-like: giam --min-scenes-per-window

## 8.2 Loi no valid STAC anchor candidate

Thu:

1. Mo rong datetime range
2. Giam min-scenes-per-window
3. Tat same orbit neu dang bat
4. Kiem tra STAC source co du VV/VH khong

## 8.3 Output chat luong kem

Can xem:

- compatibility profile co mismatch khong
- so scene pre/post qua it
- windows qua hep
- STAC timeline qua sparse

## 9. Kiem tra pair/anchor doc lap

Dung query_stac_download.py de debug truoc khi chay infer:

- inspect items
- suggest anchors
- xem diagnostics no_pair

Vi du:

```bash
python query_stac_download.py --help
```

## 10. Cong cu doi chieu voi GEE

## 10.1 gee_compare_download.py

Dung de tim cap GEE gan voi system pair STAC.

## 10.2 gee_trainlike_download.py

Dung de tao train-like pair tren GEE windows roi so sanh voi STAC train-like.

Luu y:

- 2 script nay la diagnostic/benchmark, khong phai luong production chinh.

## 11. Checklist truoc khi chay production

1. config/pipeline_config.yaml da dat mode dung
2. STAC url va collection dung
3. datetime du rong
4. AOI geojson hop le
5. target_crs va target_resolution dung nhu mong muon
6. du quyen ghi vao output dir

