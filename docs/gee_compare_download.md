# gee_compare_download.py Guide (Code-Accurate)

## 1. Muc tieu

gee_compare_download.py tao cap GEE Sentinel-1 de doi chieu voi system pair do STAC chon.

Tool nay KHONG thay the luong production STAC. No dung cho:

- benchmark
- domain check
- xac minh su khac nhau giua STAC pair va GEE pair gan nhat

## 2. Nguon system pair

Tool nhan system pair theo 2 cach:

1. Neu co --manifest:
   - doc pair tu manifest cung cap
2. Neu khong co --manifest:
   - goi lai sar_pipeline.select_best_pair de tu tim pair theo pipeline config

Thong tin pair can co:

- pair_id
- t1_id, t2_id
- t1_datetime, t2_datetime
- aoi_bbox

## 3. GEE candidate matching

## 3.1 Collection filter co ban

Cho tung moc thoi gian target (t1 hoac t2):

- collection: COPERNICUS/S1_GRD (default trong config)
- filterBounds(AOI bbox)
- filterDate(target - window_minutes, target + window_minutes)
- instrumentMode = IW
- phai co VV va VH

Mac dinh:

- match_window_minutes = 30

## 3.2 Dieu kien coverage

Candidate duoc giu khi footprint GEE covers toan bo AOI bbox polygon.

Luu y:

- Day la dieu kien manh hon intersect.

## 3.3 Ranking candidate GEE

Covered candidates duoc sort theo:

1. exact_id_match uu tien cao nhat
   - system:index == STAC id hoac system:index bat dau bang "<stac_id>_"
2. delta_to_target_seconds nho hon
3. orbit_match uu tien
4. system_index de on dinh

## 3.4 Cac ly do fail chinh

- NO_GEE_IMAGE_IN_TIME_WINDOW
- MISSING_VV_VH
- AOI_NOT_FULLY_COVERED

Diagnostics duoc ghi trong report JSON.

## 4. Grid va export

Grid dich duoc xay tu system aoi_bbox:

- CRS mac dinh: EPSG:3857
- scale mac dinh: 10m
- snap outward theo grid

Export params:

- format: GEO_TIFF
- filePerBand: false
- bands: VV, VH
- dimensions: [width, height]
- crs_transform tu grid

Sau khi download:

- rewrite lai TIFF voi band descriptions:
  - S1_VV
  - S1_VH
- compress DEFLATE
- validate cung grid giua t1/t2

## 5. Output

Moi run tao:

- system_reference_manifest.json
- s1t1_<pair_id>.tif
- s1t2_<pair_id>.tif
- gee_compare_report.json
- gee_compare_report.md

## 6. Cac metric report quan trong

System reference:

- pair id
- delta hours/days
- aoi coverage t1/t2

GEE match:

- gee id t1/t2
- gee datetime t1/t2
- orbit pass t1/t2
- exact id match true/false
- delta_to_target_seconds
- coverage_ratio

Validation:

- same_grid
- matches_expected_grid
- matches_expected_descriptions
- pair_scan_ok (scan boi infer_production)

## 7. Lenh chay

```bash
python gee_compare_download.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --pipeline-config config/pipeline_config.yaml \
  --gee-config config/gee_compare_config.yaml
```

Co the lock pair bang manifest:

```bash
python gee_compare_download.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --manifest runs/pipeline/<aoi>/<run>/manifest.json \
  --pipeline-config config/pipeline_config.yaml \
  --gee-config config/gee_compare_config.yaml
```

## 8. Config lien quan (config/gee_compare_config.yaml)

Section gee:

- project
- collection
- export_crs
- export_scale
- match_window_minutes
- band_names
- output_band_descriptions

Section output:

- root_dir

Section trainlike duoc gee_trainlike_download.py su dung, khong anh huong truc tiep logic match cua gee_compare.

## 9. Khi nao dung tool nay

Nen dung khi can:

- kiem tra su lech domain STAC vs GEE
- tao cap doi chieu nhanh de visualize
- benchmark ket qua infer tren cap GEE gan voi pair STAC

Khong nen dung nhu luong production chinh vi:

- phu thuoc Earth Engine
- mang tinh benchmark/diagnostic

