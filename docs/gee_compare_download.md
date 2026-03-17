# gee_compare_download.py Guide

## 1. Muc tieu

`gee_compare_download.py` la luong doi chieu rieng giua:

- cap `T1/T2` duoc he thong ESA/STAC chon
- cap Sentinel-1 dB tu GEE (`COPERNICUS/S1_GRD`) khop vi tri va thoi gian sat nhat

Tool nay khong thay doi hanh vi mac dinh cua `sar_pipeline.py`.
No chi tao them 1 bo du lieu doi chieu de:

- kiem tra do lech mien du lieu giua `system/STAC` va `GEE`
- tai `2 file GeoTIFF 2-band` hop le de dua thang vao `infer_production.py`
- luu report JSON/MD de so sanh cap system va cap GEE

## 2. Dau vao

- AOI geojson, vi du:
  - `geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson`
- Pipeline config de tai tao dung logic chon pair he thong:
  - `config/pipeline_config.yaml`
- GEE compare config:
  - `config/gee_compare_config.yaml`
- Tuy chon `manifest.json` cua run he thong neu muon khoa cung 1 pair tham chieu

## 3. Cach tool chon cap tham chieu

### 3.1 System pair

Tool uu tien:

1. Neu co `--manifest`: doc dung `t1_id`, `t2_id`, `t1_datetime`, `t2_datetime`, `aoi_bbox`
2. Neu khong co `--manifest`: goi lai logic hien tai cua `sar_pipeline.py` de chon cung 1 pair he thong

Voi AOI `ffa6dc...`, cap tham chieu hien tai ky vong la:

- `T1 = S1A_IW_GRDH_1SDV_20250802T225105_20250802T225130_060363_0780A7`
- `T2 = S1A_IW_GRDH_1SDV_20250805T110624_20250805T110649_060400_0781F8`
- `delta = 2.5106469744 days`
- `AOI bbox = [105.86746995207669, 20.993496127644224, 105.92914269078261, 21.05199976877436]`

### 3.2 GEE pair

Moi moc `T1` va `T2` duoc tim doc lap trong `COPERNICUS/S1_GRD` voi chinh sach:

1. exact id match neu `system:index` trung hoac bat dau bang scene id tham chieu
2. neu khong exact, chon anh co `delta_to_target_seconds` nho nhat
3. uu tien cung `orbitProperties_pass` neu map duoc tu system pair
4. bat buoc co du `VV` va `VH`
5. bat buoc footprint bao tron `AOI bbox`

Time window mac dinh:

- `+-30 phut`

Neu khong tim thay, tool fail ro rang voi cac ly do:

- `NO_GEE_IMAGE_IN_TIME_WINDOW`
- `MISSING_VV_VH`
- `AOI_NOT_FULLY_COVERED`

## 4. Dinh dang output

Tool xuat ra 2 file GeoTIFF 2-band:

- `s1t1_<pair_id>.tif`
- `s1t2_<pair_id>.tif`

Moi file co:

- band 1: `S1_VV`
- band 2: `S1_VH`
- `CRS = EPSG:3857`
- `scale = 10m`
- extent = `AOI bbox` cua he thong, duoc snap theo pixel grid

Sau khi tai xong, tool se:

- gan band descriptions dung chuan `S1_VV`, `S1_VH`
- kiem tra `count`, `CRS`, `transform`, `width`, `height`
- goi `SARInferencer._scan_input_dir(...)` de smoke test kha nang tuong thich voi `infer_production.py`

## 5. Cach chay

### 5.1 Dung manifest cua he thong de khoa cung 1 pair

```bash
python gee_compare_download.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --manifest runs/pipeline/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2/20260316T160155/manifest.json \
  --gee-project YOUR_GEE_PROJECT
```

### 5.2 Khong co manifest, de tool tu tai tao system pair

```bash
python gee_compare_download.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --pipeline-config config/pipeline_config.yaml \
  --gee-project YOUR_GEE_PROJECT
```

### 5.3 Bat interactive auth neu may chua login Earth Engine

```bash
python gee_compare_download.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --manifest runs/pipeline/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2/20260316T160155/manifest.json \
  --gee-project YOUR_GEE_PROJECT \
  --authenticate
```

## 6. Cau truc output

Moi run duoc ghi vao:

- `runs/gee_compare/<aoi_stem>/<run_id>/system_reference_manifest.json`
- `runs/gee_compare/<aoi_stem>/<run_id>/s1t1_<pair_id>.tif`
- `runs/gee_compare/<aoi_stem>/<run_id>/s1t2_<pair_id>.tif`
- `runs/gee_compare/<aoi_stem>/<run_id>/gee_compare_report.json`
- `runs/gee_compare/<aoi_stem>/<run_id>/gee_compare_report.md`

## 7. Ghi chu quan trong

- Tool nay la luong doi chieu rieng, khong thay the luong ESA/STAC hien tai.
- Environment hien tai can `earthengine-api` va mot `GEE project` hop le.
- Tool da co patch tuong thich cho mot so env Python 3.9, noi `google.api_core` goi
  `importlib.metadata.packages_distributions()` nhung stdlib chua co API do.
- Neu `ee.Initialize(project=...)` that bai, tool se bao loi ro va huong dan dung `--authenticate`.
- Tool uu tien match exact scene he thong truoc khi fallback sang nearest candidate.
- Output GEE duoc format theo layout 2-band cu de dua thang vao `infer_production.py`.
