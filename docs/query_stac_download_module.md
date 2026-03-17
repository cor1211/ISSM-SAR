# query_stac_download.py Module Guide

## 1. Module nay dung de lam gi?

`query_stac_download.py` dung de:

- doc AOI tu `--geojson` hoac `--bbox`
- query STAC Sentinel-1 GRD theo khong gian va thoi gian
- loc item theo `IW`, `GRD`, polarization, orbit filter neu user muon
- tim cap `T1/T2` phu hop cho bai toan multi-temporal SAR
- de xuat `anchor` cho workflow `stac_trainlike_composite`
- xuat manifest href tren S3/HTTP
- download `4 file 1-band` theo format tu nhien cua STAC:
  - `T1_VV`
  - `T1_VH`
  - `T2_VV`
  - `T2_VH`
- subset raster theo phan giao giua AOI bbox va item bounds

Module nay hien duoc thiet ke de lam tang query/download cho pipeline end-to-end.

## 2. Dau vao can co

### 2.1 Bien moi truong

Doc tu `.env`:

- `STAC_API_URL`: STAC API URL
- `S3_ENDPOINT`: endpoint S3/MinIO
- `S3_ACCESS_KEY`, `S3_SECRET_KEY`: credential

### 2.2 AOI

Module nhan 1 trong 2 kieu:

- `--geojson path/to/aoi.geojson`
- `--bbox LON_MIN LAT_MIN LON_MAX LAT_MAX`

Neu dung `--geojson`, script se:

- doc geometry
- tu tinh `bbox` neu file khong co san
- query bang `intersects`
- neu backend reject `intersects`, fallback sang `bbox`

## 3. Hard filters tren item

Item phai dat tat ca:

- `sar:instrument_mode == IW`
- `sar:product_type == GRD`
- chua du polarization yeu cau trong `--pols`
- moi polarization yeu cau phai co asset raster hop le

Filter tuy chon:

- `--orbit`: loc item theo `sat:orbit_state`
- `--rel-orbit`: loc item theo `sat:relative_orbit`

Hai filter nay chi loc item dau vao. Chung khong mac dinh chi phoi pairing.

## 4. Thuat toan pairing hien tai

### 4.1 Cach sinh candidate

Sau hard filter, tat ca item duoc sap xep theo `properties.datetime`, roi so tung cap `i < j`.

Mac dinh:

- khong group theo orbit
- khong group theo slice
- khong bat buoc cung `relative_orbit`
- khong bat buoc cung `datatake_id`

Mode tuy chon:

- `--same-orbit-direction`: chi chap nhan cap co `sat:orbit_state` giong nhau

### 4.2 Dieu kien 1 cap hop le

1 cap `T1/T2` duoc giu lai neu dat tat ca:

- `min_delta_hours <= Delta_t <= max_delta_days`
- `AOI_bbox_coverage_t1 >= min_aoi_coverage`
- `AOI_bbox_coverage_t2 >= min_aoi_coverage`

Trong do:

- `Delta_t`: chenh lech thoi gian giua `properties.datetime` cua `T1` va `T2`
- `AOI_bbox_coverage_t1`, `AOI_bbox_coverage_t2`:
  `intersection(AOI bbox, item bbox) / area(AOI bbox)`

Y nghia:

- vi input vao model cuoi cung la 1 anh hinh chu nhat, dung `AOI bbox coverage` se dam bao ca `T1` va `T2` deu bao tron vung chu nhat ma model thuc su nhin thay
- day la ly do module hien tai van giu coverage theo bbox, khong doi sang polygon coverage

### 4.3 `bbox_overlap` dung de lam gi?

`bbox_overlap` hien **khong con la hard filter**.

No chi duoc dung cho:

- report
- diagnostic
- tie-break ranking khi 2 cap co `Delta_t` giong nhau

Cong thuc:

- `intersection(item_bbox_1, item_bbox_2) / min(area1, area2)`

### 4.4 Ranking

Cap hop le duoc sort theo uu tien moi nhat:

1. `latest_input_datetime` moi nhat
2. `T1` moi nhat
3. `Delta_t` nho hon
4. `bbox_overlap` lon hon
5. `T1 ID`, `T2 ID` on dinh de tranh random

Voi `exact pair`, `latest_input_datetime` chinh la `T2 datetime`, vi `T2` la anh muon hon trong cap.

Y nghia:

- khi he thong nhan 1 AOI moi tu khach hang, cap duoc chon se uu tien gan hien tai nhat
- `bbox_overlap` khong duoc phep day 1 cap cu hon len tren 1 cap moi hon
- `Delta_t` van quan trong, nhung chi dung de tie-break trong nhom cap da gan hien tai nhat

## 5. Default pairing hien tai

Mac dinh moi:

- `--pols VV,VH`
- `--min-aoi-coverage 1.0`
- `--min-delta-hours 24`
- `--max-delta-days 10`
- `--same-orbit-direction` tat

`--min-overlap` van con tren CLI de tuong thich nguoc, nhung hien chi la tham so diagnostic/ranking, khong con loai pair.

## 6. Auto-relax hoat dong the nao?

Lenh `prepare` va `download-generated-pairs` co the bat `--auto-relax`.

Script thu lan luot:

- `strict`: dung threshold user truyen
- `balanced`: giu `min_aoi_coverage`, doi `max_delta_days = 30`
- `loose`: giu `min_aoi_coverage`, doi `max_delta_days = 90`

Hai diem quan trong:

- `min_delta_hours` duoc giu nguyen
- `min_aoi_coverage` khong bi ha xuong duoi gia tri user truyen
- `bbox_overlap` van chi la diagnostic/ranking

## 7. Download dang tai cai gi?

Mac dinh script khong tai full scene.

No se:

- mo remote raster tren S3/HTTP bang rasterio
- doi AOI geometry sang CRS raster
- lay `bbox giao nhau giua AOI va item bounds`
- snap outward theo pixel grid de khong bi hut goc
- ghi ra file subset

Ket qua cho pipeline la `4 file 1-band`:

- `s1t1_<pair_id>_vv.tif`
- `s1t1_<pair_id>_vh.tif`
- `s1t2_<pair_id>_vv.tif`
- `s1t2_<pair_id>_vh.tif`

Neu muon tai nguyen full item:

- them `--full-item`

## 8. Quan he voi infer_production.py

Tang query/download va tang infer hien duoc tach ro:

- `query_stac_download.py` tao `4 file raw 1-band`
- `infer_production.py` nay da duoc refactor de co the nhan truc tiep 4 file nay
- infer se tu align/resample ve cung 1 luoi tham chieu, roi moi chay model

Neu bat cache staging, infer co the ghi them:

- `t1_input.tif` gom 2 bands `VV,VH`
- `t2_input.tif` gom 2 bands `VV,VH`

Sau cung output la:

- `1 GeoTIFF 2-band` gom `SR_VV`, `SR_VH`

## 9. Cac lenh chinh

### 9.1 Liet ke item

```bash
python query_stac_download.py list \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson
```

### 9.2 Liet ke pair hop le

```bash
python query_stac_download.py pair \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson
```

Voi default moi, AOI nay se cho 1 pair strict hop le vi:

- `Delta_t ~ 2.51 days`
- `AOI bbox coverage T1 = 1.0`
- `AOI bbox coverage T2 = 1.0`
- `bbox_overlap ~ 0.683` chi la diagnostic, khong con chan pair

### 9.3 Chon cap tot nhat va download raw inputs

```bash
python query_stac_download.py prepare \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --download \
  --out-dir data/input
```

### 9.4 Batch pair download

```bash
python query_stac_download.py download-generated-pairs \
  --geojson-dir geojson/generated_aoi \
  --dry-run
```

### 9.5 De xuat anchor cho pipeline train-like

Khi can suy ra `anchor` tu STAC timeline, co the dung:

```bash
python query_stac_download.py suggest-anchor \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --datetime 2025-07-01/2025-09-10 \
  --window-before-days 30 \
  --window-after-days 30 \
  --min-scenes-per-window 1 \
  --save-manifest runs/anchor_manifests/ffa6dc_anchor_30d.json
```

Lenh nay:

- van query STAC theo AOI nhu cu
- loc item full AOI + du polarization
- dung midpoint cua cac support pair lam anchor candidates
- xep hang theo `latest_input_datetime` moi nhat
- chi dung `min_scenes_per_window` nhu nguong hop le toi thieu, khong phai muc tieu toi uu chinh
- xuat `anchor manifest`

Mac dinh production hien tai:

- `window_before_days = 30`
- `window_after_days = 30`
- `min_scenes_per_window = 1`

`anchor manifest` se chua:

- `anchor_datetime`
- `latest_input_datetime`
- `window_before_days`, `window_after_days`
- `support_t1/t2`
- `pre_scenes`, `post_scenes`
- `aoi_bbox`

Ghi chu:

- `latest_input_datetime` la moc du lieu moi nhat ma pipeline uu tien de tra SR cho khach hang
- `anchor_datetime` la moc noi bo de chia 2 window train-like, khong phai tieu chi xep hang chinh
- hien tai manifest nay duoc `sar_pipeline.py` dung truc tiep cho workflow `stac_trainlike_composite`
- voi huong production moi, cung logic `anchor` nay duoc `sar_pipeline.py` goi noi bo trong mode `stac_trainlike_composite` de tai **window scenes tren STAC** va composite local

## 10. Report co gi moi?

Report pair hien tai ghi ro:

- run config day du
- `Delta exact`, `Delta hours`, `Delta days`
- `AOI bbox coverage T1/T2/min`
- `bbox_overlap` voi vai tro diagnostic
- orbit direction, relative orbit, slice, datatake
- no-pair reason va thong ke diagnostic
