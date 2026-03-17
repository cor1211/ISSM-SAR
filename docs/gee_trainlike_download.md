# `gee_trainlike_download.py` Guide

## 1. Muc tieu

`gee_trainlike_download.py` tao bo `S1T1/S1T2` tu GEE theo **domain train/test**
ma model hien tai da hoc, thay vi tai exact acquisition pair.

Ghi chu pham vi:

- day la **nhanh thu nghiem / doi chieu**
- khong phai dinh nghia cuoi cung cho pipeline production
- pipeline production theo yeu cau hien tai phai la **STAC-only**

Tool nay giu lai cac dac diem chinh cua pipeline tao du lieu train/test:

- doc `COPERNICUS/S1_GRD`
- loc `VV`, `VH`, `IW`
- lay `ImageCollection` trong cua so thoi gian
- `median()` theo cua so
- `focal_median(15m)` de giam speckle
- export `EPSG:3857`, `10m`
- ghi ra `2-band GeoTIFF` hop le cho `infer_production.py`

Khac biet la: bo train/test goc co **mot ngay tham chieu tu nhien** (`SRC_DATE`),
con tool nay phai **suy ra ngay tham chieu** tu system pair STAC da duoc chon.
Ngay tham chieu duoc goi la `anchor`.

## 2. `anchor` la gi

`anchor` la **moc thoi gian tham chieu** de dat 2 cua so:

- cua so tao `S1T1`
- cua so tao `S1T2`

Cong thuc:

- `S1T1` dung cua so `anchor + t1_window_days`
- `S1T2` dung cua so `anchor + t2_window_days`

Mac dinh:

- `t1_window_days = [0, 7]`
- `t2_window_days = [-7, 0]`

Nghia la:

- `S1T1` = cua so **sau anchor**
- `S1T2` = cua so **truoc anchor**

## 3. Can phan biet 2 he quy chieu thoi gian

Day la diem de nham nhat.

### 3.1 System pair STAC

Trong manifest system pair:

- `t1_datetime` = anh **som hon**
- `t2_datetime` = anh **muon hon**

Vi du AOI `ffa6dc...`:

- `system t1 = 2025-08-02T22:51:17.523550Z`
- `system t2 = 2025-08-05T11:06:37.422139Z`

### 3.2 Model convention

Trong bo du lieu train/test va trong `gee_trainlike_download.py`:

- `S1T1` = cua so `[0, +7]` quanh moc tham chieu
- `S1T2` = cua so `[-7, 0]` quanh moc tham chieu

Tuc la theo **quy uoc model**:

- `S1T1` la nhanh "sau"
- `S1T2` la nhanh "truoc"

Noi cach khac:

- `system t1/t2` la **thu tu thoi gian exact**
- `S1T1/S1T2` la **ten 2 nhanh input cua model**

Hai he ten nay khong trung nhau ve nghia.

## 4. Ba cach chon `anchor`

Ham [resolve_anchor()](/mnt/data1tb/vinh/ISSM-SAR/gee_trainlike_download.py#L48) ho tro 3 cach:

- `midpoint`
- `t1`
- `t2`

### 4.1 `anchor = midpoint`

`anchor = (system_t1 + system_t2) / 2`

Voi AOI `ffa6dc...`:

- `system_t1 = 2025-08-02T22:51:17.523550Z`
- `system_t2 = 2025-08-05T11:06:37.422139Z`
- `anchor = 2025-08-04T04:58:57.472844Z`

Neu dung default windows:

- `S1T1 = [2025-08-04T04:58:57Z, 2025-08-11T04:58:57Z]`
- `S1T2 = [2025-07-28T04:58:57Z, 2025-08-04T04:58:57Z]`

Y nghia:

- dat moc tham chieu vao giua cap exact pair
- can bang khoang cach thoi gian tu 2 ben
- giong mot gia dinh "khong biet ngay tham chieu that, nen lay diem giua"

### 4.2 `anchor = t1`

`anchor = system_t1`

Voi AOI `ffa6dc...`:

- `anchor = 2025-08-02T22:51:17.523550Z`

Default windows se thanh:

- `S1T1 = [2025-08-02T22:51:17Z, 2025-08-09T22:51:17Z]`
- `S1T2 = [2025-07-26T22:51:17Z, 2025-08-02T22:51:17Z]`

Y nghia:

- coi anh exact som hon la moc tham chieu
- cua so "truoc" ket thuc dung tai exact `t1`
- cua so "sau" bat dau tu exact `t1`

### 4.3 `anchor = t2`

`anchor = system_t2`

Voi AOI `ffa6dc...`:

- `anchor = 2025-08-05T11:06:37.422139Z`

Default windows:

- `S1T1 = [2025-08-05T11:06:37Z, 2025-08-12T11:06:37Z]`
- `S1T2 = [2025-07-29T11:06:37Z, 2025-08-05T11:06:37Z]`

Y nghia:

- coi anh exact muon hon la moc tham chieu
- cua so "truoc" ket thuc tai exact `t2`
- cua so "sau" bat dau tu exact `t2`

Trong AOI `ffa6dc...`, cach nay khong robust vi cua so sau `t2` khong co du lieu GEE
hop le, nen run bi fail.

## 5. `anchor` khac gi voi pipeline tao train/test goc

Pipeline train/test goc nam o
[`download_s1.py`](/mnt/data1tb/vinh/download_s1/download_s1.py).

Script do khong dung khai niem `anchor` mot cach tuong minh, nhung ve ban chat no
co **mot moc tham chieu that**:

- `date = feature.get('SRC_DATE')`
- `start_date = date.advance(START_DAY_SHIFT, 'day')`
- `end_date = date.advance(END_DAY_SHIFT, 'day')`

Nghia la:

- `SRC_DATE` chinh la moc tham chieu tu nhien
- cua so `[-7, 0]` hoac `[0, 7]` duoc dat quanh moc do

Vi vay:

- Neu biet duoc ngay tuong duong voi `SRC_DATE`, do moi la cach khop nhat voi
  pipeline train/test goc
- `midpoint`, `t1`, `t2` chi la **chien luoc suy ra moc tham chieu** khi ta
  khong co `SRC_DATE`

## 6. Cach nao tot nhat hien tai

### 6.1 Ve mat thuc nghiem voi AOI `ffa6dc...`

Theo phase 1 A/B:

- tot nhat hien tai: `anchor = midpoint`
- kem hon: `anchor = t1`
- khong dung duoc voi AOI nay: `anchor = t2`

Chi tiet o [ffa6dc_phase1_ab.md](/mnt/data1tb/vinh/ISSM-SAR/docs/analysis/ffa6dc_phase1_ab.md).

Ket qua chinh:

- `gee_trainlike_midpoint` cho `T1/T2 corr` cao nhat
- output texture gan `reference_good` nhat
- `anchor=t1` dua them mot scene `2025-07-27` vao cua so truoc, lam composite kem on
dinh hon
- `anchor=t2` fail vi khong co anh trong cua so sau `t2`

### 6.2 Ve mat nguyen ly

Neu chi biet system exact pair, khong co `SRC_DATE` that, thi:

- `midpoint` la heuristics can bang va hop ly nhat
- no thuong tot hon `t1` va `t2` vi khong lech ve 1 phia cua cap exact pair

Nhung can nhan manh:

- `midpoint` la **cach xap xi tot nhat hien tai**
- no **khong phai** ban sao 100% cua pipeline train/test goc

### 6.3 Cach khop nhat voi train/test goc

Cach khop nhat khong phai `midpoint`, `t1`, hay `t2`.

Cach khop nhat la:

- co mot `reference date` that, dong vai tro nhu `SRC_DATE`
- roi dat:
  - `S1T1 = [0, +7]`
  - `S1T2 = [-7, 0]`

Neu sau nay tim duoc ngay tham chieu dung nghia cua mau, ta nen uu tien no hon moi
heuristic `anchor`.

## 7. Luong xu ly hien tai cua tool

1. Doc AOI tu file `geojson`
2. Doc `system manifest` de lay:
   - `system t1/t2`
   - `pair_id`
   - `aoi_bbox`
3. Chon `anchor`
4. Tao 2 cua so thoi gian tu `anchor`
5. Query `COPERNICUS/S1_GRD` trong tung cua so
6. Loc:
   - giao AOI
   - co `VV`
   - co `VH`
   - `IW`
   - `orbit_pass` neu co chi dinh
7. Tao composite bang `median()`
8. Ap `focal_median(radius_m)`
9. `clip` theo `bbox` hoac `geometry`
10. Export `EPSG:3857`, `10m`, `2-band`
11. Rewrite band descriptions thanh `S1_VV`, `S1_VH`
12. Neu bat `--run-infer` thi goi `infer_production.py`

## 8. Nhung diem da khop voi train/test goc

- cung nguon `COPERNICUS/S1_GRD`
- cung 2 phan cuc `VV`, `VH`
- cung `instrumentMode = IW`
- cung kieu composite `median()`
- cung `focal_median(15m)` mac dinh
- cung `EPSG:3857`
- cung `scale = 10m`
- cung quy uoc model:
  - `S1T1 = [0, +7]`
  - `S1T2 = [-7, 0]`

## 9. Nhung diem chua khop hoan toan voi train/test goc

- khong co `SRC_DATE` that, nen phai suy ra `anchor`
- mac dinh `clip_mode = bbox`, trong khi script cu clip theo ROI geometry roi export theo
  `image.geometry()`
- tap scene trong cua so hien tai co the khac:
  - tron `ASCENDING` va `DESCENDING`
  - tron `S1A` va `S1C`
  - co scene trung timestamp
- so luong scene trong cua so co the rat it

Day la cac ly do vi sao `gee_trainlike` da tot hon `exact pair`, nhung van chua
hoan toan bang `reference_good`.

## 10. Khi khong co san `system_t1/system_t2`

Day la truong hop cua STAC-only production pipeline:

- dau vao chi la `AOI geojson`
- khong co san exact pair manifest

Khi do, luong dung la:

1. query STAC de lay timeline giao AOI
2. tu timeline do, de xuat `anchor` phu hop cho train-like windows
3. luu `anchor manifest`
4. neu can benchmark voi GEE, dua `anchor manifest` vao `gee_trainlike_download.py`

Tool ho tro buoc 2 la:

- [query_stac_download.py](/mnt/data1tb/vinh/ISSM-SAR/query_stac_download.py)
- command: `suggest-anchor`

Vi du:

```bash
python query_stac_download.py suggest-anchor \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --datetime 2025-07-01/2025-09-10 \
  --window-before-days 30 \
  --window-after-days 30 \
  --min-scenes-per-window 1 \
  --save-manifest runs/anchor_manifests/ffa6dc_anchor_30d.json
```

Manifest sinh ra co the dua truc tiep vao:

```bash
python gee_trainlike_download.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --manifest runs/anchor_manifests/ffa6dc_anchor_30d.json \
  --run-infer \
  --infer-config config/infer_config.yaml
```

Ghi chu quan trong:

- `suggest-anchor` khong tao exact pair cho model
- no tao **support pair + anchor_datetime + window sizes**
- `gee_trainlike_download.py` se doc cac truong nay va dung dung cua so da luu trong manifest

## 11. Danh gia 7 ngay vs 30 ngay tren AOI `ffa6dc...`

### 11.1 Midpoint 7 ngay

- `anchor = 2025-08-04T04:58:57.472844Z`
- `T1 window count = 2`
- `T2 window count = 2`
- output texture:
  - `VV = 0.338`
  - `VH = 0.307`

Run:

- [20260317T101247](/mnt/data1tb/vinh/ISSM-SAR/runs/gee_trainlike_compare/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2/20260317T101247)

### 11.2 Midpoint 30 ngay

- `anchor = 2025-08-04T04:58:57.472844Z`
- `T1 window count = 6`
- `T2 window count = 7`
- output texture:
  - `VV = 0.330`
  - `VH = 0.288`

Run:

- [20260317T115852](/mnt/data1tb/vinh/ISSM-SAR/runs/phase1_midpoint_30d/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2/20260317T115852)

### 11.3 Ket luan hien tai

Voi AOI `ffa6dc...`, `midpoint +/-30 days` tot hon `midpoint +/-7 days`.

Dieu nay cho thay:

- cua so rong hon tao composite giau hon
- speckle giam hon
- output cua model gan `reference_good` hon

Nhung can nho:

- `30 days` khong con trung khop voi pipeline train/test goc nhu `7 days`
- day la mot `best empirical setting` cho AOI nay, khong phai chan ly cho moi AOI

## 12. Config

Dung chung file [gee_compare_config.yaml](/mnt/data1tb/vinh/ISSM-SAR/config/gee_compare_config.yaml).

Tham so chinh:

- `gee.project`: GEE project
- `gee.collection`: mac dinh `COPERNICUS/S1_GRD`
- `gee.export_crs`: mac dinh `EPSG:3857`
- `gee.export_scale`: mac dinh `10`
- `gee.band_names`: mac dinh `VV`, `VH`
- `gee.output_band_descriptions`: mac dinh `S1_VV`, `S1_VH`
- `trainlike.anchor_strategy`: `midpoint`, `t1`, `t2`
- `trainlike.t1_window_days`: mac dinh `[0, 7]`
- `trainlike.t2_window_days`: mac dinh `[-7, 0]`
- `trainlike.orbit_pass`: `BOTH`, `ASCENDING`, `DESCENDING`
- `trainlike.focal_median_radius_m`: mac dinh `15`
- `trainlike.clip_mode`: `bbox` hoac `geometry`
- `trainlike.max_scene_report`: so scene toi da ghi vao report
- `output.trainlike_root_dir`: thu muc luu run

## 13. Cach chay

### 11.1 Download train-like pair

```bash
python gee_trainlike_download.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --manifest runs/pipeline/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2/20260316T160155/manifest.json
```

### 11.2 Download va infer ngay

```bash
python gee_trainlike_download.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --manifest runs/pipeline/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2/20260316T160155/manifest.json \
  --run-infer \
  --infer-config config/infer_config.yaml
```

### 11.3 Override `anchor`

```bash
python gee_trainlike_download.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --manifest runs/pipeline/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2/20260316T160155/manifest.json \
  --anchor-strategy midpoint
```

```bash
python gee_trainlike_download.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --manifest runs/pipeline/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2/20260316T160155/manifest.json \
  --anchor-strategy t1
```

## 14. Dau ra

Moi run duoc ghi vao:

- `runs/gee_trainlike_compare/<aoi_stem>/<run_id>/system_reference_manifest.json`
- `runs/gee_trainlike_compare/<aoi_stem>/<run_id>/s1t1_<pair_id>.tif`
- `runs/gee_trainlike_compare/<aoi_stem>/<run_id>/s1t2_<pair_id>.tif`
- `runs/gee_trainlike_compare/<aoi_stem>/<run_id>/infer_output/<pair_id>_SR_x2.tif`
- `runs/gee_trainlike_compare/<aoi_stem>/<run_id>/gee_trainlike_report.json`
- `runs/gee_trainlike_compare/<aoi_stem>/<run_id>/gee_trainlike_report.md`

Trong report se co:

- `system_reference`
- `anchor_strategy`
- `anchor_datetime`
- `t1_window`, `t2_window`
- danh sach scene thuc su duoc dung trong tung cua so
- thong tin export grid
- validation cho `infer_production.py`

## 15. Kien nghi hien tai

Voi model hien tai va AOI `ffa6dc...`, baseline tot nhat da kiem chung la:

- `anchor_strategy = midpoint`
- `t1_window_days = [0, 30]`
- `t2_window_days = [-30, 0]`

Neu muon bam sat pipeline train/test goc hon thi giu:

- `anchor_strategy = midpoint`
- giu:
  - `t1_window_days = [0, 7]`
  - `t2_window_days = [-7, 0]`
  - `focal_median_radius_m = 15`
  - `EPSG:3857`
  - `scale = 10m`

Tom lai:

- `7 days` = gan hon voi train/test recipe goc
- `30 days` = tot hon theo thuc nghiem tren AOI `ffa6dc...`
