# STAC-Only Pipeline Definition

## 1. Pham vi

Tai lieu nay chot lai 2 viec rieng:

1. Thu nghiem da lam:
   - mo rong `window time` tu `7 ngay` len `30 ngay`
   - danh gia bang du lieu tai tu GEE de doi chieu domain
2. Pipeline production can dung:
   - **query, chon loc, download deu phai di qua STAC**
   - khong phu thuoc GEE o khau lay du lieu dau vao production

## 2. Ket luan tu thu nghiem GEE 30 ngay

Thu nghiem `anchor=midpoint`, windows:

- `S1T2 = [-30 days, anchor]`
- `S1T1 = [anchor, +30 days]`

tren AOI `ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson` cho ket qua tot hon
ban `7 ngay`.

Ket qua chinh:

- `7 ngay`
  - `T1 count = 2`
  - `T2 count = 2`
  - input corr: `VV 0.873`, `VH 0.865`
  - output texture: `VV 0.338`, `VH 0.307`
- `30 ngay`
  - `T1 count = 6`
  - `T2 count = 7`
  - input corr: `VV 0.890`, `VH 0.880`
  - output texture: `VV 0.330`, `VH 0.288`

Y nghia:

- cua so rong hon tao composite giau hon
- speckle giam hon
- output gan `reference_good` hon

Nhung day chi la **bang chung domain**, khong co nghia pipeline production se dung GEE.

## 3. Dinh huong production da chot

Pipeline production phai la:

- dau vao: `1 AOI geojson`
- query timeline: **STAC**
- chon loc item: **STAC**
- download scenes: **STAC**
- composite / preprocess: **local**
- infer: `infer_production.py`

Noi cach khac:

- GEE chi giu vai tro thu nghiem / benchmark
- du lieu production phai di tu he thong STAC cua ban

Gia dinh thiet ke:

- khong xem sparse STAC hien tai la rao can dai han
- pipeline production duoc thiet ke cho bo STAC se duoc cap day du hon trong tuong lai
- vi vay logic selection/composite phai toi uu theo **timeline STAC day du**, khong phai theo bai toan khan item tam thoi

## 4. Tai sao exact pair STAC hien tai chua du

Exact pair STAC hien tai tot cho:

- xac dinh mốc thoi gian
- kiem tra AOI coverage
- tai raw pair de debug

Nhung no chua tot cho model hien tai vi:

- model da hoc tren domain `window composite + focal_median`
- exact pair co speckle cao hon
- exact pair co T1/T2 corr thap hon

Vi vay pipeline production moi khong nen dung `1 cap exact pair` lam input cuoi.
No nen dung `STAC timeline -> window scenes -> local composite`.

## 5. Pipeline production STAC-only da duoc implement

### 5.1 Dau vao

- `AOI geojson`
- `datetime search range` lon hon, vi du:
  - `2025-07-01/2025-09-10`
- config query:
  - `VV,VH`
  - `IW`
  - `GRD`
  - `AOI bbox coverage = 1.0`

### 5.2 Query STAC

Query STAC tat ca item giao AOI, sau do hard filter:

- `instrument_mode = IW`
- `product_type = GRD`
- co du `VV`, `VH`
- moi polarization co asset raster doc duoc
- `AOI bbox coverage = 1.0`

Day la tang selection dau vao timeline.

### 5.3 Suy ra `support pair` va `anchor`

Khi user khong co san `system_t1/system_t2`, pipeline phai tu suy ra noi bo:

1. sap xep cac item theo thoi gian
2. sinh cac `support pair`
3. lay `anchor = midpoint(t1, t2)`
4. tim `latest_input_datetime` cua tung anchor candidate
5. chon duy nhat 1 anchor sao cho du lieu dau vao moi nhat

Voi production huong train-like, `support pair` khong phai input cuoi cua model.
No chi la co so de dat `anchor`.

Phan nay da duoc code trong:

- [query_stac_download.py](/mnt/data1tb/vinh/ISSM-SAR/query_stac_download.py)
- `suggest-anchor`
- va duoc `sar_pipeline.py` goi noi bo trong mode `stac_trainlike_composite`

### 5.4 Tao 2 STAC windows

Voi huong hien tai da cho ket qua tot nhat tren AOI `ffa6dc...`, uu tien:

- `pre window = [anchor - 30 days, anchor]`
- `post window = [anchor, anchor + 30 days]`

Theo quy uoc model:

- `S1T2 = pre window`
- `S1T1 = post window`

### 5.5 Download scenes tu STAC

Thay vi download 1 pair, pipeline se download:

- tat ca scene `VV`
- tat ca scene `VH`

trong tung window, neu scene do:

- giao AOI
- full AOI bbox coverage
- dat hard filters

Cach tai:

- khong tai full item scene
- moi asset deu duoc doc remote va cat truoc theo `bbox` chua AOI
- `bbox` subset duoc snap outward theo pixel grid de bao kin AOI, tranh mat goc/canh
- vi vay `window_raw/pre` va `window_raw/post` luon la cac file nho da cat, khong phai scene lon nguyen ban

Day la diem khac biet cot loi so voi exact-pair pipeline hien tai.

Phan nay da duoc code trong:

- [sar_pipeline.py](/mnt/data1tb/vinh/ISSM-SAR/sar_pipeline.py)
- thu muc run: `window_raw/pre`, `window_raw/post`

### 5.6 Align va composite noi bo

Tat ca raster tai tu STAC can duoc:

1. warp / align ve cung grid chuan
2. grid chuan nen giu:
   - `EPSG:3857`
   - `10m`
   - extent theo `AOI bbox`
3. tach theo 2 window:
   - pre
   - post
4. composite theo tung polarization bang `median()`
5. ap `focal_median(15m)`

Sau buoc nay ta moi thu duoc:

- `s1t1_<run>.tif`
- `s1t2_<run>.tif`

theo dung domain ma model can.

Phan nay da duoc code trong:

- [sar_pipeline.py](/mnt/data1tb/vinh/ISSM-SAR/sar_pipeline.py)
- thu muc run: `composite/`

### 5.7 Infer

Sau composite:

- dua `s1t1_*.tif`, `s1t2_*.tif` vao `infer_production.py`
- output la `GeoTIFF 2-band SR`

## 6. Khi khong co san `system_t1/system_t2`, phai lam sao?

Cau tra loi da chot:

- khong yeu cau user cung cap `system_t1/system_t2`
- pipeline phai tu query STAC va sinh ra `support pair` / `anchor` noi bo

Co nghia la:

- `system_t1/system_t2` tro thanh **bien noi bo cua pipeline**
- khong con la input bat buoc tu ben ngoai

Hien tai phan sinh `anchor` da co trong:

- [query_stac_download.py](/mnt/data1tb/vinh/ISSM-SAR/query_stac_download.py)
- lenh `suggest-anchor`

Lenh nay la mot phan cua STAC-only train-like pipeline.

## 7. Lam sao de dam bao input cho model la tot nhat

Voi model hien tai, muon input tot nhat thi can uu tien:

1. khong infer truc tiep tren exact pair
2. dung window composite
3. dung `median()`
4. dung `focal_median(15m)`
5. giu `EPSG:3857`, `10m`
6. dam bao `AOI bbox coverage = 1.0` cho moi scene duoc dua vao window
7. uu tien `latest_input_datetime` moi nhat
8. chi dung `min_scenes_per_window` lam nguong hop le toi thieu

Tuc la:

- chat luong model khong phu thuoc chi vao viec chon `1 pair`
- no phu thuoc vao chat luong `window set` duoc tai ve va composite
- nhung trong production uu tien khach hang, he thong van phai tra ve SR dua tren moc du lieu moi nhat co the

## 8. Trang thai code hien tai

Can tach ro 2 viec:

- hien tai codebase da co:
  - STAC exact pair query/download
  - STAC anchor suggestion
  - GEE train-like download de benchmark
- STAC-only local composite branch da duoc implement trong:
  - [sar_pipeline.py](/mnt/data1tb/vinh/ISSM-SAR/sar_pipeline.py)
  - `workflow.mode = stac_trainlike_composite`

Dieu da co:

- logic de chon `anchor`
- logic de download STAC assets
- logic infer
- local `median()` per polarization per window
- local `focal_median(15m)`
- ghi `s1t1/s1t2`
- infer trong 1 lenh thong nhat

## 9. Luong production nen chot

Luong nen chot cho production:

1. user dua `AOI geojson`
2. query STAC mot khoang thoi gian du rong
3. STAC chon `support pair`
4. tu `support pair` suy ra `anchor = midpoint`
5. tao `[-30, 0]` va `[0, +30]`
6. download tat ca STAC scenes trong 2 windows
7. local composite + local focal median
8. tao `s1t1/s1t2`
9. infer
10. ghi `SR GeoTIFF`

## 10. Y nghia cua GEE trong toan bo cau chuyen

GEE van huu ich, nhung chi de:

- benchmark domain
- doi chieu voi train/test recipe
- xac nhan rang `30-day window composite` co the tot hon `7-day`

Con production chinh:

- van phai la STAC-only

## 11. File lien quan

- query module: [query_stac_download.py](/mnt/data1tb/vinh/ISSM-SAR/query_stac_download.py)
- query docs: [query_stac_download_module.md](/mnt/data1tb/vinh/ISSM-SAR/docs/query_stac_download_module.md)
- GEE experiment docs: [gee_trainlike_download.md](/mnt/data1tb/vinh/ISSM-SAR/docs/gee_trainlike_download.md)
- phase 1 analysis: [ffa6dc_phase1_ab.md](/mnt/data1tb/vinh/ISSM-SAR/docs/analysis/ffa6dc_phase1_ab.md)
