# `query_stac_download.py` Module Guide

## 1. Day la module goc ve spatial-selection

`query_stac_download.py` la noi goc cua logic:

- doc va canonicalize AOI
- query STAC items
- hard filter item
- compute AOI geometry coverage
- representative-month scene selection
- witness support pair
- diagnostics cho sparse inventory / representative selection

Mot so helper GEE cung tai su dung semantics tu day de giu parity.

## 2. AOI duoc xu ly nhu the nao

### 2.1 Canonical AOI

`load_geojson_aoi()` luon:

1. doc geometry that tu GeoJSON
2. repair geometry neu can
3. tinh lai bbox canonical tu geometry
4. bo qua top-level `bbox` neu bbox trong file sai

### 2.2 Vi sao can lam vay

Neu chi dua vao bbox trong file, co the gap nhung van de sau:

- bbox bi viet sai
- AOI multipolygon co nhieu mien roi nhau
- AOI geometry mong, strip, hoac co hinh dang rat khac bbox

Vi vay, geometry moi la source of truth cho decision spatial.

## 3. Spatial semantics dang dung

### 3.1 Metric chinh

Item coverage duoc tinh la:

- `aoi_coverage(item) = area(intersection(AOI_geometry, item_geometry)) / area(AOI_geometry)`

Day la hard-filter metric chinh.

### 3.2 Hard gate

- item/pair hop le khi `aoi_coverage > min_aoi_coverage`
- default hien tai: `min_aoi_coverage = 0.0`

### 3.3 Geometry fallback

Module resolve geometry cua item theo thu tu:

1. `item.geometry`
2. `item.bbox -> polygon`
3. neu ca hai fail, item xem nhu coverage = 0

Invalid geometry duoc repair theo thu tu:

1. `make_valid`
2. `buffer(0)`
3. fallback sang bbox neu co

### 3.4 Diagnostic metrics

Ngoai metric chinh, module van giu:

- `aoi_bbox_coverage_*`
- `bbox_overlap`
- `aoi_union_coverage_pair`
- `pre_union_coverage`
- `post_union_coverage`
- `combined_union_coverage`

Y nghia:

- `aoi_bbox_coverage_*`, `bbox_overlap`
  - chi de diagnostic/backward comparison
- union coverage
  - dung de ranking phu va QA
  - khong phai hard gate

## 4. Hard filter item

`collect_items_with_filters()` lam cac viec sau:

1. query STAC bang `intersects`
2. loc collection / datetime / limit
3. loc `VV/VH`
4. loc `IW`
5. loc `GRD`
6. kiem tra asset can thiet
7. annotate geometry coverage cho tung item

Ham nay tra ve:

- `items`
- `aoi_bbox` canonical
- `aoi_geometry` canonical

## 5. Legacy pair logic

Standalone exact-pair selection da duoc loai khoi module nay de giu core runtime gon va dung voi cac luong chuan hien tai.

Nhung metric sau van con ton tai vi chung van co gia tri cho:

- representative-month diagnostics
- witness support pair
- QA ve AOI geometry coverage

Bao gom:

- `aoi_bbox_coverage_*`
- `bbox_overlap`
- `aoi_union_coverage_pair`
- `pre_union_coverage`
- `post_union_coverage`
- `combined_union_coverage`

## 6. Representative-month helpers

Cac helper chinh:

- `expand_month_periods()`
- `collect_period_half_items()`
- `select_representative_scene_pools()`
- `select_witness_support_pair()`
- `build_representative_period_manifest()`

## 7. `expand_month_periods()` xu ly range nhu the nao

Input:

- `datetime_range`
- `allow_partial_periods`

Hanh vi:

- parse range huu han
- chia thanh tung `calendar month`
- neu `allow_partial_periods=false`, chi giu thang day du
- period anchor = midpoint that cua period

## 8. `collect_period_half_items()` xu ly item nhu the nao

Cho 1 period:

- `pre` la scene nam trong `[period_start, period_anchor)`
- `post` la scene nam trong `[period_anchor, period_end)`
- moi item phai co `aoi_coverage > min_aoi_coverage`

Ham nay khong tu chon signature/relaxation. No chi cat item theo period split.

## 9. `select_representative_scene_pools()` lam gi

Ham nay la trai tim cua representative-month selection.

### 9.1 Relaxation ladder

Runtime canonical hien tai khong con relaxation ladder theo orbit signature.

Selection duoc co dinh o che do:

1. `forced_mixed_orbit_all_pre_post`

### 9.2 Candidate grouping

Cac item pre/post duoc group theo `signature_mode = mixed`.

Moi signature chung giua pre va post sinh ra 1 candidate pool.

### 9.3 Candidate ranking

Thu tu rank hien tai:

1. `min(pre_scene_count, post_scene_count)` lon hon
2. `abs(pre_scene_count - post_scene_count)` nho hon
3. `min(pre_unique_datetime_count, post_unique_datetime_count)` lon hon
4. `min(pre_union_coverage, post_union_coverage)` lon hon
5. `combined_union_coverage` lon hon
6. anchor gap nho hon
7. `post_latest_scene_datetime` moi hon
8. signature token on dinh

## 10. Witness support pair

`select_witness_support_pair()` chon 1 cap scene trong pools da chot.

Thu tu ranking witness pair:

1. temporal gap nho nhat qua anchor
2. uu tien cung `relative_orbit`
3. uu tien cung `orbit_state`
4. min geometry coverage cao hon
5. ID on dinh

Witness pair chi dung cho:

- provenance
- QA
- giai thich manifest

Witness pair khong dung de sinh anchor trong representative-month.

## 11. Cac truong hop spatial can hieu ro

### 11.1 AOI mong, bbox rong

Neu AOI la mot strip heo va bbox rat rong:

- geometry coverage moi la metric chinh
- bbox coverage co the cao/thap theo cach khong con y nghia bang
- representative selection van dung geometry that

### 11.2 AOI multipolygon / disjoint

Neu AOI co nhieu mien roi nhau:

- item giao 1 phan cua AOI van co the pass neu `coverage > threshold`
- union coverage giup do muc do phu tong the cua pools/pairs
- pipeline khong ep item phai phu toan bo bbox

### 11.3 Nhieu item giao nhung giua AOI co khoang trong

Neu phan tren giao item A, phan duoi giao item B, phan giua khong giao item nao:

- tung item van co the pass neu `coverage > threshold`
- `combined_union_coverage` se phan anh phan thieu nay
- pipeline khong skip chi vi co khoang trong, vi union coverage chi la ranking/diagnostic phu

## 12. CLI debug nen dung khi can

Cac command huu ich:

- `list`
  - xem items sau hard filter
- `download`
  - tai cap asset theo item IDs da biet
- `download-href`
  - tai truc tiep mot href khi can debug ket noi hoac kiem tra asset

## 13. Ket luan operational

Neu co mau thuan giua docs va pipeline behavior, uu tien doc code trong module nay va `sar_pipeline.py`.
