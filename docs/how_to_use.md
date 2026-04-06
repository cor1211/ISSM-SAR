# How To Use The Current Pipeline

Neu can mot tai lieu mo ta full end-to-end cho composite pipeline hien tai, doc truoc `docs/composite_pipeline_end_to_end.md`. File nay giu vai tro source of truth cho flow production composite.

## 1. Pipeline hien tai dang chot cai gi

Production default hien tai trong `config/pipeline_config.yaml` la:

- `workflow.mode = gee_trainlike_composite`
- `trainlike.selection_strategy = representative_calendar_period`
- `trainlike.period_mode = month`
- `trainlike.period_boundary_policy = clip_inside_period`
- `trainlike.period_split_policy = first_half_vs_second_half`
- `trainlike.clip_mode = geometry`
- `trainlike.target_crs = EPSG:3857`
- `trainlike.target_resolution = 10.0`
- `pairing.min_aoi_coverage = 0.0`
- `trainlike.componentize_seed_intersections = true`
- model semantics: `T1/S1T1 = later/post`, `T2/S1T2 = earlier/pre`

Y nghia thuc te:

- hien tai GEE la backend production de chot recipe vi inventory day hon
- STAC van duoc giu song song theo cung semantics de sau nay quay lai ma khong doi logic chon du lieu
- spatial hard filter khong dua vao bbox nua; dua vao geometry coverage that

## 2. Lenh chay co ban

### 2.1 GEE representative-month

```bash
python sar_pipeline.py \
  --geojson geojson/generated_aoi/aoi_suite_hanoi_urban_square.geojson \
  --config config/pipeline_config.yaml \
  --datetime 2025-01-01/2025-12-31 \
  --gee-project downloads1correspondings2sgdm \
  --output-dir runs/customer_jobs
```

Neu Earth Engine chua duoc auth tren may:

```bash
python sar_pipeline.py \
  --geojson geojson/generated_aoi/aoi_suite_hanoi_urban_square.geojson \
  --config config/pipeline_config.yaml \
  --datetime 2025-01-01/2025-12-31 \
  --gee-project downloads1correspondings2sgdm \
  --output-dir runs/customer_jobs \
  --authenticate
```

### 2.2 STAC representative-month

```bash
python sar_pipeline.py \
  --mode stac_trainlike_composite \
  --geojson geojson/generated_aoi/aoi_suite_hanoi_urban_square.geojson \
  --config config/pipeline_config.yaml \
  --datetime 2025-01-01/2025-12-31 \
  --output-dir runs/customer_jobs_stac
```

Luu y:

- lenh STAC tren dung cho parity/debug hoac khi inventory STAC du item
- voi inventory test hien tai, nhieu AOI se skip do thieu scene o 2 nua thang

### 2.3 Query AOI tu database va chay tuan tu

Neu da co thong tin `PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE` trong `.env`, pipeline co the doc AOI truc tiep tu `public.aois`.

Rule hien tai:

- chi lay record co `status = 'ACTIVE'`
- chi doc `id`, `geom`, `status` va metadata toi thieu can thiet
- transaction DB duoc mo theo che do `READ ONLY`
- geometry tu DB se duoc materialize thanh file GeoJSON tam roi moi di vao pipeline hien tai

Chay 1 AOI cu the theo `id`:

```bash
python sar_pipeline.py \
  --db-aoi-id 253ddb30-439d-4c33-8fa3-729e5ba73032 \
  --mode gee_trainlike_composite \
  --target-month 2025-01 \
  --output-dir runs/customer_jobs_db_single
```

Chay toan bo AOI `ACTIVE` theo thu tu truy van, co limit an toan:

```bash
python sar_pipeline.py \
  --db-all-active-aois \
  --db-limit 2 \
  --mode stac_trainlike_composite \
  --target-month 2025-03 \
  --output-dir runs/customer_jobs_db_batch
```

Output se duoc sap xep thanh:

- `jobs/<job_id>/summary.json`
- `jobs/<job_id>/job.log`
- `jobs/<job_id>/aois/<aoi_id>/periods/<period_id>/output/...`
- `jobs/<job_id>/aois/<aoi_id>/periods/<period_id>/debug/...` neu bat `--save-debug-data`

Chi tiet layout runtime da chot tai:

- `docs/runtime_storage_layout.md`

## 3. Chon AOI nao de test

Bo AOI suite khuyen nghi nam tai:

- `docs/geojson_scan/representative_aoi_suite.md`
- `docs/geojson_scan/componentized_aoi_suite.md`
- `geojson/generated_aoi/`

AOI nen uu tien:

- `geojson/generated_aoi/aoi_suite_hanoi_urban_square.geojson`
  - baseline GEE representative-month
- `geojson/generated_aoi/aoi_suite_hanoi_river_strip.geojson`
  - test geometry coverage khi AOI mong va bbox rong
- `geojson/generated_aoi/aoi_suite_hanoi_dual_patch_multipolygon.geojson`
  - test multipolygon/disjoint AOI
- `geojson/generated_aoi/aoi_suite_redriver_delta_agri_water.geojson`
  - test domain dong bang + kenh rach + nong nghiep
- `geojson/generated_aoi/aoi_suite_central_highlands_rugged.geojson`
  - test dia hinh rugged
- `geojson/generated_aoi/aoi_suite_phuquoc_coastal_waterfront.geojson`
  - test bien + waterfront
- `geojson/generated_aoi/aoi_suite_baria_coastal_mixed.geojson`
  - test coastal mixed

Neu ban can test optional child-AOI mode (`trainlike.componentize_seed_intersections = true`), uu tien bo suite rieng:

- `geojson/generated_aoi/aoi_component_hanoi_baseline_square.geojson`
- `geojson/generated_aoi/aoi_component_hanoi_river_strip.geojson`
- `geojson/generated_aoi/aoi_component_hanoi_polygon_with_hole.geojson`
- `geojson/generated_aoi/aoi_component_hanoi_bridge_dumbbell.geojson`
- `geojson/generated_aoi/aoi_component_hanoi_l_shape.geojson`
- `geojson/generated_aoi/aoi_component_hanoi_dual_patch_multipolygon.geojson`
- `geojson/generated_aoi/aoi_component_hanoi_tiny_control.geojson`
- `geojson/generated_aoi/aoi_component_hanoi_corner_split_large_parent.geojson`
- `geojson/generated_aoi/aoi_component_hanoi_nested_large_partial_small_full.geojson`
- `geojson/generated_aoi/aoi_component_hanoi_near_full_tolerance.geojson`

Tai lieu validation phu hop:

- `docs/componentized_pipeline_validation.md`
- `docs/geojson_scan/componentized_aoi_suite_smoke_false.md`
- `docs/geojson_scan/componentized_aoi_suite_smoke_true.md`
- `docs/geojson_scan/componentized_aoi_suite_contract_validation_false.md`
- `docs/geojson_scan/componentized_aoi_suite_contract_validation_true.md`
- `docs/geojson_scan/componentized_aoi_suite_mode_comparison.md`

Neu can san live `>1 child component`, dung them bo AOI boundary-focused:

- `docs/geojson_scan/componentized_boundary_search_suite.md`
- `tools/run_componentized_boundary_search.py`

Neu can mot phuong phap tot hon de tim AOI co kha nang ra live multi-child, uu tien tool data-driven:

- `tools/search_componentized_grid_aois.py`
- `docs/geojson_scan/componentized_grid_search_results.md` sau khi chay tool

Khac biet:

- `run_componentized_boundary_search.py` = thu vai AOI bien/strip co dinh
- `search_componentized_grid_aois.py` = query footprint 1 lan tren khung rong, sweep nhieu AOI ung vien, xep hang bang chinh logic componentized hien tai

Lenh mau:

```bash
python tools/search_componentized_grid_aois.py \
  --config config/pipeline_config.yaml \
  --datetime 2025-01-01/2025-12-31 \
  --gee-project downloads1correspondings2sgdm \
  --center-lon 105.896132 \
  --center-lat 21.019822 \
  --search-width-m 18000 \
  --search-height-m 18000 \
  --candidate-sizes 3000x3000,5000x5000,8000x3000,3000x8000 \
  --step-m 3000 \
  --compound-from-top 16 \
  --compound-max-pairs 48 \
  --compound-min-separation-m 4500 \
  --report-json docs/geojson_scan/componentized_grid_search_results.json \
  --report-md docs/geojson_scan/componentized_grid_search_results.md \
  --output-geojson-dir geojson/generated_aoi/grid_search_candidates
```

## 4. Workflow hien tai thuc su chay nhu the nao

Trong production default `gee_trainlike_composite`:

1. doc AOI geometry that tu GeoJSON
2. repair geometry neu can, tinh lai bbox canonical tu geometry
3. query Sentinel-1 theo AOI + `datetime`
4. chia `datetime` thanh tung `calendar month`
5. voi moi thang:
   - `anchor = midpoint that cua thang`
   - `pre/S1T2 = [period_start, anchor)`
   - `post/S1T1 = [anchor, period_end)`
6. chon scene pools bang in-period relaxation ladder
7. median composite theo `VV` va `VH`
8. ap `focal_median` neu radius > 0
9. export/align ve `EPSG:3857`, `10m`
10. infer SR voi `model(S1T1=later/post, S1T2=earlier/pre)`

Moi thang sinh toi da 1 anh SR.

Neu bat:

- `trainlike.componentize_seed_intersections = true`

thi representative-month khong con ep `1 period -> 1 output` nua. Thay vao do:

1. voi moi item `X` trong period, tao `R_X = AOI ∩ footprint(X)`
2. tim tat ca item pre/post phu `R_X`
3. neu `auto_relax_inside_period = true`, child candidate gate chi can moi ben co it nhat `1` scene de di tiep vao relaxation ladder; neu `auto_relax_inside_period = false`, no moi dung `min_scenes_per_half` ngay tai buoc candidate
4. child AOI nay se duoc composite va infer rieng
5. neu `R_Y` chua tron `R_X` va `R_Y` hop le, thi `R_X` bi suppress de tranh tao output long nhau
6. parent period khong merge child outputs; no chi gom metadata va danh sach child outputs

Luu y thuc te theo bo validation hien tai:

- live GEE suite da xac nhan child-output artifacts (`SR`, `valid mask`, `component geometry`) ton tai day du
- nhung trong bo month/AOI da chay, footprint GEE van phu tron parent AOI, nen live smoke chua sinh `>1 child output`
- cac nhanh multi-child, overlap, containment suppression hien dang duoc khoa bang unit/synthetic tests trong `docs/componentized_pipeline_validation.md`

## 5. Y nghia cac tham so CLI quan trong

### 5.1 `--geojson`

- bat buoc
- AOI geometry that se duoc doc tu file nay
- bbox top-level trong GeoJSON neu sai se khong duoc tin tuong lam source of truth

Neu khong muon truyen file tay, co the dung 1 trong 2 flag DB o ben duoi.

### 5.2 `--config`

- file config goc
- mac dinh: `config/pipeline_config.yaml`

### 5.3 `--mode`

Gia tri hop le:

- `gee_trainlike_composite`
- `stac_trainlike_composite`

Y nghia:

- `gee_trainlike_composite`
  - production default hien tai
  - chi ho tro `trainlike.selection_strategy = representative_calendar_period`
- `stac_trainlike_composite`
  - semantics giong GEE
  - chi ho tro `trainlike.selection_strategy = representative_calendar_period`
  - canonical runtime hien yeu cau `trainlike.componentize_seed_intersections = true`

### 5.4 `--datetime`

- override `stac.datetime` va `gee.datetime`
- neu bo trong `--datetime`, representative-month mode se tu dong chon **thang da ket thuc gan nhat**
  - mac dinh hien tai:
    - `trainlike.auto_datetime_strategy = previous_full_month`
    - `trainlike.auto_datetime_months_back = 1`
    - `trainlike.auto_datetime_timezone = Asia/Ho_Chi_Minh`
- co the truyen ro `--datetime auto` neu muon noi thang la dung auto mode
- neu muon backfill / chay lai thang bat ky, van co the truyen range huu han theo kieu:
  - `2025-01-01/2025-01-31`
  - `2025-01-01/2025-12-31`
- neu `allow_partial_periods = false`, pipeline chi giu cac thang day du trong range

Vi du:

- Khong truyen `--datetime`
  - neu chay vao dau thang 02 theo lich van hanh, pipeline se auto chon thang `01`
- `--datetime auto`
  - giong cach tren, nhung ro y hon khi viet script
- `2025-01-01/2025-12-31`
  - tao 12 thang day du
- `2025-01-15/2025-12-31`
  - thang 1 co the bi bo qua neu khong cho phep partial period

CLI co them 2 override phu neu can:

- `--auto-datetime-months-back`
  - vi du `2` de chay lui ve thang da ket thuc truoc nua
- `--auto-datetime-timezone`
  - doi timezone de xac dinh "thang da ket thuc gan nhat"

### 5.5 `--target-month`

- day la shortcut de backfill mot thang cu the ma khong can tu viet range
- format:
  - `YYYY-MM`
- vi du:
  - `--target-month 2025-01`
- pipeline se tu doi thanh range canonical:
  - `2025-01-01T00:00:00Z/2025-02-01T00:00:00Z`

Luu y:

- khong dung cung luc `--datetime` va `--target-month`
- `--target-month` se uu tien hon auto mode
- GEE va STAC deu dung chung rule nay
### 5.6 `--min-aoi-coverage`

- day la nguong `AOI geometry coverage`
- cong thuc:
  - `coverage = area(intersection(AOI_geometry, item_geometry)) / area(AOI_geometry)`
- hard gate hien tai la:
  - `coverage > threshold`
- default config la `0.0`

Y nghia thuc te:

- `0.0`
  - chi can item giao AOI that su
- `0.2`
  - item phai phu hon 20% dien tich AOI geometry

Luu y:

- day khong phai bbox coverage
- `aoi_bbox_coverage_*` chi la diagnostic, khong con la hard filter

### 5.7 same-orbit-direction

Khong con ho tro trong runtime canonical.

Representative-month hien tai co dinh `representative_pool_mode = mixed`, nen khong con cho phep bat che do ep cung orbit signature.

### 5.8 representative pool mode

Khong con cho phep override bang CLI hay env.

Runtime canonical hien tai luon dung:

- `trainlike.representative_pool_mode = mixed`

Nghia la pool `pre/post` duoc chon theo 1 nhom scene hop le chung trong tung nua thang, khong con giu cac bien the `auto` hay `orbit_only`.

### 5.9 `--target-crs`

- override target CRS cho composite/export
- hien khuyen nghi giu `EPSG:3857`

### 5.10 `--target-resolution`

- override pixel size theo met
- default: `10.0`
- neu GEE bi request-size limit, 2 cach xu ly thuc te la:
  - thu nho AOI
  - tang `target_resolution`

### 5.11 `--focal-median-radius-m`

- `0` -> tat focal
- `>0` -> ap median filter theo ban kinh met sau buoc composite
- default current config: `15.0`

### 5.10 `--output-dir`

- override root dir cho run outputs
- voi DB mode:
  - `--db-aoi-id` se dat outputs vao `<output-dir>/db_single_aoi/runs/...`
  - `--db-all-active-aois` se dat outputs vao `<output-dir>/db_batch_<timestamp>/runs/...`

### 5.11 `--db-aoi-id`

- truy van 1 AOI tu `public.aois` theo `id`
- chi chap nhan record co `status = 'ACTIVE'`
- neu `geom` invalid, pipeline se fail som voi message ro rang
- AOI truy van duoc se duoc materialize thanh GeoJSON tam truoc khi vao flow hien tai

### 5.12 `--db-all-active-aois`

- truy van tat ca AOI `ACTIVE`
- chay tuan tu tung AOI hop le
- AOI co `geom` invalid se bi skip va duoc ghi vao `summary.json`
- day la mode phu hop cho scheduler/noi batch

### 5.13 `--db-limit`

- limit an toan cho `--db-all-active-aois`
- rat nen dung khi smoke test hoac rollout lan dau
- vi du:
  - `--db-limit 2`

### 5.14 `--db-env-path`

- duong dan toi file `.env` chua `PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE`
- mac dinh:
  - `.env`

## 6. Cau hinh componentized child AOI

Nhung key moi duoi `trainlike`:

- `componentize_seed_intersections`
  - `false`: giu nguyen flow whole-AOI hien tai, moi period toi da 1 SR cho toan AOI
  - `true`: chuyen sang mode `child infer + parent mosaic`
    - pipeline tao candidate child region tu giao giua AOI va footprint item
    - moi child hop le duoc chon `pre/post`, composite va infer doc lap
    - output cuoi van la **1 cap VV/VH + 1 file JSON** cho parent AOI
    - pixel nao khong co child hop le se la `nodata`
- `component_parent_mosaic`
  - `true`: sau khi infer tung child, ghep cac child output ve mot canvas cha theo bbox AOI
  - hien tai day la delivery mode duoc khuyen nghi va da duoc smoke test
- `component_item_min_coverage`
  - nguong item phai phu bao nhieu phan tram cua `R_X`
  - cong thuc:
    - `area(intersection(item_geometry, R_X)) / area(R_X)`
  - mac dinh hien tai la `0.99`
  - nghia la item phai phu gan nhu toan bo `R_X` moi duoc nhan vao child do
  - step nay chi loc item cho `R_X`, khong tao them region nho hon
  - gia tri cao hon se nghiem khac hon, va co the lam vung lon fail nhung vung con seed rieng van song
- `component_min_area_ratio`
  - nguong dien tich toi thieu cua `R_X` so voi AOI cha
  - dung de bo sliver/tiny child AOI

Output representative-monthly whole-AOI hien tai:

```text
<job_dir>/
  summary.json
  job.log
  aois/
    <aoi_id>/
      periods/
        YYYY-MM/
          output/
            <item_id>.json
            <item_id>_vv.tif
            <item_id>_vh.tif
          debug/   # optional, chi co khi bat --save-debug-data
            window_raw/
              pre/
              post/
            composite/
              s1t1_period_YYYY-MM.tif
              s1t2_period_YYYY-MM.tif
```

Y nghia:

- `job.log` = file log duy nhat cua ca run
- `*_vv.tif`, `*_vh.tif` = artifact uu tien de dua len he thong; moi file la 1-band COG
- `<item_id>.json` = STAC Item-like JSON cho output SR; assets chi gom `sr_vv`, `sr_vh`; `S1T1/S1T2` nam trong properties/provenance
- `input_aoi.geojson`, `*_valid_mask.tif`, `*_SR_x2.tif` khong duoc persist trong runtime tree nua
- neu khong truyen, dung `output.root_dir` trong config

Neu bat `componentize_seed_intersections=true` va `--save-debug-data`, layout period se mo rong them debug theo child:

```text
<job_dir>/
  aois/
    <aoi_id>/
      periods/
        YYYY-MM/
          output/
            <item_id>.json
            <item_id>_vv.tif
            <item_id>_vh.tif
          debug/
            components/
              <child_id>/
                window_raw/
                  pre/
                  post/
                composite/
                  s1t1_period_YYYY-MM__<child_id>.tif
                  s1t2_period_YYYY-MM__<child_id>.tif
```

Y nghia them:

- `debug/components/<child_id>/window_raw/` = raw subset da tai cho tung child duoc chon
- `debug/components/<child_id>/composite/` = composite trung gian cua tung child
- `output/` van chi giu **parent delivery artifacts**
- khong persist child `VV/VH`, khong persist child `SR_x2`, khong persist valid-mask
- mac dinh pipeline **khong** luu `debug/`; muon giu lai thi bat `--save-debug-data`

### 5.15 `--log-level`

- ho tro: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- mac dinh: `INFO`
- thu tu uu tien:
  - `--log-level`
  - `PIPELINE_LOG_LEVEL`
  - `config.logging.level`
  - fallback `INFO`
- khi can debug `.env`, STAC query/fallback, S3 credential, S3 subset, nen dung:
  - `--log-level DEBUG`

### 5.16 `--cache-staging`

- bat buoc giu lai aligned/staged inputs o cac workflow co staging local
- huu ich cho QA, debug, visual inspection
- voi GEE representative-month, run summary van ghi `cache_staging`, nhung workflow nay chu yeu ghi `composite` va `output` theo period

### 5.17 `--gee-project`

- override `gee.project`
- bat buoc cho GEE neu config khong co project hop le

### 5.18 `--authenticate`

- goi `ee.Authenticate()` khi can
- chi can cho workflow GEE

### 5.19 `--min-delta-hours`, `--max-delta-days`

- day la nhom tham so con lai vi ly do compatibility
- canonical representative-month hien tai khong dua vao bo tham so nay de quyet dinh duong chay chinh

### 5.20 `--auto-relax`

- flag nay khong nam trong duong chay canonical hien tai
- representative-month dung `trainlike.auto_relax_inside_period`, khong dung flag nay

### 5.20 `--window-before-days`, `--window-after-days`, `--min-scenes-per-window`

- bo tham so nay khong nam trong runtime canonical hien tai
- representative-month componentized khong dung bo tham so nay de chia thang

### 5.21 `--device`

- override device cho infer, vi du `cpu` hoac `cuda`
- huu ich khi debug memory, benchmark toc do, hoac buoc phai chay tren CPU

## 6. Tham so nao dang active, tham so nao la legacy

### 6.1 Active cho representative-month hien tai

Cac tham so quan trong thuc su tac dong trong production default:

- `workflow.mode`
- `pairing.min_aoi_coverage`
- `pairing.same_orbit_direction` hoac `trainlike.same_orbit_direction`
- `trainlike.selection_strategy`
- `trainlike.period_mode`
- `trainlike.period_boundary_policy`
- `trainlike.period_split_policy`
- `trainlike.allow_partial_periods`
- `trainlike.min_scenes_per_half`
- `trainlike.auto_relax_inside_period`
- `trainlike.orbit_pass` (GEE)
- `trainlike.clip_mode` (GEE)
- `trainlike.target_crs`
- `trainlike.target_resolution`
- `trainlike.resampling` (STAC local composite)
- `trainlike.focal_median_radius_m`
- `gee.project`
- `stac.datetime` / `gee.datetime`

### 6.2 Legacy / khong phai tham so chinh cho representative-month

Cac tham so sau co the van ton tai trong config cu hoac tool analysis, nhung khong nam trong production recipe hien tai:

- `trainlike.window_before_days`
- `trainlike.window_after_days`
- `trainlike.min_scenes_per_window`
- `trainlike.auto_relax_min_scenes`
- `trainlike.anchor_pick_index`
- `trainlike.anchor_min_delta_hours`
- `pairing.auto_relax`
- `pairing.strict_slice`
- `pairing.min_overlap`

Y nghia:

- day khong phai nhom tham so chinh can uu tien chinh sua cho representative-month componentized

### 6.3 Cac nhom config khac can hieu dung

- `pairing.pols`
  - end-to-end pipeline hien tai yeu cau `VV,VH`
  - gia tri khac se bi tu choi som
- `pairing.orbit`, `pairing.rel_orbit`
  - filter/debug pre-selection cho item query
  - khong phai nhom tham so production chinh
- `stac.limit`
  - gioi han so item query tu STAC
  - tac dong den inventory duoc nhin thay trong STAC path
- `compatibility.trained_input_profile`
  - profile ma model duoc train tren do
- `compatibility.current_download_profile`
  - profile cua du lieu dang tai
- `compatibility.allow_domain_mismatch`
  - neu `false`, pipeline co the fail som de tranh chay voi domain sai
- `inference.config_path`
  - file infer config ma pipeline nap truoc khi goi inferencer
- `output.root_dir`
  - thu muc root mac dinh cua tat ca run
- `staging.cache_aligned_inputs`
  - co y nghia ro nhat o local STAC composite path
  - dung de giu input da align cho QA/debug

## 7. Relaxation ladder trong representative-month

Thu tu pipeline thu la:

1. `same_orbit_state_and_relative_orbit`
   - cung `orbit_state`
   - cung `relative_orbit`
   - moi nua thang can it nhat `max(2, min_scenes_per_half)` scene
2. `same_orbit_state_only`
   - cung `orbit_state`
   - bo rang buoc `relative_orbit`
   - moi nua thang can it nhat `max(2, min_scenes_per_half)` scene
3. `same_orbit_state_one_scene_min`
   - cung `orbit_state`
   - moi nua thang can it nhat 1 scene
4. `mixed_orbit_allowed`
   - cho phep mixed orbit
   - moi nua thang can it nhat 1 scene

Neu `same_orbit_direction = true`, level 4 se bi loai.

## 8. Cac truong hop skip/fail pho bien

Representative-month co the skip 1 period khi:

- thang khong nam tron trong range, trong khi `allow_partial_periods = false`
- khong co scene o nua dau thang
- khong co scene o nua sau thang
- co scene nhung khong level nao trong ladder hop le
- child mode khong con child nao song sau candidate gate, representative selection, va containment suppression

Pipeline co the fail ca run khi:

- AOI GeoJSON khong ton tai
- AOI geometry hong va khong repair duoc
- `datetime` khong huu han cho representative-month
- GEE project thieu / Earth Engine chua init duoc
- AOI qua lon vuot GEE download limit
- composite all-NaN
- thieu dependency infer

Can chot ro:

- `skip` = business outcome hop le, khong phai technical crash
- `fail` = loi ky thuat that su

## 9. Cach doc summary cho dung

`jobs/<job_id>/summary.json` cho biet:

- workflow dang chay
- source AOI la DB hay file
- config da resolve
- tong so AOI trong job
- status tong hop cua job
- danh sach AOI va period long ben trong

Trong `summary.json -> aois[]`, moi AOI cho biet:

- `aoi_id`
- source ref cua AOI
- final status cua AOI
- compatibility
- period counts

Trong `summary.json -> aois[].periods[]`, moi period cho biet:

- period start/end/anchor
- relaxation level nao duoc chon
- pre/post scene counts
- union coverage diagnostics
- witness support pair nao duoc chon
- output JSON/VV/VH cua period do
- thong tin child-component neu bat componentized mode

Can phan biet:

- `aoi_coverage_*`
  - metric chinh, geometry-based
- `aoi_bbox_coverage_*`
  - chi de diagnostic/backward comparison
- `witness_support_pair`
  - chi dung cho provenance/QA, khong dung de sinh anchor trong representative-month

## 10. Tai lieu nen doc tiep

- `docs/sar_pipeline.md`
- `docs/query_stac_download_module.md`
- `docs/stac_only_pipeline.md`
- `docs/t1_t2_semantics.md`
