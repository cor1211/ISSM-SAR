# Slide Content: SR SAR Packaging (Code-Accurate)

## Slide 1 - Problem Statement

- Can 1 luong end-to-end de tao input dung domain cho SR model
- Can phan biet ro production path va benchmark path
- Can bao toan tinh tai lap qua manifest + run summary

## Slide 2 - Two Workflows in Repository

- exact_pair:
  - query -> exact T1/T2 -> 4 file 1-band -> infer
  - dung cho debug va baseline
- stac_trainlike_composite:
  - timeline -> anchor -> pre/post windows -> composite -> infer
  - la production default

## Slide 3 - Hard Filters (Code Facts)

Ap dung o tang STAC item:

- instrument_mode = IW
- product_type = GRD
- co VV va VH
- co href doc duoc cho moi pol
- orbit/relative_orbit co the buoc match neu bat

Coverage formula:

- coverage = inter(AOI_bbox, item_bbox) / area(AOI_bbox)

## Slide 4 - Exact Pair Logic

Pair valid neu:

- min_delta_hours <= delta <= max_delta_days
- coverage_t1 >= min_aoi_coverage
- coverage_t2 >= min_aoi_coverage
- optional same_orbit_direction

Ranking pair:

1. t2 moi nhat
2. t1 moi nhat
3. delta nho hon
4. overlap lon hon

Fallback:

- strict -> balanced(30d) -> loose(90d) neu auto_relax bat

## Slide 5 - Train-Like Anchor Logic

Support pair gate:

- delta >= anchor_min_delta_hours (fallback pairing.min_delta_hours)
- delta <= window_before_days + window_after_days

Anchor candidate:

- anchor = midpoint(t1, t2)
- pre = [anchor - before, anchor)
- post = [anchor, anchor + after]
- dedupe scenes theo scene key
- phai dat min_scenes_per_window cho ca pre va post

Ranking anchor:

1. post latest input moi nhat
2. anchor moi nhat
3. support_t2 moi nhat
4. pre latest moi nhat
5. support gap nho hon

## Slide 6 - Local Composite Implementation

Per window, per polarization:

1. reproject ve 1 grid chung (EPSG:3857, 10m)
2. nanmedian(scene stack)
3. focal median (radius mac dinh 15m)

Output:

- s1t1_<id>.tif (2-band: S1_VV, S1_VH)
- s1t2_<id>.tif (2-band: S1_VV, S1_VH)

## Slide 7 - Inference Packaging

- exact_pair: run_pair_from_single_band_files
- trainlike: run_pair_from_multiband_files
- output SR: <aoi>__<id>_SR_x2.tif

Metadata packaging:

- manifest.json
- run_summary.json
- run_summary.md

## Slide 8 - Production vs Benchmark

Production:

- stac_trainlike_composite tren STAC/S3

Benchmark/Diagnostics:

- gee_compare_download.py
- gee_trainlike_download.py

Thong diep:

- GEE path giu vai tro doi chieu, khong phai luong van hanh chinh

## Slide 9 - Operational Checklist

- STAC endpoint + collection dung
- datetime range du rong
- AOI hop le
- min_scenes_per_window phu hop muc do sparse
- kiem tra run_summary sau moi run

## Slide 10 - Key Risks and Mitigation

Rui ro:

- sparse timeline -> khong du anchor candidate
- domain mismatch voi model profile
- coverage khong dat 1.0 tren AOI bbox

Giam thieu:

- mo rong datetime
- auto relax scene count
- theo doi compatibility block/warn
- su dung docs query_stac_download_module.md de debug

