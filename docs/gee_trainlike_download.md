# `gee_trainlike_download.py` Guide

## 1. Vai tro hien tai

`gee_trainlike_download.py` khong con giu CLI benchmark/debug pair-anchored trong core runtime nua.

Hien tai module nay chi giu lai cac helper phuc vu luong GEE representative-composite chuan:

- `clip_geometry()`
- `build_collection()`
- `collection_summary()`
- `collection_scene_items()`
- `build_trainlike_image()`
- `download_gee_image()`

## 2. Cac helper nay duoc dung o dau

Nhung ham tren duoc tai su dung boi:

- `sar_pipeline.py` cho `gee + whole_aoi`
- `sar_pipeline.py` cho `gee + componentized_parent_mosaic`
- mot so tool canonical can phan tich inventory GEE

## 3. Pham vi helper duoc giu lai

Module van giu lai dung cac thao tac can cho composite runtime hien tai:

- tao image collection theo AOI + thoi gian + orbit pass
- chuyen collection GEE thanh scene items de dung chung voi logic selection
- median composite `VV/VH`
- clip geometry theo `bbox` hoac `geometry`
- download GeoTIFF tu GEE

## 4. Luu y operational

Neu can benchmark hoac tai tao workflow pair-anchored cu, nen xem do la workflow analysis/legacy rieng, khong phai mot phan cua runtime production.
