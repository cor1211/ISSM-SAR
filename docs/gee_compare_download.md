# `gee_compare_download.py` Guide

## 1. Vai tro hien tai

`gee_compare_download.py` khong con la CLI benchmark `GEE exact pair` trong core runtime nua.

Hien tai file nay chi giu lai mot nhom helper dung chung cho cac luong GEE canonical:

- `init_gee()`
- `build_target_grid()`
- `build_export_params()`
- `rewrite_with_descriptions()`
- `validate_pair()`

Cac helper nay duoc tai su dung boi:

- `sar_pipeline.py`
- `gee_trainlike_download.py`
- mot so tool canonical can khoi tao GEE

## 2. Vi sao da don giam module

Luong `GEE exact pair` cu chi phuc vu benchmark/debug, khong nam trong 2 luong chuan hien tai:

- `stac + componentized_parent_mosaic`
- `gee + componentized_parent_mosaic`

De tranh drift logic va giam rui ro van hanh, phan CLI/selection exact compare da duoc loai khoi core module nay.

## 3. Nhung gi van duoc giu lai

Nhung helper con lai deu co vai tro ro rang trong runtime hien tai:

- khoi tao Earth Engine an toan
- chuan hoa export grid
- ghi lai band descriptions sau export
- validate cap `s1t1_/s1t2_` cho infer path hien tai

## 4. Luu y operational

Neu can benchmark/phan tich legacy exact-pair, nen coi do la workflow analysis rieng, khong phai mot phan cua runtime production.
