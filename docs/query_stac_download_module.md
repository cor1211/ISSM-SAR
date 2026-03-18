# query_stac_download.py Module Guide (Code-Accurate)

## 1. Muc tieu module

query_stac_download.py cung cap tang STAC low-level de:

- query item
- hard-filter item
- tim exact pairs
- rank pairs
- diagnose khi khong co pair
- de xuat train-like anchors
- tao manifest exact pair va trainlike anchor

Module nay duoc dung truc tiep boi:

- sar_pipeline.py
- gee_compare_download.py

## 2. Input, filter khong gian, thoi gian

## 2.1 Spatial filter

Module ho tro:

- bbox
- geojson intersects

Qua trinh:

1. resolve_spatial_filter lay bbox + intersects geometry
2. STAC client query theo collection, datetime, limit

## 2.2 Hard filter item

apply_hard_filters loai item neu vi pham:

- instrument_mode != IW
- product_type != GRD
- thieu polarization bat buoc (VV/VH)
- khong co href doc duoc cho polarization can dung
- neu co orbit_state filter thi phai match
- neu co relative_orbit filter thi phai match

Y nghia:

- Day la hard gate truoc khi vao bai toan pair/anchor

## 3. Cong thuc hinh hoc va metric

## 3.1 bbox_intersection

inter(b1, b2) = dien tich giao nhau cua 2 bbox.

## 3.2 bbox_area

area(b) = (maxx - minx) * (maxy - miny), co clamp >= 0.

## 3.3 bbox_overlap_ratio

overlap = inter(bbox1, bbox2) / min(area1, area2)

Neu khong giao nhau thi overlap = 0.

## 3.4 coverage_ratio

coverage = inter(reference_bbox, item_bbox) / area(reference_bbox)

Trong pipeline hien tai, reference_bbox thuong la AOI bbox.

## 4. Tim exact pairs

## 4.1 Dieu kien pair hop le trong find_pairs

Voi sorted items theo datetime, pair (i, j), i < j:

1. Neu same_orbit_direction = true:
   orbit_state(i) va orbit_state(j) phai bang nhau
2. delta_sec = t_j - t_i
3. delta_sec >= min_delta_hours * 3600
4. delta_sec <= max_delta_days * 86400
5. coverage_t1 >= min_aoi_coverage
6. coverage_t2 >= min_aoi_coverage

Luu y:

- Neu delta_sec > max_delta_sec thi break inner loop (vi list da sort theo time)
- overlap duoc ghi lai, khong phai hard filter
- min_overlap duoc truyen qua API, nhung trong find_pairs hien tai khong su dung de cat pair

## 4.2 Ranking pair

pair_rank_key:

1. t2 moi nhat
2. t1 moi nhat
3. delta_seconds nho hon
4. bbox_overlap lon hon
5. t1_id, t2_id

search_pairs_sorted = find_pairs + sorted(pair_rank_key)

## 4.3 Chan doan khong co pair

diagnose_no_pair thong ke:

- item_count
- total pair duyet
- pair fail do orbit direction
- pair fail do min time
- pair fail do max time
- pair fail do coverage

Tra ve reason chinh:

- INSUFFICIENT_ITEMS
- NO_PAIR_WITH_ORBIT_DIRECTION
- ONLY_SAME_DATATAKE_CANDIDATES
- NO_PAIR_WITH_MIN_TIME_GAP
- NO_PAIR_WITH_MAX_TIME_WINDOW
- NO_PAIR_WITH_FULL_AOI_COVERAGE
- NO_VALID_PAIR

Dong thoi tao best_relaxed (nhe dieu kien) de debug.

## 5. Anchor de xuat cho train-like

## 5.1 Y tuong

support pair -> midpoint -> anchor candidate -> danh gia pre/post windows -> rank.

## 5.2 Window definition

Voi anchor = A:

- pre: [A - window_before_days, A)
- post: [A, A + window_after_days]

collect_anchor_window_items giu item khi:

- coverage(AOI, item) >= min_aoi_coverage
- datetime nam trong pre hoac post

## 5.3 Dieu kien support pair trong suggest_trainlike_anchors

Support pair dat khi:

- delta >= min_delta_hours
- delta <= (window_before_days + window_after_days)
- coverage_t1, coverage_t2 >= min_aoi_coverage
- same_orbit_direction (neu bat)

Luu y mapping tham so:

- min_delta_hours o ham nay la gia tri caller truyen vao.
- Trong sar_pipeline train-like, gia tri do duoc lay tu trainlike.anchor_min_delta_hours,
  neu khong co thi fallback pairing.min_delta_hours.

Luu y quan trong:

- O day khong dung pairing.max_delta_days.
- Tran tren support gap duoc tinh theo tong do dai 2 windows.

## 5.4 Dedupe scene

summarize_unique_scenes dung scene key:

- (datetime, platform, orbit_state, relative_orbit, slice_number)

Muc tieu:

- tranh mot acquisition dong gop nhieu lan cho cung window.

## 5.5 Dieu kien anchor candidate hop le

Sau dedupe:

- len(pre_scenes) >= min_scenes_per_window
- len(post_scenes) >= min_scenes_per_window

## 5.6 Ranking anchor

anchor_rank_key uu tien:

1. post_latest_scene_datetime moi nhat
2. anchor_datetime moi nhat
3. support_t2_datetime moi nhat
4. pre_latest_scene_datetime moi nhat
5. support_pair_delta_seconds nho hon
6. support_t1_id, support_t2_id

## 6. Manifest formats

## 6.1 Exact pair manifest

build_manifest_for_pair tao thong tin:

- pair_id
- t1/t2 id, datetime
- delta_seconds/hours/days
- orbit, relative_orbit, slice
- bbox_overlap
- aoi_bbox_coverage_t1/t2
- assets href cho moi pol (VV, VH)

## 6.2 Trainlike anchor manifest

build_trainlike_anchor_manifest tao:

- manifest_type = trainlike_anchor
- anchor_source = stac_midpoint_pair
- anchor_datetime
- window_before_days, window_after_days
- required_polarizations
- pre/post scene lists
- latest_input_datetime = post_latest_scene_datetime
- support pair metadata (t1/t2, delta)

## 7. CLI chinh cua module

Script query_stac_download.py co cac command de:

- query + inspect items
- tim pair + report
- suggest anchor
- tao manifest
- download theo manifest

Nen dung khi can debug source STAC doc lap voi pipeline chinh.

## 8. Ghi chu thuc te

1. min_overlap hien la diagnostic metric, khong phai hard filter trong find_pairs.
2. strict_slice duoc truyen qua call stack, nhung logic hard-filter slice trong find_pairs hien tai khong duoc apply.
3. Coverage duoc tinh tren bbox, khong tinh tren polygon AOI chi tiet.
4. Pair overlap va AOI coverage la 2 metric khac nhau, khong thay the nhau.

## 9. Pseudo-code tong hop

```text
items = STAC.search(...)
items = apply_hard_filters(items)

# exact pair
pairs = find_pairs(items,
  min_delta_hours,
  max_delta_days,
  min_aoi_coverage,
  same_orbit_direction)
pairs = sort(pair_rank_key)

# trainlike anchor
candidates = []
for support_pair in item_pairs:
  if support_pair passes min_delta, max_support_gap, coverage, orbit:
    anchor = midpoint(t1, t2)
    pre, post = collect_window_items(anchor)
    pre = dedupe(pre)
    post = dedupe(post)
    if len(pre) >= min_scenes and len(post) >= min_scenes:
      candidates.append(anchor_candidate)

candidates = sort(anchor_rank_key)
```

