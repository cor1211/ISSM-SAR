Cách chạy chuẩn hiện tại là dùng [sar_pipeline.py](/mnt/data1tb/vinh/ISSM-SAR/sar_pipeline.py).  
Mode mặc định đã là `stac_trainlike_composite`, tức:

- đọc `AOI geojson`
- query timeline trên STAC
- chọn `1 anchor` theo `latest_input_datetime` mới nhất
- tải các scene STAC đã **cắt theo bbox chứa AOI**
- composite local
- infer
- trả ra `1 GeoTIFF SR`

**1. Chuẩn bị**
Cần:
- env `issm`
- STAC/S3 truy cập được
- weights/model config đúng
- `.env` có S3 credentials nếu hệ thống STAC asset cần auth

File config chính:
- pipeline: [pipeline_config.yaml](/mnt/data1tb/vinh/ISSM-SAR/config/pipeline_config.yaml)
- infer: [infer_config.yaml](/mnt/data1tb/vinh/ISSM-SAR/config/infer_config.yaml)

Lưu ý quan trọng:
- khi chạy `sar_pipeline.py`, `infer_config.yaml` chủ yếu dùng cho:
  - `device`
  - checkpoints
  - normalization
  - patch/batch/AMP
- `input.input_dir` trong `infer_config.yaml` **không phải** đầu vào của pipeline này

**2. Lệnh chạy production khuyến nghị**
Ví dụ với AOI của bạn:

```bash
source /mnt/data1tb/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/data1tb/conda_envs/issm

python sar_pipeline.py \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --config config/pipeline_config.yaml \
  --datetime 2025-07-01/2025-09-10 \
  --output-dir runs/customer_jobs \
  --cache-staging
```

Đây là lệnh nên dùng trước tiên.

**3. Các tham số chính của `sar_pipeline.py`**
CLI hiện tại ở [sar_pipeline.py](/mnt/data1tb/vinh/ISSM-SAR/sar_pipeline.py).

Bắt buộc:
- `--geojson`
  - AOI đầu vào

Rất nên truyền:
- `--datetime`
  - khoảng thời gian query STAC
  - ví dụ: `2025-07-01/2025-09-10`
  - vì pipeline đang `newest-first`, bạn nên để khoảng đủ rộng để hệ thống tự chọn mốc mới nhất tốt nhất

Tùy chọn workflow:
- `--config`
  - file config pipeline
- `--mode`
  - `stac_trainlike_composite`: production khuyến nghị
  - `exact_pair`: chỉ để debug/so sánh

Tùy chọn pair/filter:
- `--min-delta-hours`
  - delta tối thiểu giữa support pair
  - mặc định: `24`
- `--max-delta-days`
  - delta tối đa giữa support pair
  - mặc định: `10`
- `--min-aoi-coverage`
  - coverage tối thiểu của `AOI bbox` trên mỗi item
  - mặc định: `1.0`
- `--same-orbit-direction`
  - nếu bật, support pair/anchor chỉ dùng các item cùng `ascending/descending`
- `--auto-relax`
  - chỉ hữu ích hơn trong mode `exact_pair`; với production train-like thường không cần bật

Tùy chọn train-like composite:
- `--window-before-days`
  - độ dài window trước anchor
  - mặc định: `30`
- `--window-after-days`
  - độ dài window sau anchor
  - mặc định: `30`
- `--min-scenes-per-window`
  - số scene unique tối thiểu mỗi window
  - mặc định hiện tại: `1`
- `--target-crs`
  - CRS chuẩn để align/composite
  - mặc định: `EPSG:3857`
- `--target-resolution`
  - pixel size target
  - mặc định: `10`
- `--focal-median-radius-m`
  - radius của bước làm mượt sau composite
  - mặc định: `15`

Tùy chọn chạy/infer:
- `--device`
  - `cuda`, `cuda:0`, `cpu`
- `--output-dir`
  - thư mục root của run
- `--cache-staging`
  - lưu thêm input aligned/composite để kiểm tra

**4. Giá trị mặc định hiện tại**
Trong [pipeline_config.yaml](/mnt/data1tb/vinh/ISSM-SAR/config/pipeline_config.yaml):

- `workflow.mode = stac_trainlike_composite`
- `pairing.pols = VV,VH`
- `pairing.min_aoi_coverage = 1.0`
- `pairing.min_delta_hours = 24`
- `pairing.max_delta_days = 10`
- `pairing.same_orbit_direction = false`
- `trainlike.window_before_days = 30`
- `trainlike.window_after_days = 30`
- `trainlike.min_scenes_per_window = 1`
- `trainlike.target_crs = EPSG:3857`
- `trainlike.target_resolution = 10`
- `trainlike.focal_median_radius_m = 15`

**5. Lệnh production nên dùng theo từng tình huống**

Chạy mặc định:
```bash
python sar_pipeline.py \
  --geojson geojson/your_aoi.geojson \
  --datetime 2025-01-01/2025-12-31
```

Chạy production, ép cùng hướng orbit:
```bash
python sar_pipeline.py \
  --geojson geojson/your_aoi.geojson \
  --datetime 2025-01-01/2025-12-31 \
  --same-orbit-direction
```

Chạy production với window khác:
```bash
python sar_pipeline.py \
  --geojson geojson/your_aoi.geojson \
  --datetime 2025-01-01/2025-12-31 \
  --window-before-days 30 \
  --window-after-days 30 \
  --min-scenes-per-window 1
```

Chạy debug bằng exact pair:
```bash
python sar_pipeline.py \
  --geojson geojson/your_aoi.geojson \
  --mode exact_pair \
  --datetime 2025-01-01/2025-12-31
```

**6. Cách pipeline hiện chọn dữ liệu**
Cho production `stac_trainlike_composite`:

1. query STAC theo AOI + `datetime`
2. hard filter theo `VV,VH`, `IW`, `GRD`, `AOI bbox coverage`
3. sinh support pair
4. từ mỗi support pair suy ra `anchor = midpoint`
5. xếp hạng anchor theo:
   - `latest_input_datetime` mới nhất
   - rồi mới đến các tie-break khác
6. chọn đúng `1 anchor`
7. lấy:
   - `pre window = [anchor - before_days, anchor]`
   - `post window = [anchor, anchor + after_days]`
8. tải các scene STAC trong 2 window dưới dạng **subset bbox chứa AOI**
9. composite + focal median
10. infer

**7. Dữ liệu tải về có phải full item không?**
Không.  
Pipeline hiện tại tải **subset bbox AOI**, không tải full scene.

Logic này ở [query_stac_download.py:699](/mnt/data1tb/vinh/ISSM-SAR/query_stac_download.py:699).

**8. Output nằm ở đâu**
Sau mỗi run, thư mục có dạng:

- `runs/.../<aoi_stem>/<run_id>/manifest.json`
- `runs/.../<aoi_stem>/<run_id>/run_summary.json`
- `runs/.../<aoi_stem>/<run_id>/run_summary.md`
- `runs/.../<aoi_stem>/<run_id>/window_raw/pre/*.tif`
- `runs/.../<aoi_stem>/<run_id>/window_raw/post/*.tif`
- `runs/.../<aoi_stem>/<run_id>/composite/s1t1_*.tif`
- `runs/.../<aoi_stem>/<run_id>/composite/s1t2_*.tif`
- `runs/.../<aoi_stem>/<run_id>/output/*_SR_x2.tif`

Ví dụ run hiện tại:
- [run_summary.md](/mnt/data1tb/vinh/ISSM-SAR/runs/customer_jobs/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2/20260317T152309/run_summary.md)

**9. Lệnh debug rất hữu ích trước khi chạy full pipeline**

Xem support pairs:
```bash
python query_stac_download.py pair \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --datetime 2025-07-01/2025-09-10 \
  --top-k 10
```

Xem anchor candidates:
```bash
python query_stac_download.py suggest-anchor \
  --geojson geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson \
  --datetime 2025-07-01/2025-09-10 \
  --window-before-days 30 \
  --window-after-days 30 \
  --min-scenes-per-window 1 \
  --top-k 10
```

**10. Khuyến nghị thực tế**
Nếu mục tiêu là “AOI khách hàng vào, trả SR từ dữ liệu mới nhất có thể”, mình khuyên dùng mặc định này:

```bash
python sar_pipeline.py \
  --geojson geojson/<customer_aoi>.geojson \
  --config config/pipeline_config.yaml \
  --datetime 2025-01-01/2025-12-31 \
  --output-dir runs/customer_jobs \
  --cache-staging
```

với:
- `mode = stac_trainlike_composite`
- `window_before_days = 30`
- `window_after_days = 30`
- `min_scenes_per_window = 1`
- `same_orbit_direction = false`

**11. Khi nào nên đổi tham số**
- Muốn dữ liệu “mới nhất tuyệt đối”: giữ `same_orbit_direction = false`
- Muốn đồng nhất hình học hơn: bật `--same-orbit-direction`
- STAC rất dày và bạn muốn tránh window quá nghèo: tăng `--min-scenes-per-window`
- Muốn support pair không quá xa nhau: giảm `--max-delta-days`
- Muốn broaden candidate mới nhất: tăng rộng `--datetime`

Nếu bạn muốn, mình có thể làm tiếp một phần rất hữu ích:
1. viết sẵn 3 command template cho `production`, `strict newest`, và `same-orbit`
2. hoặc thêm một `docs/how_to_run_pipeline.md` riêng, gọn kiểu runbook để bạn dùng trực tiếp cho vận hành.
