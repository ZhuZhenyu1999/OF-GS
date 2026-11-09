#!/usr/bin/env bash
# Run from an activated test2-slam environment on the dataset server.
set -euo pipefail

cd -- "${1:-/mnt/data_23/zzy/TEST2-SLAM}"

python scripts/calibration_preflight.py --check_local_ba --check_trajectory_stages
python -c 'import openpyxl; print("Excel dependency OK")'

# Rerun all sequences: the failed export did not save the final model/metrics.
python scripts/run_all_slam.py --profile server --dataset_type waymo \
  --cuda_visible_devices 2 --waymo_image_model opencv \
  --local_ba_every 5 --local_ba_recent_keyframes 5 \
  --local_ba_window_size 10 --local_ba_fixed_keyframes 2 \
  --viewer_mode none --output_root results/waymo_gpu2_no_final

python scripts/run_kitti_slam.py --profile server \
  --cuda_visible_devices 2 \
  --local_ba_every 5 --local_ba_recent_keyframes 5 \
  --local_ba_window_size 10 --local_ba_fixed_keyframes 2 \
  --viewer_mode none --output_root results/kitti_gpu2_no_final

python - <<'PY'
import json
from pathlib import Path
from scripts.run_all_slam import write_excel_summary

records = []
for dataset in ("waymo", "kitti"):
    path = Path(f"results/{dataset}_gpu2_no_final/slam_batch_summary.json")
    with path.open(encoding="utf-8") as stream:
        records.extend(json.load(stream))

output = write_excel_summary("results/waymo_kitti_gpu2_no_final_summary.xlsx", records)
if output is None:
    raise SystemExit("Excel generation failed: check openpyxl.")
print(f"Combined Excel: {output.resolve()}")
failed = [f"{record['dataset_type']}/{record['sequence']}"
          for record in records if record.get("returncode") != 0]
if failed:
    raise SystemExit("Failed sequences (also listed in Excel): " + ", ".join(failed))
PY
