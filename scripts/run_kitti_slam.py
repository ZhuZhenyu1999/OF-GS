"""Run all KITTI sequences serially using the server dataset root by default."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_all_slam import main as run_all


def main(argv=None):
    # The shared runner supplies the server KITTI root and per-sequence outputs.
    # Explicit CLI options follow these defaults so users can override them.
    options = list(sys.argv[1:] if argv is None else argv)
    return run_all(["--dataset_type", "kitti", "--cuda_visible_devices", "0", *options])


if __name__ == "__main__":
    main()
