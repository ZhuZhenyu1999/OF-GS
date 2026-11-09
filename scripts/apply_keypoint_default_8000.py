"""Update the two TEST2-SLAM keypoint defaults on an offline server checkout.

Copy this file to the server, then run:
    python apply_keypoint_default_8000.py --root /mnt/data_23/zzy/TEST2-SLAM
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile


REPLACEMENTS = {
    "args.py": (
        "parser.add_argument('--num_kpts', type=int, default=7000,",
        "parser.add_argument('--num_kpts', type=int, default=8000,",
    ),
    "scripts/run_all_slam.py": (
        'parser.add_argument("--num_kpts", type=int, default=7000)',
        'parser.add_argument("--num_kpts", type=int, default=8000)',
    ),
}


def update(root: Path, check_only: bool = False) -> None:
    root = root.expanduser().resolve(strict=True)
    if not (root / "slam.py").is_file():
        raise ValueError(f"Not a TEST2-SLAM checkout: {root}")

    changes = {}
    for name, (before, after) in REPLACEMENTS.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or linked source file: {path}")
        source = path.read_text(encoding="utf-8")
        if source.count(after) == 1 and before not in source:
            print(f"OK {name}: already 8000")
            continue
        if source.count(before) != 1 or after in source:
            raise ValueError(f"Unexpected --num_kpts definition in {path}; no files changed")
        changes[name] = source.replace(before, after, 1)

    if check_only:
        for name in changes:
            print(f"PENDING {name}: 7000 -> 8000")
        return
    if not changes:
        print("Both keypoint defaults are already 8000")
        return

    backup = root / ".keypoint-backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup.mkdir(parents=True, exist_ok=False)
    for name in changes:
        saved = backup / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, saved)
    print(f"Backup: {backup}")

    written = []
    try:
        for name, source in changes.items():
            target = root / name
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=target.parent,
                                             prefix=".keypoint-", delete=False) as handle:
                temp_path = Path(handle.name)
                handle.write(source)
            shutil.copymode(target, temp_path)
            temp_path.replace(target)
            written.append(name)
        for name, (_, after) in REPLACEMENTS.items():
            if (root / name).read_text(encoding="utf-8").count(after) != 1:
                raise RuntimeError(f"Post-update check failed: {name}")
    except Exception:
        for name in written:
            shutil.copy2(backup / name, root / name)
        raise
    print("UPDATE OK: args.py and scripts/run_all_slam.py default to 8000")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    options = parser.parse_args()
    update(options.root, options.check_only)


if __name__ == "__main__":
    main()
