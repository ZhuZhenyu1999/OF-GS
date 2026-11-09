"""Export raw relative depth and PNG previews using local Depth Anything V2."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depth_export import export_da2_images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", nargs="+", required=True, help="Original source image path(s).")
    parser.add_argument("--output_dir", required=True, help="Directory for NPY, gray/color PNGs and metadata.")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda[:index] (CUDA uses float16).")
    parser.add_argument("--checkpoint", default=None, help="Local ViT-L checkpoint; defaults to models/depth_anything_v2_vitl.pth.")
    args = parser.parse_args(argv)
    results = export_da2_images(args.image, args.output_dir, device=args.device, checkpoint=args.checkpoint)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


if __name__ == "__main__":
    main()
