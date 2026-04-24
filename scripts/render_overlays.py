"""Eyeball check: render each crop in a COCO set with its ellipse mask overlaid.

Reads ``<out_dir>/pits.json``, writes ``<out_dir>/overlays/<file>.png`` with
the mask drawn as a translucent red overlay plus a yellow centroid marker.
No deps beyond PIL + numpy + pycocotools.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parent.parent


def overlay(img: Image.Image, mask: np.ndarray, color=(255, 40, 40), alpha=0.45) -> Image.Image:
    base = img.convert("RGBA")
    rgba = np.array(base, dtype=np.float32)
    m = mask.astype(bool)
    rgba[m, 0] = rgba[m, 0] * (1 - alpha) + color[0] * alpha
    rgba[m, 1] = rgba[m, 1] * (1 - alpha) + color[1] * alpha
    rgba[m, 2] = rgba[m, 2] * (1 - alpha) + color[2] * alpha
    out = Image.fromarray(rgba.astype(np.uint8))

    ys, xs = np.where(m)
    if len(xs):
        cx, cy = int(xs.mean()), int(ys.mean())
        draw = ImageDraw.Draw(out)
        draw.line([(cx - 6, cy), (cx + 6, cy)], fill=(255, 255, 0, 255), width=1)
        draw.line([(cx, cy - 6), (cx, cy + 6)], fill=(255, 255, 0, 255), width=1)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--coco", type=Path, default=ROOT / "data" / "ellipse_ds" / "pits.json")
    p.add_argument("--image-dir", type=Path, default=ROOT / "data" / "ellipse_ds" / "images")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "ellipse_ds" / "overlays")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = json.loads(args.coco.read_text())
    by_img: dict[int, list] = {}
    for a in d["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    for im in d["images"]:
        path = args.image_dir / im["file_name"]
        if not path.exists():
            print("missing:", path)
            continue
        img = Image.open(path).convert("RGB")
        mask = np.zeros((im["height"], im["width"]), dtype=bool)
        for a in by_img.get(im["id"], []):
            seg = a["segmentation"]
            if isinstance(seg, dict):
                rle = {"counts": seg["counts"].encode("ascii") if isinstance(seg["counts"], str) else seg["counts"],
                       "size": seg["size"]}
                m = mask_utils.decode(rle)
            else:
                m = mask_utils.decode(mask_utils.frPyObjects(seg, im["height"], im["width"]))
            if m.ndim == 3:
                m = m.max(axis=-1)
            mask |= m.astype(bool)
        out = overlay(img, mask)
        out.save(args.out_dir / im["file_name"])
    print(f"wrote {len(d['images'])} overlays -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
