"""Minimal tkinter tool for hand-drawing polygon masks on pit crops.

Shows each crop from the ellipse dataset with the LPA ellipse (center +
outline) overlaid as a reference. You draw a polygon; the polygon becomes
the ground-truth mask. Writes a separate COCO file so the original
ellipse dataset stays untouched.

Keys / mouse:
    left-click       add vertex
    right-click      undo last vertex
    mouse-wheel      zoom at cursor
    shift + drag     pan
    + / - / 0        zoom in / out / reset
    Enter/Return     close polygon, save, advance to next image
    A                save polygon and stay on this image (add another pit)
    Backspace        undo last vertex
    R                reset current polygon
    E                toggle reference ellipse overlay
    S                skip (mark as "no-clear-pit", no annotation saved)
    P                previous image
    Q / Esc          save progress and quit

Usage:
    python scripts/label_pits.py \\
        --coco data/ellipse_ds/pits.json \\
        --image-dir data/ellipse_ds/images \\
        --out data/ellipse_ds/pits_handlabeled.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tkinter as tk
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parent.parent


def _decode_rle(seg: dict, h: int, w: int) -> np.ndarray:
    counts = seg["counts"].encode("ascii") if isinstance(seg["counts"], str) else seg["counts"]
    rle = {"counts": counts, "size": seg["size"]}
    m = mask_utils.decode(rle)
    if m.ndim == 3:
        m = m.max(axis=-1)
    return m.astype(bool)


def _mask_centroid(mask: np.ndarray) -> Optional[tuple[int, int]]:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return int(xs.mean()), int(ys.mean())


def _polygon_mask(poly: list[tuple[int, int]], h: int, w: int) -> np.ndarray:
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
    return np.array(img, dtype=bool)


class Labeler:
    def __init__(
        self,
        coco: dict,
        image_dir: Path,
        out_path: Path,
        log_path: Path,
    ) -> None:
        self.coco_src = coco
        self.image_dir = image_dir
        self.out_path = out_path
        self.log_path = log_path

        self.images = list(coco["images"])
        self.by_img: dict[int, list] = {}
        for a in coco["annotations"]:
            self.by_img.setdefault(a["image_id"], []).append(a)

        self.out = self._load_or_init_out()
        self.done_ids: set[int] = {im["id"] for im in self.out["images"]}
        self.log_rows: list[dict] = self._load_log()

        self.idx = self._first_unlabeled()
        self.verts: list[tuple[float, float]] = []
        self.show_ref = True

        # zoom / pan state (verts are stored in image coords)
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.pan_start: Optional[tuple[int, int]] = None
        self.pan_origin: Optional[tuple[float, float]] = None

        self._build_ui()
        self._show()

    def _load_or_init_out(self) -> dict:
        if self.out_path.exists():
            return json.loads(self.out_path.read_text())
        return {
            "info": {"description": "LPA pits, hand-labeled polygon masks"},
            "licenses": [],
            "categories": self.coco_src["categories"],
            "images": [],
            "annotations": [],
        }

    def _load_log(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        with open(self.log_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _first_unlabeled(self) -> int:
        logged_files = {r["file_name"] for r in self.log_rows}
        for i, im in enumerate(self.images):
            if im["id"] in self.done_ids:
                continue
            if im["file_name"] in logged_files:
                continue
            return i
        return len(self.images)

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("luna pit labeler")
        self.status = tk.StringVar()
        bar = tk.Label(self.root, textvariable=self.status, anchor="w",
                       font=("Consolas", 10))
        bar.pack(fill="x", side="top")
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Button-3>", lambda e: self._undo())
        self.canvas.bind("<Shift-Button-1>", self._on_pan_start)
        self.canvas.bind("<Shift-B1-Motion>", self._on_pan_move)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)
        self.root.bind("<Return>", lambda e: self._finalize())
        self.root.bind("<KP_Enter>", lambda e: self._finalize())
        self.root.bind("<BackSpace>", lambda e: self._undo())
        self.root.bind("<Key-0>", lambda e: self._reset_zoom())
        self.root.bind("<Key-plus>", lambda e: self._zoom_at(None, 1.25))
        self.root.bind("<Key-equal>", lambda e: self._zoom_at(None, 1.25))
        self.root.bind("<Key-minus>", lambda e: self._zoom_at(None, 0.8))
        self.root.bind("<Key-r>", lambda e: self._reset())
        self.root.bind("<Key-R>", lambda e: self._reset())
        self.root.bind("<Key-e>", lambda e: self._toggle_ref())
        self.root.bind("<Key-E>", lambda e: self._toggle_ref())
        self.root.bind("<Key-a>", lambda e: self._finalize(stay=True))
        self.root.bind("<Key-A>", lambda e: self._finalize(stay=True))
        self.root.bind("<Key-s>", lambda e: self._skip())
        self.root.bind("<Key-S>", lambda e: self._skip())
        self.root.bind("<Key-p>", lambda e: self._prev())
        self.root.bind("<Key-P>", lambda e: self._prev())
        self.root.bind("<Key-q>", lambda e: self._quit())
        self.root.bind("<Key-Q>", lambda e: self._quit())
        self.root.bind("<Escape>", lambda e: self._quit())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    def _render_image(self, im_meta: dict) -> Image.Image:
        path = self.image_dir / im_meta["file_name"]
        base = Image.open(path).convert("RGBA")
        if not self.show_ref:
            return base
        anns = self.by_img.get(im_meta["id"], [])
        if not anns:
            return base
        h, w = im_meta["height"], im_meta["width"]
        ref = np.zeros((h, w), dtype=bool)
        for a in anns:
            seg = a["segmentation"]
            if isinstance(seg, dict):
                ref |= _decode_rle(seg, h, w)
        c = _mask_centroid(ref)
        if c is None:
            return base
        cx, cy = c
        # Minimal marker: tiny low-alpha cross at the (biased) projected center.
        # Bilinear-from-4-corners can drift ~100 px on long NAC strips; the
        # polygon the user draws is the real label, so keep this hint subtle.
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.line([(cx - 4, cy), (cx + 4, cy)], fill=(255, 230, 0, 110), width=1)
        d.line([(cx, cy - 4), (cx, cy + 4)], fill=(255, 230, 0, 110), width=1)
        return Image.alpha_composite(base, overlay)

    def _show(self) -> None:
        if self.idx >= len(self.images):
            self.status.set("all images labeled — press Q to save+quit")
            self.canvas.delete("all")
            return
        im = self.images[self.idx]
        img = self._render_image(im)
        self.current_img = img
        sw = max(1, int(round(img.width * self.scale)))
        sh = max(1, int(round(img.height * self.scale)))
        resample = Image.NEAREST if self.scale >= 1.0 else Image.BILINEAR
        view = img.resize((sw, sh), resample)
        self.tkimg = ImageTk.PhotoImage(view)
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y,
                                 anchor="nw", image=self.tkimg)
        self._redraw_poly()
        n_done = len(self.out["images"]) + sum(
            1 for r in self.log_rows if r.get("status") == "skip"
        )
        self.status.set(
            f"[{self.idx+1}/{len(self.images)}]  pit={im.get('pit_id','?')}  "
            f"product={im.get('product_id','?')}  "
            f"labeled={len(self.out['images'])}  done={n_done}  "
            f"ref={'on' if self.show_ref else 'off'}  "
            f"verts={len(self.verts)}  zoom={self.scale:.2f}x  "
            f"[L vertex | R-click/Bksp undo | wheel zoom | shift-drag pan | "
            f"0/+/- | Enter save+next | A save+add another | R clear | "
            f"E ref | S skip | P prev | Q quit]"
        )

    def _redraw_poly(self) -> None:
        self.canvas.delete("poly")
        if not self.verts:
            return
        pts = [self._image_to_canvas(ix, iy) for ix, iy in self.verts]
        for x, y in pts:
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                    outline="#ff2040", width=2, tags="poly")
        if len(pts) >= 2:
            flat = [c for xy in pts for c in xy]
            self.canvas.create_line(*flat, fill="#ff2040", width=2, tags="poly")
        if len(pts) >= 3:
            x0, y0 = pts[0]
            x1, y1 = pts[-1]
            self.canvas.create_line(x1, y1, x0, y0, fill="#ff2040",
                                    width=1, dash=(4, 3), tags="poly")

    def _canvas_to_image(self, cx: float, cy: float) -> tuple[float, float]:
        return (cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale

    def _image_to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        return self.offset_x + ix * self.scale, self.offset_y + iy * self.scale

    def _zoom_at(self, anchor, factor: float) -> None:
        new_scale = max(0.25, min(12.0, self.scale * factor))
        if new_scale == self.scale:
            return
        if anchor is None:
            ax = self.canvas.winfo_width() / 2
            ay = self.canvas.winfo_height() / 2
        else:
            ax, ay = anchor
        img_x = (ax - self.offset_x) / self.scale
        img_y = (ay - self.offset_y) / self.scale
        self.scale = new_scale
        self.offset_x = ax - img_x * self.scale
        self.offset_y = ay - img_y * self.scale
        self._show()

    def _on_wheel(self, ev) -> None:
        if getattr(ev, "num", None) == 5 or getattr(ev, "delta", 0) < 0:
            factor = 0.8
        else:
            factor = 1.25
        self._zoom_at((ev.x, ev.y), factor)

    def _reset_zoom(self) -> None:
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._show()

    def _reset_view_state(self) -> None:
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

    def _on_pan_start(self, ev) -> None:
        self.pan_start = (ev.x, ev.y)
        self.pan_origin = (self.offset_x, self.offset_y)

    def _on_pan_move(self, ev) -> None:
        if self.pan_start is None or self.pan_origin is None:
            return
        dx = ev.x - self.pan_start[0]
        dy = ev.y - self.pan_start[1]
        self.offset_x = self.pan_origin[0] + dx
        self.offset_y = self.pan_origin[1] + dy
        self._show()

    def _on_click(self, ev) -> None:
        if self.idx >= len(self.images):
            return
        ix, iy = self._canvas_to_image(ev.x, ev.y)
        self.verts.append((ix, iy))
        self._redraw_poly()
        self._update_status()

    def _undo(self) -> None:
        if self.verts:
            self.verts.pop()
            self._redraw_poly()
            self._update_status()

    def _reset(self) -> None:
        self.verts.clear()
        self._redraw_poly()
        self._update_status()

    def _toggle_ref(self) -> None:
        self.show_ref = not self.show_ref
        self._show()

    def _skip(self) -> None:
        if self.idx >= len(self.images):
            return
        im = self.images[self.idx]
        self.log_rows.append({
            "image_id": im["id"],
            "file_name": im["file_name"],
            "pit_id": im.get("pit_id", ""),
            "product_id": im.get("product_id", ""),
            "status": "skip",
            "n_vertices": 0,
        })
        self.verts.clear()
        self._reset_view_state()
        self.idx += 1
        self._show()

    def _prev(self) -> None:
        if self.idx == 0:
            return
        self.idx -= 1
        self.verts.clear()
        self._reset_view_state()
        self._show()

    def _finalize(self, stay: bool = False) -> None:
        """Save the current polygon. ``stay=True`` keeps the same image loaded
        so the user can add another pit; the default advances."""
        if self.idx >= len(self.images):
            self._quit()
            return
        if len(self.verts) < 3:
            return
        im = self.images[self.idx]
        h, w = im["height"], im["width"]
        poly_int = [(int(round(ix)), int(round(iy))) for ix, iy in self.verts]
        poly = [(float(ix), float(iy)) for ix, iy in self.verts]
        mask = _polygon_mask(poly_int, h, w)
        if not mask.any():
            return
        ys, xs = np.where(mask)
        bx0, by0 = int(xs.min()), int(ys.min())
        bw = int(xs.max() - bx0 + 1)
        bh = int(ys.max() - by0 + 1)
        flat_poly = [float(c) for xy in poly for c in xy]
        # Reuse an existing output-image record if this source was already
        # partly labeled (multi-pit case); otherwise mint a new one. This
        # keeps COCO semantics correct: one image, N annotations.
        out_im = next(
            (oi for oi in self.out["images"] if oi.get("source_image_id") == im["id"]),
            None,
        )
        if out_im is None:
            next_img_id = max((i["id"] for i in self.out["images"]), default=0) + 1
            out_im = dict(im)
            out_im["id"] = next_img_id
            out_im["source_image_id"] = im["id"]
            self.out["images"].append(out_im)
        next_ann_id = max((a["id"] for a in self.out["annotations"]), default=0) + 1
        self.out["annotations"].append({
            "id": next_ann_id,
            "image_id": out_im["id"],
            "category_id": 1,
            "bbox": [bx0, by0, bw, bh],
            "area": float(mask.sum()),
            "iscrowd": 0,
            "segmentation": [flat_poly],
            "pit_id": im.get("pit_id", ""),
        })
        self.log_rows.append({
            "image_id": im["id"],
            "file_name": im["file_name"],
            "pit_id": im.get("pit_id", ""),
            "product_id": im.get("product_id", ""),
            "status": "labeled",
            "n_vertices": len(poly),
        })
        self._save()
        self.verts.clear()
        if stay:
            self._redraw_poly()
            self._update_status()
        else:
            self._reset_view_state()
            self.idx += 1
            self._show()

    def _update_status(self) -> None:
        if self.idx >= len(self.images):
            return
        im = self.images[self.idx]
        self.status.set(
            f"[{self.idx+1}/{len(self.images)}]  pit={im.get('pit_id','?')}  "
            f"product={im.get('product_id','?')}  "
            f"labeled={len(self.out['images'])}  "
            f"ref={'on' if self.show_ref else 'off'}  "
            f"verts={len(self.verts)}  zoom={self.scale:.2f}x  "
            f"[L vertex | R-click/Bksp undo | wheel zoom | shift-drag pan | "
            f"0/+/- | Enter save+next | A save+add another | R clear | "
            f"E ref | S skip | P prev | Q quit]"
        )

    def _save(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(self.out, indent=2))
        fieldnames = ["image_id", "file_name", "pit_id", "product_id", "status", "n_vertices"]
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.log_rows)

    def _quit(self) -> None:
        self._save()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--coco", type=Path, default=ROOT / "data" / "ellipse_ds" / "pits.json")
    p.add_argument("--image-dir", type=Path, default=ROOT / "data" / "ellipse_ds" / "images")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "ellipse_ds" / "pits_handlabeled.json")
    p.add_argument("--log", type=Path, default=ROOT / "data" / "ellipse_ds" / "label_log.csv")
    args = p.parse_args()

    coco = json.loads(args.coco.read_text())
    app = Labeler(coco, args.image_dir, args.out, args.log)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
