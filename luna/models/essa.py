from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import nms

from luna.config import LROC_VALID_MIN, WEIGHTS_DIR
from luna.models.maskrcnn import build_essa_model

log = logging.getLogger(__name__)

ESSA_TILE = 2048
ESSA_CLASSES = {1: "skylight", 2: "pit"}
DEFAULT_WEIGHTS = WEIGHTS_DIR / "essa.pt"


@dataclass(frozen=True)
class RefinedHit:
    rank: int
    product_id: str
    votes: int
    dino_score: float
    essa_score: float
    essa_class: str
    lon: float
    lat: float
    x_offset: int
    y_offset: int


class ESSARefiner:
    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self._model = model.eval()
        self._device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_WEIGHTS,
        device: str | None = None,
    ) -> ESSARefiner:
        if device is None:
            device = (
                "mps" if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available() else
                "cpu"
            )
        dev = torch.device(device)
        log.info("Loading ESSA from %s on %s ...", checkpoint, dev)
        model = build_essa_model(checkpoint=checkpoint, map_location=str(dev))
        return cls(model=model.to(dev), device=dev)

    @staticmethod
    def _extract_crop(
        img_path: Path,
        label_offset: int,
        img_width: int,
        img_height: int,
        cx: int,
        cy: int,
        crop_size: int = ESSA_TILE,
    ) -> np.ndarray:
        half = crop_size // 2
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(img_width, x0 + crop_size)
        y1 = min(img_height, y0 + crop_size)

        raw = np.memmap(str(img_path), dtype=np.int16, mode="r",
                         offset=label_offset, shape=(img_height, img_width))
        crop = raw[y0:y1, x0:x1].copy().astype(np.float32)
        del raw

        valid = crop[crop > LROC_VALID_MIN]
        lo, hi = (valid.min(), valid.max()) if valid.size > 0 else (0.0, 1.0)
        crop_norm = np.clip((crop - lo) / (hi - lo + 1e-6), 0, 1)

        if crop_norm.shape != (crop_size, crop_size):
            padded = np.zeros((crop_size, crop_size), dtype=np.float32)
            padded[:crop_norm.shape[0], :crop_norm.shape[1]] = crop_norm
            crop_norm = padded

        return crop_norm

    def _score_crop(
        self,
        crop: np.ndarray,
        score_thr: float = 0.5,
        iou_thr: float = 0.5,
    ) -> tuple[float, str, np.ndarray | None]:
        """Return (best_score, class_name, bounding_box)."""
        t = torch.from_numpy(crop[None, ...]).to(self._device)

        with torch.no_grad():
            out = self._model([t])[0]

        out = {k: v.detach().cpu() for k, v in out.items()}

        keep_mask = out["scores"] >= score_thr
        if not keep_mask.any():
            return 0.0, "none", None

        out = {k: v[keep_mask] for k, v in out.items()}

        if out["boxes"].numel() > 0:
            keep = nms(out["boxes"], out["scores"], iou_thr)
            out = {k: v[keep] for k, v in out.items()}

        for cls_id in (2, 1):
            cls_mask = out["labels"] == cls_id
            if cls_mask.any():
                idx = out["scores"][cls_mask].argmax()
                best_score = float(out["scores"][cls_mask][idx])
                best_box = out["boxes"][cls_mask][idx].numpy()
                return best_score, ESSA_CLASSES[cls_id], best_box

        return 0.0, "none", None

    def refine(
        self,
        hits: list,
        nac_img_paths: dict[str, Path],
        nac_offsets: dict[str, int],
        nac_dims: dict[str, tuple[int, int]],
        score_thr: float = 0.5,
        essa_min_score: float = 0.0,
        output_dir: str | Path | None = None,
    ) -> list[RefinedHit]:
        candidates = []

        for hit in hits:
            pid = hit.product_id
            if pid not in nac_img_paths:
                log.warning("No NAC path registered for %s — skipping ESSA.", pid)
                continue

            cx = hit.x_offset + 128
            cy = hit.y_offset + 128
            w, h = nac_dims[pid]

            crop = self._extract_crop(
                img_path=nac_img_paths[pid],
                label_offset=nac_offsets[pid],
                img_width=w,
                img_height=h,
                cx=cx, cy=cy,
            )

            essa_score, essa_class, box = self._score_crop(crop, score_thr=score_thr)
            if essa_score < essa_min_score or essa_class == "none":
                continue

            candidates.append({
                "hit": hit,
                "crop": crop,
                "box": box,
                "essa_score": essa_score,
                "essa_class": essa_class
            })

        candidates.sort(key=lambda x: (-x["essa_score"], -x["hit"].votes))

        refined: list[RefinedHit] = []
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for rank, item in enumerate(candidates, start=1):
            h = item["hit"]
            pid = h.product_id
            
            refined_hit = RefinedHit(
                rank=rank, product_id=pid, votes=h.votes, dino_score=h.score,
                essa_score=item["essa_score"], essa_class=item["essa_class"],
                lon=h.lon, lat=h.lat, x_offset=h.x_offset, y_offset=h.y_offset,
            )
            refined.append(refined_hit)

            if output_dir:
                self._save_debug_plot(
                    output_dir, rank, pid, item["crop"], 
                    item["essa_class"], item["essa_score"], item["box"]
                )

        log.info("ESSA refined %d → %d hits.", len(hits), len(refined))
        return refined

    @staticmethod
    def _save_debug_plot(
        out_dir: Path,
        rank: int,
        pid: str,
        crop: np.ndarray,
        cls_name: str,
        score: float,
        box: np.ndarray | None,
    ) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(crop, cmap="gray", origin="upper")
        
        if box is not None:
            x0, y0, x1, y1 = box
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(x0, max(0, y0 - 10), f"{cls_name}: {score:.4f}", color="red", fontsize=12, weight="bold")

        ax.set_title(f"Rank {rank} | {pid} (Center x: 1024, y: 1024)")
        ax.axis("off")
        
        plt.savefig(out_dir / f"rank_{rank:02d}_{pid}_{cls_name}.svg", bbox_inches="tight", dpi=150)
        plt.close(fig)