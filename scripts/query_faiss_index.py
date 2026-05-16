import logging
import os
import pickle
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Callable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import pvl

from luna.io.pds_index import PDSIndex
from luna.io.projection import LinearProjection, pixel_to_lonlat
from luna.models.dinov3 import DINOEncoder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("luna.consensus_geo_search")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NAC_PRODUCT_IDS: list[str] = [
    "M1118880788RC",
    "M1210724899LC",
    # "M102285549RE",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
SCRATCH_DIR  = DATA_DIR / "_scratch"
PITS_DIR     = SCRATCH_DIR / "pits"
INDEX_DIR    = SCRATCH_DIR / "indices"

SEARCH_K     = 1000
FINAL_TOP_K  = 100
ZOOM_SIZE    = 256
MIN_DIST_PX  = 512.0
OUTPUT_PLOT  = PROJECT_ROOT / "temp" / "consensus_geo_multi.svg"


# ---------------------------------------------------------------------------
# Per-NAC context
# ---------------------------------------------------------------------------

@dataclass
class NACContext:
    product_id: str
    img_path: Path
    projection: LinearProjection
    img_width: int
    img_height: int
    label_offset: int
    index: object   # faiss index, loaded lazily
    metadata: list

@dataclass
class _SpiceProjection:
    """Duck-type wrapper so a SPICE coord function works wherever LinearProjection is expected."""
    _fn: Callable[[float, float], tuple[float, float]]

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        return self._fn(x, y)

def _project(projection, cx: float, cy: float) -> tuple[float, float]:
    if isinstance(projection, _SpiceProjection):
        return projection(cx, cy)
    return pixel_to_lonlat(projection, cx, cy)

def _read_img_geometry(img_path: Path) -> tuple[int, int, int]:
    """Return (lines, samples, label_byte_offset) from a PDS3 label."""
    with open(img_path, "rb") as f:
        label = pvl.load(f)
    img_block    = label["IMAGE"]
    lines        = int(img_block["LINES"])
    samples      = int(img_block["LINE_SAMPLES"])
    label_offset = int(label["RECORD_BYTES"]) * int(label.get("LABEL_RECORDS", 1))
    return lines, samples, label_offset


def _build_nac_context(product_id: str, pds: PDSIndex) -> NACContext | None:
    import faiss
    img_path   = SCRATCH_DIR / f"{product_id}.IMG"
    index_path = INDEX_DIR   / f"faiss_{product_id}.index"
    meta_path  = INDEX_DIR   / f"faiss_{product_id}_meta.pkl"

    for p in (img_path, index_path, meta_path):
        if not p.exists():
            log.warning("Missing file for %s: %s", product_id, p.name)
            return None

    lines, samples, label_offset = _read_img_geometry(img_path)

    raw_geo  = pds.geometry_for(product_id)
    geometry = {str(k).lower(): (v.value if hasattr(v, "value") else v) for k, v in raw_geo.items()}

    try:
        projection = LinearProjection.from_nac_geometry(geometry, samples=samples, lines=lines)
        log.info("Bilinear projection loaded for %s", product_id)
    except (ValueError, KeyError, TypeError) as e:
        log.warning("Bilinear projection failed (%s), falling back to SPICE for %s", e, product_id)
        try:
            from luna.io.spice_project import ensure_kernels_for_label
            from luna.screening.candidate_gen import DataIngestor
            ensure_kernels_for_label(img_path)
            spice_fn = DataIngestor._build_spice_coord_fn(img_path)
            # Wrap SPICE fn in a LinearProjection-compatible duck-type
            projection = _SpiceProjection(spice_fn)
        except Exception as spice_e:
            log.error("SPICE fallback also failed for %s: %s", product_id, spice_e)
            return None

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return NACContext(
        product_id   = product_id,
        img_path     = img_path,
        projection   = projection,
        img_width    = samples,
        img_height   = lines,
        label_offset = label_offset,
        index        = faiss.read_index(str(index_path)),
        metadata     = metadata,
    )


# ---------------------------------------------------------------------------
# Multi-index search + aggregation
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    nac: NACContext
    meta_idx: int
    votes: int
    best_dist: float


def search_all(contexts, query_vecs, k):
    import faiss

    vote_map: dict[tuple[str, int], int] = Counter()
    best_dist_map: dict[tuple[str, int], float] = {}
    context_map: dict[tuple[str, int], NACContext] = {}

    for ctx in contexts:
        q = np.ascontiguousarray(query_vecs, dtype=np.float32)
        faiss.normalize_L2(q)
        dists, ids = ctx.index.search(q, k)

        for row in range(ids.shape[0]):
            for dist, idx in zip(dists[row], ids[row]):
                if idx < 0:
                    continue
                
                key = (ctx.product_id, int(idx))
                vote_map[key] += 1
                context_map[key] = ctx
                
                if key not in best_dist_map or dist > best_dist_map[key]:
                    best_dist_map[key] = float(dist)

    ranked = sorted(
        best_dist_map.keys(),
        key=lambda k: best_dist_map[k],
        reverse=True
    )

    hits: list[Hit] = []
    accepted: list[tuple[float, float]] = []

    for key in ranked:
        ctx = context_map[key]
        meta_idx = key[1]
        meta = ctx.metadata[meta_idx]
        cx = meta.x_offset + meta.width / 2.0
        cy = meta.y_offset + meta.height / 2.0

        if any(
            np.sqrt((cx - ax) ** 2 + (cy - ay) ** 2) < MIN_DIST_PX
            for ax, ay in accepted
            if context_map.get(key, ctx).product_id == ctx.product_id
        ):
            continue

        accepted.append((cx, cy))
        hits.append(Hit(
            nac=ctx,
            meta_idx=meta_idx,
            votes=vote_map[key],
            best_dist=best_dist_map[key]
        ))
        
        if len(hits) == FINAL_TOP_K:
            break

    return hits


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_hits(
    hits: list[Hit],
    anchor_img: np.ndarray | None,
    output_path: Path,
) -> None:
    n_cols  = 10
    n_rows  = (FINAL_TOP_K + 1 + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(32, 3.2 * n_rows))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    axes[0].imshow(anchor_img if anchor_img is not None else np.zeros((256, 256)), cmap="gray")
    axes[0].set_title("ANCHOR", fontweight="bold", color="steelblue", fontsize=8)

    log.info("Rank | NAC            | Votes | Dist  | X_px   | Y_px   | Latitude  | Longitude")
    log.info("─" * 85)

    for i, hit in enumerate(hits):
        if i + 1 >= len(axes):
            break

        meta = hit.nac.metadata[hit.meta_idx]
        cx   = int(meta.x_offset + meta.width  // 2)
        cy   = int(meta.y_offset + meta.height // 2)
        lat  = meta.lat
        lon  = meta.lon

        log.info("%03d  | %-14s | %3d   | %.3f | %6d | %6d | %9.5f | %9.5f",
                 i + 1, hit.nac.product_id, hit.votes, hit.best_dist, cx, cy, lat, lon)

        w, h = hit.nac.img_width, hit.nac.img_height
        x0   = max(0, cx - ZOOM_SIZE // 2)
        y0   = max(0, cy - ZOOM_SIZE // 2)
        x1   = min(w, x0 + ZOOM_SIZE)
        y1   = min(h, y0 + ZOOM_SIZE)

        raw  = np.memmap(str(hit.nac.img_path), dtype=np.int16, mode="r",
                         offset=hit.nac.label_offset, shape=(h, w))
        crop = raw[y0:y1, x0:x1].copy().astype(np.float32)
        del raw

        valid         = crop[crop > -32752]
        c_min, c_max  = (valid.min(), valid.max()) if valid.size > 0 else (0, 1)
        crop_norm     = np.clip((crop - c_min) / (c_max - c_min + 1e-6), 0, 1)

        axes[i + 1].imshow(crop_norm, cmap="gray")
        axes[i + 1].set_title(
            f"R:{i+1} V:{hit.votes} [{hit.nac.product_id[-4:]}]\n"
            f"X:{cx} Y:{cy}\n"
            f"{lat:.4f}N {lon:.4f}E",
            fontsize=6.5,
        )


    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, format="svg")
    log.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import faiss
    faiss.omp_set_num_threads(1)

    pds      = PDSIndex()
    contexts = [c for pid in NAC_PRODUCT_IDS if (c := _build_nac_context(pid, pds)) is not None]

    if not contexts:
        log.error("No valid NAC contexts. Check index/meta/img files.")
        return

    log.info("Loaded %d / %d NAC contexts.", len(contexts), len(NAC_PRODUCT_IDS))

    encoder    = DINOEncoder(lora_dir="F1nnSBK/lunar-dinov3-lora",
                             base_weights_path="F1nnSBK/lunar-dinov3-lora", device="mps")
    pit_paths  = list(PITS_DIR.glob("*.npy"))
    log.info("Encoding %d pit anchors ...", len(pit_paths))

    all_q_vecs: list[np.ndarray] = []
    anchor_img: np.ndarray | None = None

    for p in pit_paths:
        arr   = np.load(p).astype(np.float32)
        valid = arr[arr > -32752]
        f_min, f_max = (valid.min(), valid.max()) if valid.size > 0 else (0, 1)
        norm  = np.clip((arr - f_min) / (f_max - f_min + 1e-6), 0, 1)
        if "Aristarchus_6" in p.name:
            anchor_img = norm
        all_q_vecs.append(encoder.encode((np.expand_dims(norm, 0) * 255).astype(np.uint8)))

    query_vecs = np.vstack(all_q_vecs).astype(np.float32)
    log.info("Query vec sample: %s", query_vecs[0, :8])
    log.info("Query vec norm: %.4f", np.linalg.norm(query_vecs[0]))


    log.info("Searching %d indices with k=%d ...", len(contexts), SEARCH_K)
    hits = search_all(contexts, query_vecs, k=SEARCH_K)

    plot_hits(hits, anchor_img, OUTPUT_PLOT)


if __name__ == "__main__":
    main()