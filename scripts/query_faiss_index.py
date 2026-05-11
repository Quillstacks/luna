import logging
import os
import pickle

# Must be set before any OpenMP-linked library is imported
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.pyplot as plt

from luna.io.pds_index import PDSIndex
from luna.io.projection import LinearProjection, pixel_to_lonlat
from luna.models.dinov3 import DINOEncoder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("luna.scripts.query_faiss_index")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PRODUCT_ID = "M157906985RC"

INDEX_PATH = f"data/indices/faiss_{PRODUCT_ID}.index"
META_PATH  = f"data/indices/faiss_{PRODUCT_ID}_meta.pkl"
IMG_PATH   = f"data/_scratch/{PRODUCT_ID}.IMG"

QUERY_NPY = (
    "/Users/finnhertsch/projects/luna/data/_scratch/pos/"
    "Aristillus_2_M111592038LC.npy"
)

TOP_K        = 10
PDS3_OFFSET  = 5064
IMG_WIDTH    = 5064

DINO_LORA_DIR      = "/Users/finnhertsch/projects/luna/luna/models/dinov3"
DINO_MATRYOSHKA    = 384
DINO_DEVICE        = "mps"

ESSA_CKPT_PATH     = "checkpoints/ESSA_ResNet50FPN_best_version.pt"
ESSA_NUM_CLASSES   = 91          # original head size (Frankenstein fix)
ESSA_BOX_CLASSES   = 3           # background + skylight + pit
ESSA_SOURCE_RES    = 0.47        # m/px  — NAC resolution
ESSA_TARGET_RES    = 1.5         # m/px  — ESSA training resolution
ESSA_INPUT_SIZE    = 2048
ESSA_SCORE_THRESH  = 0.8
ESSA_LABEL_SKYLIGHT = 1
ESSA_LABEL_PIT      = 2

ZOOM_SIZE          = 1024        # px — human-readable context tile
OUTPUT_DIR         = "temp"
OUTPUT_PLOT        = f"{OUTPUT_DIR}/two_stage_pipeline_results.png"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    lo, hi = image.min(), image.max()
    if hi > lo:
        image = (image - lo) / (hi - lo)
    else:
        image = np.zeros_like(image)
    return (image * 255.0).astype(np.uint8)


def norm_raw_crop(crop: np.ndarray, fallback_range: float = 1500.0) -> np.ndarray:
    """Normalise a raw int16 NAC crop to [0, 1], ignoring fill pixels."""
    valid = crop[crop > -32752]
    lo, hi = (valid.min(), valid.max()) if len(valid) > 0 else (0.0, fallback_range)
    return np.clip((crop - lo) / max(hi - lo, fallback_range), 0.0, 1.0)


def format_coordinates(projection: LinearProjection | None, x: int, y: int) -> str:
    if projection is None:
        return "N/A"
    try:
        lon, lat = pixel_to_lonlat(projection, x, y)
        return f"Lat={lat:.5f}°, Lon={lon:.5f}°"
    except Exception:
        return "transformation error"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — DINO EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────
def build_query_vector(query_path: str) -> tuple[np.ndarray, np.ndarray]:
    log.info("Loading query anchor: %s", os.path.basename(query_path))
    encoder = DINOEncoder(
        lora_dir=DINO_LORA_DIR,
        matryoshka_dim=DINO_MATRYOSHKA,
        device=DINO_DEVICE,
    )
    image      = np.load(query_path)
    normalized = normalize_to_uint8(image)
    batch      = np.expand_dims(normalized, axis=0)
    vector     = encoder.encode(batch)
    del encoder
    log.info("DINO embedding generated — shape: %s", vector.shape)
    return vector, normalized


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────
def load_projection(product_id: str) -> LinearProjection | None:
    log.info("Loading geometry metadata for %s", product_id)
    pds = PDSIndex()
    try:
        raw = pds.geometry_for(product_id)
        geometry: dict[str, float | str] = {}
        for key, value in raw.items():
            parsed = getattr(value, "value", value)
            if parsed in (None, ""):
                continue
            try:
                geometry[str(key).lower()] = float(parsed)
            except (ValueError, TypeError):
                geometry[str(key).lower()] = parsed

        samples = int(geometry.get("line_samples") or geometry.get("samples") or 5064)
        lines   = int(geometry.get("image_lines")  or geometry.get("lines")   or 52224)
        proj    = LinearProjection.from_nac_geometry(geometry, samples=samples, lines=lines)
        log.info("Projection matrix created (%sx%s)", samples, lines)
        return proj
    except Exception as exc:
        log.warning("Failed to load geometry for %s: %s", product_id, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FAISS
# ─────────────────────────────────────────────────────────────────────────────
def load_index(index_path: str, meta_path: str):
    import faiss  # lazy — avoids OpenMP issues at module level

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    log.info("Loaded FAISS index — %d vectors", index.ntotal)
    return index, metadata


def search_index(index, query_vector: np.ndarray, top_k: int):
    import faiss  # lazy — avoids OpenMP issues at module level

    query = np.ascontiguousarray(query_vector, dtype=np.float32)
    faiss.normalize_L2(query)
    log.info("Searching top-%d nearest neighbours", top_k)
    return index.search(query, top_k)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — MASK R-CNN VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def _load_essa_model(device: str):
    """Load the ESSA Mask R-CNN checkpoint with its mixed head configuration."""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
    in_box  = model.roi_heads.box_predictor.cls_score.in_features
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    model.roi_heads.box_predictor  = FastRCNNPredictor(in_box, ESSA_BOX_CLASSES)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, ESSA_NUM_CLASSES)

    if not os.path.exists(ESSA_CKPT_PATH):
        raise FileNotFoundError(f"ESSA checkpoint not found: {ESSA_CKPT_PATH}")

    state = torch.load(ESSA_CKPT_PATH, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval().to(device)
    log.info("ESSA Mask R-CNN loaded on %s", device)
    return model


def _essa_crop_to_tensor(
    raw_img: np.memmap,
    cx: int,
    cy: int,
    source_size: int,
    height: int,
    device: str,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Crop a source_size window around (cx, cy), normalise, and resize to ESSA_INPUT_SIZE."""
    half  = source_size // 2
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(IMG_WIDTH, x0 + source_size), min(height, y0 + source_size)

    crop  = raw_img[y0:y1, x0:x1].copy().astype(np.float32)
    normed = norm_raw_crop(crop)

    pil   = Image.fromarray((normed * 255).astype(np.uint8))
    pil   = pil.resize((ESSA_INPUT_SIZE, ESSA_INPUT_SIZE), resample=Image.BILINEAR)
    arr   = np.array(pil, dtype=np.float32) / 255.0
    t     = torch.from_numpy(np.stack([arr] * 3, axis=0)).to(device, dtype=torch.float32)
    return t, (x0, y0, x1, y1)


def _project_mask_to_zoom(
    masks: np.ndarray,
    label_idx: int,
    src_box: tuple[int, int, int, int],
    zoom_box: tuple[int, int, int, int],
    source_size: int,
) -> np.ndarray | None:
    """Slice the ESSA output mask to the human-readable zoom window."""
    x0, y0, x1, y1   = src_box
    zx0, zy0, zx1, zy1 = zoom_box
    ratio = ESSA_INPUT_SIZE / source_size

    mx0 = int((zx0 - x0) * ratio)
    my0 = int((zy0 - y0) * ratio)
    mx1 = int((zx1 - x0) * ratio)
    my1 = int((zy1 - y0) * ratio)

    mask_crop = masks[label_idx, 0][my0:my1, mx0:mx1]
    resized   = Image.fromarray(mask_crop).resize((zx1 - zx0, zy1 - zy0), resample=Image.NEAREST)
    return np.array(resized) >= 0.5


def run_mask_rcnn_on_top_k(
    query_img: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    metadata_store: list,
) -> None:
    """Stage 2: run ESSA Mask R-CNN over 2048×2048 context tiles around DINO hits."""
    if not os.path.exists(IMG_PATH):
        log.warning("Raw .IMG file missing at %s — Stage 2 skipped", IMG_PATH)
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    try:
        model = _load_essa_model(device)
    except FileNotFoundError as exc:
        log.error("%s — Stage 2 skipped", exc)
        return

    scale_factor  = ESSA_SOURCE_RES / ESSA_TARGET_RES
    source_size   = int(ESSA_INPUT_SIZE / scale_factor)
    log.info(
        "Stage 2 resampling: %.2fm/px → %.2fm/px  |  source window: %dpx → %dpx",
        ESSA_SOURCE_RES, ESSA_TARGET_RES, source_size, ESSA_INPUT_SIZE,
    )

    file_size = os.path.getsize(IMG_PATH)
    height    = (file_size - PDS3_OFFSET) // (IMG_WIDTH * 2)
    raw_img   = np.memmap(
        IMG_PATH, dtype=np.int16, mode="r",
        offset=PDS3_OFFSET, shape=(height, IMG_WIDTH),
    )

    fig, axes = plt.subplots(3, 4, figsize=(18, 14))
    axes = axes.flatten()

    axes[0].imshow(query_img, cmap="gray")
    axes[0].set_title("STAGE 1: QUERY ANCHOR\n(DINO Search)", fontweight="bold", color="green")
    axes[0].axis("off")
    axes[1].axis("off")

    log.info("Running Mask R-CNN inference on %d context tiles...", TOP_K)

    label_names = {ESSA_LABEL_SKYLIGHT: "SKYLIGHT", ESSA_LABEL_PIT: "PIT"}

    for plot_idx, (rank, dist, idx) in enumerate(
        zip(range(1, TOP_K + 1), distances[0], indices[0]), start=2
    ):
        meta = metadata_store[idx]
        cx   = meta.x_offset + meta.width  // 2
        cy   = meta.y_offset + meta.height // 2

        tensor, src_box = _essa_crop_to_tensor(raw_img, cx, cy, source_size, height, device)
        x0, y0, x1, y1 = src_box

        # Human-readable zoom window
        half_z           = ZOOM_SIZE // 2
        zx0, zy0         = max(0, cx - half_z), max(0, cy - half_z)
        zx1, zy1         = min(IMG_WIDTH, zx0 + ZOOM_SIZE), min(height, zy0 + ZOOM_SIZE)
        zoom_box         = (zx0, zy0, zx1, zy1)

        human_crop = raw_img[zy0:zy1, zx0:zx1].copy().astype(np.float32)
        plot_img   = (norm_raw_crop(human_crop) * 255).astype(np.uint8)

        with torch.no_grad():
            out = model([tensor])[0]

        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        masks  = out["masks"].cpu().numpy()

        best_score  = 0.0
        best_mask   = None
        best_label  = "NONE"

        for i, score in enumerate(scores):
            if score >= ESSA_SCORE_THRESH and labels[i] in label_names:
                best_score = float(score)
                best_label = label_names[labels[i]]
                best_mask  = _project_mask_to_zoom(masks, i, src_box, zoom_box, source_size)
                break

        axes[plot_idx].imshow(plot_img, cmap="gray")
        if best_mask is not None:
            overlay              = np.zeros((zy1 - zy0, zx1 - zx0, 4), dtype=np.float32)
            overlay[best_mask, 0] = 1.0   # red channel
            overlay[best_mask, 3] = 0.5   # alpha
            axes[plot_idx].imshow(overlay)
            axes[plot_idx].set_title(
                f"Rank {rank}: {best_label}\nScore: {best_score:.2f}",
                color="red", fontsize=10,
            )
        else:
            axes[plot_idx].set_title(
                f"Rank {rank}: REJECTED\n(no pit detected)",
                color="black", fontsize=10,
            )
        axes[plot_idx].axis("off")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Stage 2 complete — results saved to %s", OUTPUT_PLOT)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    query_vector, query_image = build_query_vector(QUERY_NPY)
    projection                = load_projection(PRODUCT_ID)
    index, metadata_store     = load_index(INDEX_PATH, META_PATH)
    distances, indices        = search_index(index, query_vector, TOP_K)

    log.info("=" * 60)
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        meta   = metadata_store[idx]
        cx, cy = meta.x_offset + meta.width // 2, meta.y_offset + meta.height // 2
        coords = format_coordinates(projection, cx, cy)
        log.info("Rank %02d | Score: %.4f | Center: X=%d, Y=%d | %s", rank, dist, cx, cy, coords)
    log.info("=" * 60)

    run_mask_rcnn_on_top_k(query_image, distances, indices, metadata_store)


if __name__ == "__main__":
    main()