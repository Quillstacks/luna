import logging
import rasterio
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from luna.models.dinov3 import DINOEncoder
from luna.utils import LunaNormalizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("luna.scripts.test_embeddings")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_REPO_ID       = "F1nnSBK/lunar-dinov3-lora"
DINO_DIM         = 384
SAMPLES_PER_CLASS = 3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_DIR  = PROJECT_ROOT / "data" / "_scratch"
STATS_FILE   = PROJECT_ROOT / "data" / "nac_stats.json"

DINO_DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)
    return np.load(path).astype(np.float32)


def embed_files(
    paths: list[Path],
    encoder: DINOEncoder,
    normalizer: LunaNormalizer,
) -> np.ndarray:
    vectors = []
    for path in paths:
        arr  = load_array(path)
        arr  = normalizer.normalize(arr, path)
        vec  = encoder.encode(np.expand_dims(arr, axis=0))
        vectors.append(vec[0])
        log.debug("Embedded %s → %s", path.name, vec.shape)
    return np.array(vectors)


def print_similarity_matrix(
    sim_matrix: np.ndarray,
    labels: list[str],
    paths: list[Path],
) -> None:
    short_labels = [f"{l[:3]}{i}" for i, l in enumerate(labels)]
    header       = f"{'':>25} | " + " | ".join(short_labels)

    log.info("\n--- COSINE SIMILARITY MATRIX ---")
    log.info(header)
    log.info("-" * len(header))

    for i, (label, path) in enumerate(zip(labels, paths)):
        row = " | ".join(f"{v:5.2f}" for v in sim_matrix[i])
        log.info("%3s | %20s | %s", label, path.name[:20], row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pit_files = list((SCRATCH_DIR / "pits").glob("*.npy"))[:SAMPLES_PER_CLASS]
    neg_files = list((SCRATCH_DIR / "negatives").glob("*.npy"))[:SAMPLES_PER_CLASS]

    if not pit_files and not neg_files:
        log.error("No files found — check SCRATCH_DIR: %s", SCRATCH_DIR)
        return

    all_files = pit_files + neg_files
    labels    = ["PIT"] * len(pit_files) + ["NEG"] * len(neg_files)
    log.info("Embedding test: %d pits, %d negatives", len(pit_files), len(neg_files))

    normalizer = LunaNormalizer(STATS_FILE)
    encoder    = DINOEncoder(
        lora_dir          = HF_REPO_ID,
        base_weights_path = HF_REPO_ID,
        matryoshka_dim    = DINO_DIM,
        device            = DINO_DEVICE,
    )

    embeddings = embed_files(all_files, encoder, normalizer)
    del encoder

    sim_matrix = cosine_similarity(embeddings)
    print_similarity_matrix(sim_matrix, labels, all_files)


if __name__ == "__main__":
    main()