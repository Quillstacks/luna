import glob
import logging
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from luna.models.dinov3 import DINOEncoder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("luna.scripts.test_embeddings")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DINO_LORA_DIR   = "/Users/finnhertsch/projects/luna/luna/models/dinov3"
DINO_DIM        = 384
DINO_DEVICE     = "mps"

DATASET_ROOT    = "/Users/finnhertsch/projects/luna_hole/data/processed/dataset/test"
PIT_GLOB        = f"{DATASET_ROOT}/pits/*.npy"
NEG_GLOB        = f"{DATASET_ROOT}/negatives/*.npy"
SAMPLES_PER_CLASS = 3


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


def embed_files(paths: list[str], encoder: DINOEncoder) -> np.ndarray:
    vectors = []
    for path in paths:
        arr = normalize_to_uint8(np.load(path))
        vec = encoder.encode(np.expand_dims(arr, axis=0))
        vectors.append(vec[0])
        log.debug("Embedded %s → shape %s", os.path.basename(path), vec.shape)
    return np.array(vectors)


def print_similarity_matrix(
    sim_matrix: np.ndarray,
    labels: list[str],
    file_paths: list[str],
) -> None:
    short_labels = [f"{l[:3]}{i}" for i, l in enumerate(labels)]
    header       = f"{'':>25} | " + " | ".join(short_labels)
    separator    = "-" * len(header)

    log.info("\n--- COSINE SIMILARITY MATRIX ---")
    log.info(header)
    log.info(separator)

    for i, (label, path) in enumerate(zip(labels, file_paths)):
        name = os.path.basename(path)[:20]
        row  = " | ".join(f"{v:5.2f}" for v in sim_matrix[i])
        log.info("%3s | %20s | %s", label, name, row)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    pit_files = glob.glob(PIT_GLOB)[:SAMPLES_PER_CLASS]
    neg_files = glob.glob(NEG_GLOB)[:SAMPLES_PER_CLASS]

    if not pit_files and not neg_files:
        log.error("No .npy files found — check PIT_GLOB / NEG_GLOB paths")
        return

    all_files = pit_files + neg_files
    labels    = ["PIT"] * len(pit_files) + ["NEG"] * len(neg_files)
    log.info("Embedding test: %d pits, %d negatives", len(pit_files), len(neg_files))

    encoder    = DINOEncoder(lora_dir=DINO_LORA_DIR, matryoshka_dim=DINO_DIM, device=DINO_DEVICE)
    embeddings = embed_files(all_files, encoder)
    del encoder

    sim_matrix = cosine_similarity(embeddings)
    print_similarity_matrix(sim_matrix, labels, all_files)


if __name__ == "__main__":
    main()