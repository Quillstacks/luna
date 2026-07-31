import sys
import pickle
import math
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from luna.pipeline import LunaPipeline
from luna.io.pds_index import PDSIndex
from luna.screening.pithos import PithosMIDB

# The 14 cataloged pits and their closest tile_idx in M1129801944RC (from our previous verification)
catalog_pit_tiles = {
    "Aristarchus 3c": 2664,
    "Aristarchus 3b": 2690,
    "Aristarchus 3a": 2690,
    "Aristarchus 4": 2766,
    "Aristarchus 8": 2819,
    "Aristarchus 1": 2892,
    "Aristarchus 2a": 2964,
    "Aristarchus 2b": 2965,
    "Aristarchus 2c": 2991,
    "Aristarchus 9": 3018,
    "Aristarchus 5": 4484,
    "Aristarchus 6": 5187,
    "Aristarchus 7a": 5236,
    "Aristarchus 7b": 5235
}

def main():
    from luna.config import HF_REPO_ID
    pipeline = LunaPipeline.from_pretrained(HF_REPO_ID)
    pid = "M1129801944RC"
    index_prefix = f"data/_scratch/indices/pithos_{pid}"
    
    print("Loading queries...")
    query_vecs, families, thresholds = pipeline._load_and_encode_pit_queries("data/_scratch/pits/")
    
    print(f"Loading metadata for {pid}...")
    with open(f"{index_prefix}_meta.pkl", "rb") as f:
        metadata = pickle.load(f)
        
    print(f"Executing Pithos search for {pid}...")
    db = PithosMIDB()
    db.load_index(pid, f"{index_prefix}.bin")
    
    voting_mask = np.zeros(len(metadata), dtype=np.uint8)
    resonant_count = db.query_planetary_grid(
        index_name=pid,
        queries=query_vecs,
        families=families.astype(np.int32),
        thresholds=thresholds.astype(np.int32),
        voting_mask=voting_mask,
    )
    db.drop_index(pid)
    
    print(f"\nResonant search results for the 14 cataloged pit tiles:")
    print("=================================================================")
    print("Pit Name         | Tile ID | In Stage-1 Candidates? | Votes / 8 Families")
    print("-----------------------------------------------------------------")
    for name, tile_idx in catalog_pit_tiles.items():
        votes = voting_mask[tile_idx]
        in_cands = "YES" if votes > 0 else "NO"
        print(f"{name:<16} | {tile_idx:<7} | {in_cands:<22} | {votes} families")
    print("=================================================================\n")
    
    # Check if they were filtered out by NMS
    candidate_indices = np.where(voting_mask != 0)[0].tolist()
    print(f"Total Stage-1 candidates (votes > 0): {len(candidate_indices)}")
    
    # Run NMS
    nms_hits = pipeline._nms(
        candidate_indices, voting_mask, metadata,
        top_k=150, min_dist_px=512.0
    )
    nms_indices = [hit[0] for hit in nms_hits]
    
    print("\nNMS selection results:")
    print("=================================================================")
    print("Pit Name         | Tile ID | Passed NMS?")
    print("-----------------------------------------------------------------")
    for name, tile_idx in catalog_pit_tiles.items():
        passed = "YES" if tile_idx in nms_indices else "NO"
        print(f"{name:<16} | {tile_idx:<7} | {passed}")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
