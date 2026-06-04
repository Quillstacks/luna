import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["KMP_WARNINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
from pathlib import Path
import numpy as np
from luna import LunaPipeline
from luna.screening.lcvk import LcvkEngine

def main():
    pipeline = LunaPipeline.from_pretrained("F1nnSBK/lunar-dinov3-lora")
    nac = "M1116841932RC"
    query_dir = Path("data/_scratch/pits/")
    
    # 1. Get alphabetically sorted list of query templates
    query_paths = sorted(query_dir.glob("*.npy"))
    query_names = [p.stem for p in query_paths]
    
    # 2. Run the LCVK scan manually to trace the matching template names
    index_prefix = f"data/_scratch/indices/lcvk_{nac}"
    print(f"Loading metadata for {nac}...")
    with open(f"{index_prefix}_meta.pkl", "rb") as f:
        metadata = pickle.load(f)
        
    print("Encoding query anchors...")
    query_vecs = pipeline._encode_queries(query_dir)
    
    print("Executing Native LCVK batch search...")
    # Binarize queries
    query_bin = LcvkEngine.binarize(query_vecs)
    
    vote_map = {}
    best_dist = {}
    best_query_idx = {} # maps tile index -> query index
    
    with LcvkEngine() as engine:
        engine.load_index(nac, f"{index_prefix}.bin")
        # k = 1000 search
        ids_mat, dists_mat = engine.batch_search(nac, query_bin, k=1000)
        
    for q_idx in range(ids_mat.shape[0]):
        for idx, dist in zip(ids_mat[q_idx], dists_mat[q_idx]):
            idx = int(idx)
            if idx < 0:
                continue
            vote_map[idx] = vote_map.get(idx, 0) + 1
            if idx not in best_dist or dist < best_dist[idx]:
                best_dist[idx] = float(dist)
                best_query_idx[idx] = q_idx

    # 3. Sort candidates by Hamming distance (ascending)
    ranked = sorted(best_dist.keys(), key=lambda i: best_dist[i])
    
    # 4. Run NMS
    print("Applying spatial NMS filtering (512px)...")
    nms_hits = pipeline._nms(ranked, vote_map, best_dist, metadata, top_k=100, min_dist_px=512.0)
    
    print(f"\n--- Top 30 LCVK Retrieval Matches (ESSA bypassed) ---")
    print(f"{'Rank':<5} | {'DINO Dist':<10} | {'Votes':<6} | {'Coords (Lon, Lat)':<25} | {'Best Matching Feature Template':<35}")
    print("-" * 95)
    
    for rank, (idx, votes, score) in enumerate(nms_hits[:30], start=1):
        meta = metadata[idx]
        q_idx = best_query_idx[idx]
        feature_name = query_names[q_idx]
        
        # Clean up feature name for display (e.g. remove redundant part numbers/image IDs)
        display_name = feature_name.split("_M1")[0].replace("_", " ")
        
        # Convert longitude to [-180, 180] range if it is in [0, 360] to match visualize_hits style
        lon = meta.lon
        if lon > 180:
            lon -= 360
            
        print(f"{rank:<5} | {score:<10.1f} | {votes:<6} | ({lon:>9.5f}°, {meta.lat:>8.5f}°) | {display_name:<35}")

if __name__ == "__main__":
    main()
