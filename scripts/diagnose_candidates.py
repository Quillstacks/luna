import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["KMP_WARNINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from luna import LunaPipeline

def main():
    pipeline = LunaPipeline.from_pretrained("F1nnSBK/lunar-dinov3-lora")
    nac = "M1116841932RC"
    query_dir = "data/_scratch/pits/"
    out_dir = "data/_scratch/dumps/"
    
    print("Running LCVK Scan...")
    hits = pipeline.scan(nac, query_dir=query_dir)
    
    print(f"Scanning completed. Got {len(hits)} candidates.")
    
    # We run refine with score_thr=0.01 and essa_min_score=0.0 to see all detections
    print("Running ESSA Refiner with 0.01 threshold...")
    refined = pipeline.refine(
        hits,
        score_thr=0.01,
        essa_min_score=0.0,
        output_dir=out_dir,
        skip_preprocess=True,
    )
    
    print(f"\n--- Top 30 Candidates by ESSA / LCVK ---")
    print(f"{'Rank':<5} | {'Votes':<6} | {'DINO Score':<10} | {'ESSA Score':<10} | {'ESSA Class':<10} | {'Coords (Lon, Lat)':<25}")
    print("-" * 78)
    for i, h in enumerate(refined[:30]):
        print(f"{h.rank:<5} | {h.votes:<6} | {h.dino_score:<10.4f} | {h.essa_score:<10.4f} | {h.essa_class:<10} | ({h.essa_lon:.5f}, {h.essa_lat:.5f})")

if __name__ == "__main__":
    main()
