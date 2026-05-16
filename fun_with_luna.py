import logging
import time
from luna import LunaPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)

pipeline = LunaPipeline.from_pretrained("F1nnSBK/lunar-dinov3-lora")

# Profile Phase 1: DINOv3 Scanning & FAISS Retrieval
start_scan = time.perf_counter()
hits = pipeline.scan("M1118880788RC", query_dir="data/_scratch/pits/")
duration_scan = time.perf_counter() - start_scan

# Profile Phase 2: ESSA Mask R-CNN Verification
start_refine = time.perf_counter()
refined = pipeline.refine(hits, essa_min_score=0.5, output_dir="data/_scratch/dumps/")
duration_refine = time.perf_counter() - start_refine

# ---------------------------------------------------------------------------
# Console Reporting
# ---------------------------------------------------------------------------

print("\n" + "=" * 90)
print(f"VERIFIED LUNAR TARGETS ({len(refined)} Skylights/Pits Confirmed)")
print("=" * 90)

for h in refined:
    center_x = h.x_offset + 128
    center_y = h.y_offset + 128
    
    print(
        f"Rank {h.rank:02d} | [{h.essa_class.upper()}] Conf: {h.essa_score:.4f} | "
        f"Lat: {h.lat:10.6f}°, Lon: {h.lon:10.6f}° | "
        f"Center (X: {center_x:5d}, Y: {center_y:5d}) [Top-Left: {h.x_offset}, {h.y_offset}]"
    )

print("=" * 90)
print("PIPELINE PERFORMANCE SUMMARY:")
print(f"  - DINOv3 Vector Scan Stage : {duration_scan:6.2f} seconds")
print(f"  - ESSA Refinement Stage    : {duration_refine:6.2f} seconds")
print(f"  - Total Processing Time    : {duration_scan + duration_refine:6.2f} seconds")
print("=" * 90)