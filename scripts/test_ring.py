import os
import time
import threading
import numpy as np
from PIL import Image
from luna.screening.engine import DisruptorEngine, MappedStripe, push_stripe_to_ring
from luna.screening.engine import NACTransformer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

NAC_FILE    = "/Users/finnhertsch/projects/luna/data/_scratch/M109452586LC.IMG"
OUTPUT_DIR  = "temp/tiles_debug"
IMG_WIDTH   = 5064
HEADER_BYTES = 5064
TILE_SIZE   = 256
BATCH_SIZE  = 64


# ─────────────────────────────────────────────────────────────────────────────
# COLLECTOR THREAD
# ─────────────────────────────────────────────────────────────────────────────

def tile_collector(transformer: NACTransformer, sink: list, stop: threading.Event) -> None:
    """
    Drains completed batches from C memory into `sink` as NumPy arrays.
    Signals back to C++ after each copy so the feeder buffer can be reused.
    """
    print("[Collector] Running. Waiting for batches...")

    while not stop.is_set():
        if transformer.is_batch_ready:
            sink.append(transformer.get_current_batch())
            transformer.is_batch_ready = 0
        else:
            time.sleep(0.001)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(NAC_FILE):
        print(f"[ERROR] File not found: {NAC_FILE}")
        return

    # ── Geometry ──────────────────────────────────────────────────────────────
    file_size      = os.path.getsize(NAC_FILE)
    height         = (file_size - HEADER_BYTES) // (IMG_WIDTH * 2)
    expected_tiles = (IMG_WIDTH // TILE_SIZE) * (height // TILE_SIZE)

    print(f"[INFO] Dimensions   : {IMG_WIDTH} x {height}")
    print(f"[INFO] Expected tiles: {expected_tiles}")

    # ── Engine init ───────────────────────────────────────────────────────────
    engine      = DisruptorEngine(size=8, num_consumers=1)
    stripe      = MappedStripe(NAC_FILE)
    transformer = NACTransformer(engine, max_batch_size=BATCH_SIZE, consumer_id=0)

    # ── Threads ───────────────────────────────────────────────────────────────
    all_tiles   = []
    stop_event  = threading.Event()

    t_consumer  = threading.Thread(target=transformer.run_forever, daemon=True)
    t_collector = threading.Thread(
        target=tile_collector,
        args=(transformer, all_tiles, stop_event),
        daemon=True,
    )

    t_consumer.start()
    t_collector.start()

    # ── Produce ───────────────────────────────────────────────────────────────
    print(f"[INFO] Processing: {os.path.basename(NAC_FILE)}")
    t0 = time.time()

    push_stripe_to_ring(engine, stripe, width=IMG_WIDTH, height=height, stripe_id=1337)

    # ── Drain ─────────────────────────────────────────────────────────────────
    while True:
        collected = sum(b.shape[0] for b in all_tiles)
        if collected >= expected_tiles:
            break
        if (expected_tiles - collected) < BATCH_SIZE:
            transformer.trigger_final_flush()
        time.sleep(0.01)

    print(f"[INFO] Completed in {time.time() - t0:.4f}s")

    # ── Shutdown ──────────────────────────────────────────────────────────────
    stop_event.set()
    transformer.stop()
    t_collector.join()
    t_consumer.join()

    # ── Inspection dump ───────────────────────────────────────────────────────
    if not all_tiles:
        print("[ERROR] Tile buffer empty.")
        return

    full_stack = np.concatenate(all_tiles, axis=0)
    print(f"[INFO] {len(full_stack)} tiles in RAM.")

    limit = min(100, len(full_stack))
    print(f"[INFO] Writing {limit} tiles to {OUTPUT_DIR} ...")
    for i in range(limit):
        Image.fromarray(full_stack[i]).save(
            os.path.join(OUTPUT_DIR, f"tile_{i:03d}.png")
        )
    print("[INFO] Done. ✓")


if __name__ == "__main__":
    main()