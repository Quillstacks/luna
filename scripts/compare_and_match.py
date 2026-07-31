import csv
import math
import sys
import json
from pathlib import Path
from luna.io.projection import LinearProjection, pixel_to_lonlat
from luna.io.nac_reader import read_nac

LUNAR_RADIUS_METERS = 1737400.0

def compute_lunar_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes surface distance between two points using localized flat approximation."""
    dlat = math.radians(lat1 - lat2)
    delta_lon = (lon1 % 360) - (lon2 % 360)
    if delta_lon > 180: delta_lon -= 360
    elif delta_lon < -180: delta_lon += 360
        
    dlon = math.radians(delta_lon)
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    dy = LUNAR_RADIUS_METERS * dlat
    dx = LUNAR_RADIUS_METERS * dlon * math.cos(mean_lat)
    return math.sqrt(dx**2 + dy**2)

def main():
    catalog_pits = []
    catalog_path = Path("catalogs/lpa.csv")
    if not catalog_path.exists():
        print(f"Error: Catalog not found at {catalog_path.resolve()}")
        sys.exit(1)

    with open(catalog_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["host"].strip() == "Aristarchus":
                raw_lon = float(row["longitude"])
                if raw_lon > 180.0: raw_lon -= 360.0
                catalog_pits.append({
                    "name": row["name"].strip(),
                    "lat": float(row["latitude"]),
                    "lon": raw_lon,
                    "depth_m": float(row["depth_m"]) if row["depth_m"] else None
                })

    json_path = Path("data/_scratch/refined_hits.json")
    if not json_path.exists():
        print(f"Error: Hits file not found at {json_path.resolve()}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        refined = json.load(f)

    matches = []
    new_discoveries = []
    matched_catalog_names = set()
    proj_cache = {}

    for h_dict in refined:
        # Reconstruct precise bilinear projection for the specific source NAC frame
        prod_id = h_dict["product_id"]
        if prod_id not in proj_cache:
            from luna.config import SCRATCH_DIR
            img_path = SCRATCH_DIR / f"{prod_id}.IMG"
            if img_path.exists():
                try:
                    img = read_nac(img_path, geometry=True)
                    proj_cache[prod_id] = LinearProjection.from_nac_geometry(img.geometry, samples=img.samples, lines=img.lines)
                except Exception as e:
                    proj_cache[prod_id] = None
            else:
                proj_cache[prod_id] = None

        proj = proj_cache.get(prod_id)
        
        if proj is not None:
            # Map absolute frame pixel coords through the non-linear inverse map
            # This directly neutralizes the trapezoidal mapping distortion
            precise_x = int(h_dict["x_offset"])
            precise_y = int(h_dict["y_offset"])
            true_lon, precise_lat = pixel_to_lonlat(proj, precise_x, precise_y)
            
            # Ensure unified longitude standard
            true_lon = true_lon - 360.0 if true_lon > 180.0 else true_lon
        else:
            precise_lat, true_lon = h_dict["lat"], h_dict["lon"]

        best_dist = float('inf')
        best_pit = None
        
        for pit in catalog_pits:
            dist = compute_lunar_distance(precise_lat, true_lon, pit["lat"], pit["lon"])
            if dist < best_dist:
                best_dist = dist
                best_pit = pit
                
        # With non-linear inversion active, match threshold is set to 150 meters
        if best_dist < 150.0 and best_pit is not None:
            matches.append({"rank": h_dict["rank"], "score": h_dict["dino_similarity"], "lat": precise_lat, "lon": true_lon, "pit": best_pit, "dist": best_dist})
            matched_catalog_names.add(best_pit["name"])
        else:
            new_discoveries.append({"rank": h_dict["rank"], "score": h_dict["dino_similarity"], "lat": precise_lat, "lon": true_lon, "best_pit": best_pit, "dist": best_dist})

    unmatched_catalog = [p for p in catalog_pits if p["name"] not in matched_catalog_names]
    

    report_path = Path("data/aristarchus_comparison_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Aristarchus Pit Detections vs LPA Catalog Comparison\n\n")
        f.write(f"We scanned the Aristarchus region and detected **{len(refined)}** candidates using the Stage-2 Dense Decoder.\n")
        f.write(f"We compared them to the **{len(catalog_pits)}** confirmed Aristarchus pits in the Lunar Pit Atlas (LPA) catalog.\n\n")
        
        f.write("## Summary\n")
        f.write(f"- **Total Detections:** {len(refined)}\n")
        f.write(f"- **Matches (Dist < 150m):** {len(matches)} / {len(catalog_pits)} cataloged pits\n")
        f.write(f"- **New Discoveries (Dist >= 150m):** {len(new_discoveries)}\n")
        f.write(f"- **Missed Cataloged Pits:** {len(unmatched_catalog)}\n\n")
        
        f.write("## Matches\n")
        f.write("| Rank | Class | Score | Lat (Det) | Lon (Det) | Catalog Name | Lat (Catalog) | Lon (Catalog) | Offset (m) |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for m in sorted(matches, key=lambda x: x["score"], reverse=True):
            p = m["pit"]
            f.write(f"| {m['rank']:02d} | PIT | {m['score']:.4f} | {m['lat']:.6f}° | {m['lon']:.6f}° | {p['name']} | {p['lat']:.6f}° | {p['lon']:.6f}° | {m['dist']:.1f}m |\n")
            
        f.write("\n## New Discoveries (Potential Pits)\n")
        f.write("| Rank | Class | Score | Lat (Det) | Lon (Det) | Closest Catalog Pit | Distance to Closest |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for nd in sorted(new_discoveries, key=lambda x: x["score"], reverse=True):
            p = nd["best_pit"]
            pit_name = p["name"] if p else "N/A"
            dist_str = f"{nd['dist']:.1f}m" if nd["dist"] != float('inf') else "N/A"
            f.write(f"| {nd['rank']:02d} | PIT | {nd['score']:.4f} | {nd['lat']:.6f}° | {nd['lon']:.6f}° | {pit_name} | {dist_str} |\n")
            
        f.write("\n## Missed Cataloged Pits (False Negatives)\n")
        f.write("| Catalog Name | Lat (Catalog) | Lon (Catalog) | Catalog Depth (m) |\n")
        f.write("| --- | --- | --- | --- |\n")
        for p in unmatched_catalog:
            depth_str = f"{p['depth_m']}m" if p['depth_m'] else "N/A"
            f.write(f"| {p['name']} | {p['lat']:.6f}° | {p['lon']:.6f}° | {depth_str} |\n")

    print(f"\nReport successfully saved to: {report_path.resolve()}")

if __name__ == "__main__":
    main()