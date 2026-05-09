import matplotlib.pyplot as plt
import numpy as np
from luna.io.projection import get_image_of_roi
from luna.io.nac_reader import get_nacs_from_polygon
from luna.io.pds_index import PDSIndex

TARGET_LAT, TARGET_LON = 23.39629, 312.45872
WIDTH, HEIGHT = 1024, 1024
EPSILON = 0.01 

corners = [
    (TARGET_LON - EPSILON, TARGET_LAT + EPSILON),
    (TARGET_LON + EPSILON, TARGET_LAT + EPSILON),
    (TARGET_LON + EPSILON, TARGET_LAT - EPSILON),
    (TARGET_LON - EPSILON, TARGET_LAT - EPSILON)
]
features = get_nacs_from_polygon(corners)

if not features:
    raise ValueError("No NAC images found.")

# Sort features by quality: lowest Resolution first, then lowest Incidence
sorted_features = sorted(features, key=lambda f: (
    f["properties"]["attributes"]["Resolution"], 
    f["properties"]["attributes"]["Incidence"]
))

pds = PDSIndex()
selected_id = None
selected_attr = None

print("Filtering for the best image with valid geometry...")
for feat in sorted_features:
    pid = feat["properties"]["label"]
    try:
        geom = pds.geometry_for(pid)
        
        corner_keys = [
            "upper_left_longitude", "upper_right_longitude",
            "lower_left_longitude", "lower_right_longitude",
            "upper_left_latitude",  "upper_right_latitude",
            "lower_left_latitude",  "lower_right_latitude"
        ]
        
        # FIX: Explicitly cast to float to handle PVL objects/strings
        coords = []
        for k in corner_keys:
            val = geom.get(k, np.nan)
            try:
                coords.append(float(getattr(val, "value", val)))
            except (TypeError, ValueError):
                coords.append(np.nan)

        if np.isnan(coords).any():
            print(f"  Skipping {pid}: Invalid geometry (NaN corners).")
            continue
            
        selected_id = pid
        selected_attr = feat["properties"]["attributes"]
        break 
        
    except Exception as e:
        print(f"  Skipping {pid}: {e}")

if not selected_id:
    raise ValueError("None of the available images have valid geometry data.")

print(f"Selected Valid Image: {selected_id}")
print(f" - Resolution: {selected_attr['Resolution']} m/px")
print(f" - Incidence Angle: {selected_attr['Incidence']}°")

print("Fetching ROI...")
tile = get_image_of_roi(selected_id, lat=TARGET_LAT, lon=TARGET_LON, width=WIDTH, height=HEIGHT)

plt.figure(figsize=(8, 8))
plt.imshow(tile, cmap='gray', origin='upper')
plt.title(f"LROC NAC ROI: {selected_id}\n{selected_attr['Resolution']}m/px, {selected_attr['Incidence']}° Inc")
plt.colorbar(label="I/F Reflectance")
plt.gca().set_facecolor('xkcd:salmon') 
plt.tight_layout()
plt.show()