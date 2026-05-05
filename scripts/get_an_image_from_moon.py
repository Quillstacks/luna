import matplotlib.pyplot as plt
from luna.io.projection import get_image_of_roi
from luna.io.nac_reader import get_nacs_from_polygon # Adjust import path if needed

# Target ROI center
target_lat = -4.53220
target_lon = 351.48930

# 1. Create a tiny bounding box polygon around our target to query QuickMap
epsilon = 0.01 # Roughly ~300 meters at the equator
corners = [
    (target_lon - epsilon, target_lat + epsilon),
    (target_lon + epsilon, target_lat + epsilon),
    (target_lon + epsilon, target_lat - epsilon),
    (target_lon - epsilon, target_lat - epsilon)
]

print("Searching QuickMap for NAC images in this region...")
features = get_nacs_from_polygon(corners)

if not features:
    raise ValueError("No NAC images found overlapping this location!")

# 2. Extract the Product ID from the first result
# QuickMap returns GeoJSON-like features, so the ID is usually in the properties.
PRODUCT_ID = features[0]["properties"]["label"]
print(f"Found image: {PRODUCT_ID}")

# 3. Fetch, project, and crop the tile
print("Fetching and projecting image tile...")
tile = get_image_of_roi(PRODUCT_ID, lat=target_lat, lon=target_lon, width=1024, height=1024)

# 4. Plot!
plt.figure(figsize=(6, 6))

# The valid image data will be gray/light gray. 
plt.imshow(tile, cmap='gray', origin='upper')

plt.title(f"LROC NAC ROI ({PRODUCT_ID})\nCentered at Lat: {target_lat}, Lon: {target_lon}")
plt.colorbar(label="Pixel Intensity")

# Make any NaNs visually pop out as salmon
plt.gca().set_facecolor('xkcd:salmon') 

plt.tight_layout()
plt.show()