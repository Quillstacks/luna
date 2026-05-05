# tests/test_projection.py
"""Tests for luna.io.projection — LinearProjection, lonlat_to_pixel, pixel_to_lonlat.

We use a synthetic rectangular NAC geometry with analytically known corner
coordinates so every expected value can be derived by hand.

Geometry (samples=1000, lines=2000):
    upper-left  (lon=350.0, lat=-4.0)   → pixel (0,    0)
    upper-right (lon=352.0, lat=-4.0)   → pixel (999,  0)
    lower-left  (lon=350.0, lat=-6.0)   → pixel (0,    1999)
    lower-right (lon=352.0, lat=-6.0)   → pixel (999,  1999)
"""


import pytest

from luna.io.projection import LinearProjection, lonlat_to_pixel, pixel_to_lonlat


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

SAMPLES = 1000
LINES   = 2000

RECT_GEOM: dict = {
    "upper_left_longitude":  350.0,
    "upper_left_latitude":    -4.0,
    "upper_right_longitude": 352.0,
    "upper_right_latitude":   -4.0,
    "lower_left_longitude":  350.0,
    "lower_left_latitude":    -6.0,
    "lower_right_longitude": 352.0,
    "lower_right_latitude":   -6.0,
    "line_samples": SAMPLES,
    "image_lines":  LINES,
}


@pytest.fixture
def proj() -> LinearProjection:
    return LinearProjection.from_nac_geometry(RECT_GEOM)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestFromNacGeometry:
    def test_dimensions_stored(self, proj):
        assert proj.samples == SAMPLES
        assert proj.lines   == LINES

    def test_corner_arrays_shape(self, proj):
        assert proj.lon_corners.shape == (4,)
        assert proj.lat_corners.shape == (4,)

    def test_missing_corner_raises(self):
        bad = {k: v for k, v in RECT_GEOM.items() if k != "upper_left_longitude"}
        with pytest.raises(ValueError, match="Missing or invalid corner"):
            LinearProjection.from_nac_geometry(bad)

    def test_nan_corner_raises(self):
        bad = {**RECT_GEOM, "upper_left_longitude": float("nan")}
        with pytest.raises(ValueError, match="NaN"):
            LinearProjection.from_nac_geometry(bad)

    def test_missing_dimensions_raises(self):
        bad = {k: v for k, v in RECT_GEOM.items()
               if k not in ("line_samples", "image_lines")}
        with pytest.raises(ValueError, match="Missing image dimensions"):
            LinearProjection.from_nac_geometry(bad)

    def test_explicit_dimensions_override_geom(self):
        proj = LinearProjection.from_nac_geometry(RECT_GEOM, samples=512, lines=1024)
        assert proj.samples == 512
        assert proj.lines   == 1024


# ---------------------------------------------------------------------------
# pixel_to_lonlat — forward direction, analytically exact for rectangles
# ---------------------------------------------------------------------------

class TestPixelToLonLat:
    @pytest.mark.parametrize("x, y, expected_lon, expected_lat", [
        (0,          0,     350.0, -4.0),   # upper-left corner
        (SAMPLES-1,  0,     352.0, -4.0),   # upper-right corner
        (0,          LINES-1, 350.0, -6.0), # lower-left corner
        (SAMPLES-1,  LINES-1, 352.0, -6.0), # lower-right corner
        (499,        999,   350.999, -4.999),  # approximately centre
    ])
    def test_corners_and_centre(self, proj, x, y, expected_lon, expected_lat):
        lon, lat = pixel_to_lonlat(proj, x, y)
        assert lon == pytest.approx(expected_lon, abs=1e-3)
        assert lat == pytest.approx(expected_lat, abs=1e-3)


# ---------------------------------------------------------------------------
# lonlat_to_pixel — inverse direction
# ---------------------------------------------------------------------------

class TestLonLatToPixel:
    @pytest.mark.parametrize("lon, lat, expected_x, expected_y", [
        (350.0, -4.0,  0,          0),
        (352.0, -4.0,  SAMPLES-1,  0),
        (350.0, -6.0,  0,          LINES-1),
        (352.0, -6.0,  SAMPLES-1,  LINES-1),
        (351.0, -5.0,  499,        999),   # centre (rounded)
    ])
    def test_known_coordinates(self, proj, lon, lat, expected_x, expected_y):
        x, y = lonlat_to_pixel(proj, lon, lat)
        assert x == pytest.approx(expected_x, abs=1)
        assert y == pytest.approx(expected_y, abs=1)


# ---------------------------------------------------------------------------
# Round-trip stability
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """pixel → lonlat → pixel and lonlat → pixel → lonlat must be stable."""

    @pytest.mark.parametrize("x, y", [
        (0, 0), (999, 0), (0, 1999), (999, 1999),   # corners
        (500, 1000),                                  # centre
        (123, 456), (876, 1543),                      # interior samples
    ])
    def test_pixel_lonlat_pixel(self, proj, x, y):
        lon, lat = pixel_to_lonlat(proj, x, y)
        x2, y2  = lonlat_to_pixel(proj, lon, lat)
        assert x2 == pytest.approx(x, abs=1)
        assert y2 == pytest.approx(y, abs=1)

    @pytest.mark.parametrize("lon, lat", [
        (350.0, -4.0), (352.0, -6.0),
        (351.0, -5.0),
        (350.5, -4.75), (351.8, -5.3),
    ])
    def test_lonlat_pixel_lonlat(self, proj, lon, lat):
        x, y       = lonlat_to_pixel(proj, lon, lat)
        lon2, lat2 = pixel_to_lonlat(proj, x, y)
        # Rounding to int pixel introduces up to half-pixel error in lon/lat.
        assert lon2 == pytest.approx(lon, abs=0.005)
        assert lat2 == pytest.approx(lat, abs=0.005)