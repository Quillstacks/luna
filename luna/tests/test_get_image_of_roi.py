"""Tests for luna.io.nac_reader.get_image_of_roi.

External I/O (fetch_nac, read_nac) and the projection layer are mocked so the
tests run offline and deterministically.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from luna.io.projection import LinearProjection, get_image_of_roi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLES = 1000
LINES   = 2000
PRODUCT = "M1522860307RC"

RECT_GEOM: dict = {
    "upper_left_longitude":  350.0,  "upper_left_latitude":   -4.0,
    "upper_right_longitude": 352.0,  "upper_right_latitude":  -4.0,
    "lower_left_longitude":  350.0,  "lower_left_latitude":   -6.0,
    "lower_right_longitude": 352.0,  "lower_right_latitude":  -6.0,
    "line_samples": SAMPLES,
    "image_lines":  LINES,
}


def _make_mock_img(pixel_value: float = 0.5) -> MagicMock:
    """Return a NACImage mock with a uniform float32 pixel array."""
    img            = MagicMock()
    img.samples    = SAMPLES
    img.lines      = LINES
    img.geometry   = RECT_GEOM
    img.pixels     = np.full((LINES, SAMPLES), pixel_value, dtype=np.float32)
    return img


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_io(tmp_path):
    """Patch fetch_nac and read_nac; yield the mock NACImage for inspection."""
    img = _make_mock_img()
    with (
        patch("luna.io.projection.fetch_nac", return_value=tmp_path / "fake.img") as _fetch,
        patch("luna.io.projection.read_nac",  return_value=img) as _read,
    ):
        yield img, _fetch, _read


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

class TestReturnShape:
    def test_default_256x256(self, mock_io):
        img, _, _ = mock_io
        tile = get_image_of_roi(PRODUCT, lat=-5.0, lon=351.0)
        assert tile.shape == (256, 256)

    def test_custom_size(self, mock_io):
        tile = get_image_of_roi(PRODUCT, lat=-5.0, lon=351.0, width=128, height=64)
        assert tile.shape == (64, 128)

    def test_dtype_is_float32(self, mock_io):
        tile = get_image_of_roi(PRODUCT, lat=-5.0, lon=351.0)
        assert tile.dtype == np.float32


class TestIoCallContract:
    def test_read_nac_called_with_geometry_true(self, mock_io):
        _, _fetch, _read = mock_io
        get_image_of_roi(PRODUCT, lat=-5.0, lon=351.0)
        _read.assert_called_once()
        _, kwargs = _read.call_args
        assert kwargs.get("geometry") is True

    def test_fetch_nac_called_with_product_id(self, mock_io):
        _, _fetch, _ = mock_io
        get_image_of_roi(PRODUCT, lat=-5.0, lon=351.0)
        _fetch.assert_called_once_with(PRODUCT)


# ---------------------------------------------------------------------------
# Pixel content — interior crop (no padding needed)
# ---------------------------------------------------------------------------

class TestInteriorCrop:
    def test_pixel_values_match_source(self, mock_io):
        """Pixels in an interior crop must equal the source array slice."""
        img, _, _ = mock_io
        # Stamp a unique pattern around the expected centre pixel.
        proj  = LinearProjection.from_nac_geometry(RECT_GEOM)
        # Fix: use the correct floored center of the 1000x2000 array
        col, row = 499, 999       
        img.pixels[row - 128 : row + 128, col - 128 : col + 128] = 0.99

        tile = get_image_of_roi(PRODUCT, lat=-5.0, lon=351.0, width=256, height=256)
        assert np.all(tile == pytest.approx(0.99, abs=1e-4))

    def test_no_nan_in_interior_crop(self, mock_io):
        tile = get_image_of_roi(PRODUCT, lat=-5.0, lon=351.0)
        assert not np.isnan(tile).any()


# ---------------------------------------------------------------------------
# Edge / corner crops — NaN padding
# ---------------------------------------------------------------------------

class TestEdgePadding:
    @pytest.mark.parametrize("lat, lon, nan_region", [
        # Upper-left corner: expect NaN in top-left quadrant of tile.
        (-4.002, 349.998, (slice(None, 128), slice(None, 128))),
        # Lower-right corner: expect NaN in bottom-right quadrant of tile.
        # Fix: Use -6.002 and 352.002 to be just outside the bottom-right bound
        (-6.002, 352.002, (slice(128, None), slice(128, None))),
    ])
    def test_nan_padding_at_border(self, mock_io, lat, lon, nan_region):
        tile = get_image_of_roi(PRODUCT, lat=lat, lon=lon, width=256, height=256)
        assert tile.shape == (256, 256), "Shape must always equal requested size"
        assert np.isnan(tile[nan_region]).all(), "Out-of-frame region must be NaN"


# ---------------------------------------------------------------------------
# Out-of-bounds centre → ValueError
# ---------------------------------------------------------------------------

class TestOutOfBounds:
    @pytest.mark.parametrize("lat, lon", [
        (-3.0, 351.0),    # north of frame
        (-7.0, 351.0),    # south of frame
        (-5.0, 348.0),    # west of frame
        (-5.0, 354.0),    # east of frame
    ])
    def test_raises_for_centre_outside_frame(self, mock_io, lat, lon):
        with pytest.raises(ValueError, match="outside the NAC frame"):
            get_image_of_roi(PRODUCT, lat=lat, lon=lon)