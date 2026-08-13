import numpy as np
import pytest
from luna.io.projection import LinearProjection, lonlat_to_pixel, pixel_to_lonlat


def test_linear_projection_lon_unwrapping():
    # Setup a mock geometry around lon = 312.4 (or -47.6)
    geom = {
        "upper_left_longitude": 312.0,
        "upper_right_longitude": 312.8,
        "lower_left_longitude": 312.0,
        "lower_right_longitude": 312.8,
        "upper_left_latitude": 20.0,
        "upper_right_latitude": 20.0,
        "lower_left_latitude": 19.0,
        "lower_right_latitude": 19.0,
        "line_samples": 1001,
        "image_lines": 2001,
    }
    
    proj = LinearProjection.from_nac_geometry(geom)
    
    # Test center coordinate in 0-360 range vs -180..180 range
    u1, v1 = proj._solve_uv(312.4, 19.5)
    u2, v2 = proj._solve_uv(-47.6, 19.5)
    
    assert np.isclose(u1, 0.5, atol=1e-5)
    assert np.isclose(v1, 0.5, atol=1e-5)
    assert np.isclose(u1, u2, atol=1e-5)
    assert np.isclose(v1, v2, atol=1e-5)
    
    # Test lonlat_to_pixel
    px1, py1 = lonlat_to_pixel(proj, 312.4, 19.5)
    px2, py2 = lonlat_to_pixel(proj, -47.6, 19.5)
    assert px1 == px2
    assert py1 == py2
    assert px1 == 500
    assert py1 == 1000


def test_linear_projection_meridian_crossing():
    # Setup a geometry that crosses the 0/360 boundary
    geom = {
        "upper_left_longitude": 359.5,
        "upper_right_longitude": 0.5,
        "lower_left_longitude": 359.5,
        "lower_right_longitude": 0.5,
        "upper_left_latitude": 10.0,
        "upper_right_latitude": 10.0,
        "lower_left_latitude": 9.0,
        "lower_right_latitude": 9.0,
        "line_samples": 1000,
        "image_lines": 2000,
    }
    
    proj = LinearProjection.from_nac_geometry(geom)
    
    # Center is at 360.0 == 0.0
    u1, v1 = proj._solve_uv(360.0, 9.5)
    u2, v2 = proj._solve_uv(0.0, 9.5)
    
    assert np.isclose(u1, 0.5, atol=1e-5)
    assert np.isclose(v1, 0.5, atol=1e-5)
    assert np.isclose(u1, u2, atol=1e-5)
    assert np.isclose(v1, v2, atol=1e-5)
