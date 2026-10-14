from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import spiceypy as sp
from spiceypy.utils.exceptions import SpiceyError

from luna.exceptions import LunaError, MissingKernelError
from luna.io.dem import (
    LunarDEM,
    default_dem,
    fetch_dem_for_region,
    get_sldem2015_tile_name,
)
from luna.io.spice_project import (
    BODY_FRAME,
    DEFAULT_KERNEL_ROOT,
    _nac_side_from_pid,
    _resolve_alt_m,
    _spice_call,
    furnish_kernels,
    furnish_kernels_for_date,
    ground_to_image,
    image_to_ground,
    lonlat_to_pixel_spice,
)


def _write_synthetic_spice_tree(root: Path) -> None:
    """Create a real ASCII SPICE kernel hierarchy loadable by CSPICE sp.furnsh."""
    for sub in ("lsk", "pck", "sclk", "ck"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    (root / "lsk" / "naif0012.tls").write_text(
        "KPL/LSK\n\\begindata\n"
        "DELTET/DELTA_T_A = 32.184\n"
        "DELTET/K         = 1.657D-3\n"
        "DELTET/EB        = 1.671D-2\n"
        "DELTET/M         = ( 6.239996D0 1.99096871D-7 )\n"
        "DELTET/DELTA_AT  = ( 10, @1972-JAN-1\n"
        "                     35, @2012-JUL-1 )\n"
        "\\begintext\n"
    )
    (root / "pck" / "pck00010.tpc").write_text(
        "KPL/PCK\n\\begindata\n"
        "BODY301_RADII = ( 1737.4 1737.4 1737.4 )\n"
        "\\begintext\n"
    )
    # Two SCLK versions: furnish_kernels_for_date must select only the latest (.tsc[-1])
    (root / "sclk" / "lro_clkcor_2020001_v00.tsc").write_text(
        "KPL/SCLK\n\\begindata\nSCLK_DATA_TYPE_85 = ( 1 )\n\\begintext\n"
    )
    (root / "sclk" / "lro_clkcor_2021001_v00.tsc").write_text(
        "KPL/SCLK\n\\begindata\nSCLK_DATA_TYPE_85 = ( 1 )\n\\begintext\n"
    )
    # Timed CK files + non-spacecraft/non-LROC .bc that must be filtered out
    (root / "ck" / "lrosc_2012090_2012120_v01.bc").write_text(
        "KPL/FK\n\\begindata\nTEST_WIN_2012 = 1\n\\begintext\n"
    )
    (root / "ck" / "lrosc_2010100_2010130_v01.bc").write_text(
        "KPL/FK\n\\begindata\nTEST_WIN_2010 = 1\n\\begintext\n"
    )
    (root / "ck" / "lrosa_2012090_2012120_v01.bc").write_text(
        "KPL/FK\n\\begindata\nTEST_SOLAR_ARRAY_IGNORED = 1\n\\begintext\n"
    )


def test_furnish_kernels_for_date(tmp_path: Path):
    kernel_root = (
        DEFAULT_KERNEL_ROOT
        if (DEFAULT_KERNEL_ROOT.exists() and any(DEFAULT_KERNEL_ROOT.iterdir()))
        else tmp_path / "spice"
    )
    if kernel_root == tmp_path / "spice":
        _write_synthetic_spice_tree(kernel_root)

    dt_2012 = datetime(2012, 4, 15, 12, 0, 0, tzinfo=timezone.utc)
    count = furnish_kernels_for_date(dt_2012, kernel_root)

    assert count == 4  # naif0012.tls, latest .tsc, pck00010.tpc, lrosc_2012090_2012120_v01.bc
    radii = sp.bodvrd("MOON", "RADII", 3)[1]
    assert len(radii) == 3
    assert 1737.0 < radii[0] < 1738.5
    # Verify solar array .bc was excluded
    with pytest.raises(SpiceyError):
        sp.gdpool("TEST_SOLAR_ARRAY_IGNORED", 0, 1)

    # Cached window hit (no reload)
    dt_2012_same_win = datetime(2012, 4, 20, 0, 0, 0, tzinfo=timezone.utc)
    assert furnish_kernels_for_date(dt_2012_same_win, kernel_root) == count

    # Switching to 2010 window clears pool and loads 2010 kernel
    dt_2010 = datetime(2010, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    count_2010 = furnish_kernels_for_date(dt_2010, kernel_root)
    assert count_2010 == 4
    sp.kclear()


def test_furnish_kernels_batch(tmp_path: Path):
    kernel_root = (
        DEFAULT_KERNEL_ROOT
        if (DEFAULT_KERNEL_ROOT.exists() and any(DEFAULT_KERNEL_ROOT.iterdir()))
        else tmp_path / "spice_batch"
    )
    if kernel_root == tmp_path / "spice_batch":
        _write_synthetic_spice_tree(kernel_root)

    sp.kclear()
    count = furnish_kernels(kernel_root)
    assert count >= 4
    # If external code clears CSPICE pool, furnish_kernels must recover and reload
    sp.kclear()
    assert sp.ktotal("ALL") == 0
    count_reloaded = furnish_kernels(kernel_root)
    assert count_reloaded == count
    sp.kclear()


def test_resolve_alt_m_optional_and_mutually_exclusive():
    # Default (neither alt_m nor dem) -> 0.0 m reference sphere
    assert _resolve_alt_m(120.0, 15.0, None, None) == 0.0

    # Explicit alt_m
    assert _resolve_alt_m(120.0, 15.0, -2010.5, None) == pytest.approx(-2010.5)

    # DEM lookup
    mock_dem = MagicMock()
    mock_dem.get_elevation_m.return_value = -1523.0
    assert _resolve_alt_m(120.0, 15.0, None, mock_dem) == pytest.approx(-1523.0)
    mock_dem.get_elevation_m.assert_called_once_with(120.0, 15.0)

    # Passing both raises ValueError
    with pytest.raises(ValueError, match="pass either alt_m or dem, not both"):
        _resolve_alt_m(120.0, 15.0, 0.0, mock_dem)


def test_missing_kernel_error_hierarchy_and_spice_call(tmp_path: Path):
    assert issubclass(MissingKernelError, LunaError)
    assert issubclass(MissingKernelError, RuntimeError)

    def _failing_spice(*_args):
        raise SpiceyError("SPICE(SPKINSUFFDATA) -- Insufficient ephemeris data")

    with pytest.raises(MissingKernelError, match="SPKINSUFFDATA"):
        _spice_call(_failing_spice, 1, 2)

    # Empty kernel pool -> ground_to_image and image_to_ground must raise MissingKernelError
    # (converting SpiceNOLEAPSECONDS / SpiceKERNELVARNOTFOUND into MissingKernelError)
    sp.kclear()
    lbl = tmp_path / "M1103677996LC.LBL"
    lbl.write_text(
        'PDS_VERSION_ID = PDS3\n'
        'PRODUCT_ID = "M1103677996LC"\n'
        'START_TIME = 2012-09-09T11:59:59.000\n'
        'STOP_TIME = 2012-09-09T12:00:01.000\n'
        'LRO:LINE_EXPOSURE_DURATION = 1.0 <ms>\n'
        'OBJECT = IMAGE\n'
        '  LINES = 2000\n'
        '  LINE_SAMPLES = 5064\n'
        'END_OBJECT = IMAGE\n'
        'END\n'
    )
    empty_root = tmp_path / "empty_kernels"
    empty_root.mkdir()
    with pytest.raises(MissingKernelError, match="NOLEAPSECONDS"):
        ground_to_image(lbl, 10.0, 10.0, kernel_root=empty_root, ensure_kernels=False)
    with pytest.raises(MissingKernelError, match="NOLEAPSECONDS"):
        image_to_ground(lbl, 2500.0, 1000.0, kernel_root=empty_root, ensure_kernels=False)


def test_nac_side_from_pid():
    assert _nac_side_from_pid("M1118880788RC") == "R"
    assert _nac_side_from_pid("M1118880788LE") == "L"
    assert _nac_side_from_pid("M1118880788L") == "L"
    with pytest.raises(ValueError, match="cannot infer NAC side"):
        _nac_side_from_pid("WAC12345")


def test_spice_roundtrip_and_dem_iteration_mocked(tmp_path: Path):
    """Verify exact SPICE ground_to_image <-> image_to_ground math with synthetic orbit."""
    lbl_path = tmp_path / "M1118880788LC.LBL"
    lbl_path.touch()

    fake_lbl = {
        "PRODUCT_ID": "M1118880788LC",
        "START_TIME": datetime(2012, 4, 15, 12, 0, 0, tzinfo=timezone.utc),
        "STOP_TIME": datetime(2012, 4, 15, 12, 0, 10, tzinfo=timezone.utc),
        "IMAGE": {"LINES": 10001, "LINE_SAMPLES": 5064},
        "LRO:LINE_EXPOSURE_DURATION": 1.0,  # 1 ms -> 0.001 s/line
    }

    r_moon = 1737.4
    alt_sc = 50.0  # 50 km orbit along equator (lat=0, lon sweeping -0.1 to +0.1 rad)
    et0 = 1000.0
    et1 = 1010.0
    omega = 0.002  # rad/s along longitude (X-Y plane)

    def fake_spkpos(target, et, frame, abcorr, obs):
        theta = (et - 1005.0) * omega
        r = r_moon + alt_sc
        pos = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
        return pos, 0.0

    def fake_pxform(from_frame, to_frame, et):
        # Camera frame at (theta):
        # +Z_cam points nadir (-radial = [-cos(theta), -sin(theta), 0])
        # +X_cam points along-track ([-sin(theta), cos(theta), 0])
        # +Y_cam points across-track ([0, 0, -1]) so right-handed: X x Y = Z
        theta = (et - 1005.0) * omega
        x_cam_bf = np.array([-np.sin(theta), np.cos(theta), 0.0])
        y_cam_bf = np.array([0.0, 0.0, -1.0])
        z_cam_bf = np.array([-np.cos(theta), -np.sin(theta), 0.0])
        rot_cam_to_bf = np.column_stack([x_cam_bf, y_cam_bf, z_cam_bf])
        if from_frame == BODY_FRAME and to_frame == "J2000":
            return np.eye(3)
        if from_frame == "J2000" and to_frame == "LRO_LROCNACL":
            return rot_cam_to_bf.T
        if from_frame == "LRO_LROCNACL" and to_frame == BODY_FRAME:
            return rot_cam_to_bf
        return np.eye(3)

    with (
        patch("luna.io.spice_project._prepare_kernels", return_value=fake_lbl),
        patch("luna.io.spice_project.pvl.load", return_value=fake_lbl),
        patch(
            "luna.io.spice_project._read_times",
            return_value=(et0, et1, 0.001, 10001, 5064),
        ),
        patch("luna.io.spice_project.sp.bodvrd", return_value=(3, [r_moon, r_moon, r_moon])),
        patch("luna.io.spice_project.sp.spkpos", side_effect=fake_spkpos),
        patch("luna.io.spice_project.sp.pxform", side_effect=fake_pxform),
    ):
        # 1. Default optional elevation (alt_m=None, dem=None -> 0.0 m sphere)
        s0, l0 = ground_to_image(lbl_path, lon_deg=0.0, lat_deg=0.0, ensure_kernels=False)
        assert s0 == pytest.approx(2548.0, abs=1e-4)
        assert l0 == pytest.approx(5000.0, abs=1e-4)

        lon_rt, lat_rt = image_to_ground(lbl_path, s0, l0, ensure_kernels=False)
        assert (lon_rt % 360.0) in (pytest.approx(0.0, abs=1e-6), pytest.approx(360.0, abs=1e-6))
        assert lat_rt == pytest.approx(0.0, abs=1e-6)

        # 2. Off-boresight roundtrip with explicit alt_m = -2000.0 m
        target_lon, target_lat = 0.05, -0.02
        s_alt, l_alt = ground_to_image(
            lbl_path, target_lon, target_lat, alt_m=-2000.0, ensure_kernels=False
        )
        lon_back, lat_back = image_to_ground(
            lbl_path, s_alt, l_alt, alt_m=-2000.0, ensure_kernels=False
        )
        assert lon_back == pytest.approx(target_lon, abs=1e-5)
        assert lat_back == pytest.approx(target_lat, abs=1e-5)

        # 3. Roundtrip with LunarDEM mock
        mock_dem = MagicMock()
        mock_dem.get_elevation_m.return_value = -1500.0
        s_dem, l_dem = ground_to_image(
            lbl_path, target_lon, target_lat, dem=mock_dem, ensure_kernels=False
        )
        lon_dem, lat_dem = image_to_ground(
            lbl_path, s_dem, l_dem, dem=mock_dem, ensure_kernels=False
        )
        assert lon_dem == pytest.approx(target_lon, abs=1e-5)
        assert lat_dem == pytest.approx(target_lat, abs=1e-5)

        # 4. Integer pixel wrapper
        si, li = lonlat_to_pixel_spice(lbl_path, 0.0, 0.0, ensure_kernels=False)
        assert (si, li) == (2548, 5000)

        # 5. Passing both alt_m and dem to image_to_ground raises ValueError
        with pytest.raises(ValueError, match="pass either alt_m or dem, not both"):
            image_to_ground(lbl_path, s0, l0, alt_m=0.0, dem=mock_dem, ensure_kernels=False)

        # 6. Out-of-window coordinate raises ValueError
        with pytest.raises(ValueError, match="outside NAC exposure window"):
            ground_to_image(lbl_path, lon_deg=45.0, lat_deg=0.0, ensure_kernels=False)

        # 7. Off-limb look ray (disc < 0) raises ValueError
        with pytest.raises(ValueError, match="misses the Moon"):
            image_to_ground(lbl_path, sample=500000.0, line=5000.0, ensure_kernels=False)

        # 8. Oscillating / non-converging DEM terminates cleanly after max_iter
        osc_dem = MagicMock()
        osc_dem.get_elevation_m.side_effect = [-1000.0, -2000.0, -1000.0, -2000.0]
        lon_osc, lat_osc = image_to_ground(
            lbl_path, s0, l0, dem=osc_dem, ensure_kernels=False, max_iter=4
        )
        assert osc_dem.get_elevation_m.call_count == 4
        assert not np.isnan(lon_osc) and not np.isnan(lat_osc)

    # 9. Inverted attitude pointing away from the Moon (t <= 0) raises ValueError
    with (
        patch("luna.io.spice_project._prepare_kernels", return_value=fake_lbl),
        patch("luna.io.spice_project.pvl.load", return_value=fake_lbl),
        patch(
            "luna.io.spice_project._read_times",
            return_value=(et0, et1, 0.001, 10001, 5064),
        ),
        patch("luna.io.spice_project.sp.bodvrd", return_value=(3, [r_moon, r_moon, r_moon])),
        patch("luna.io.spice_project.sp.spkpos", side_effect=fake_spkpos),
        patch("luna.io.spice_project.sp.pxform", return_value=np.diag([1.0, -1.0, 1.0])),
    ):
        with pytest.raises(ValueError, match="misses the Moon"):
            image_to_ground(lbl_path, 2548.0, 5000.0, ensure_kernels=False)


def test_sldem2015_tile_name_resolution():
    assert get_sldem2015_tile_name(14.091, 303.228) == "SLDEM2015_512_00N_30N_270_315"
    assert get_sldem2015_tile_name(-15.0, -45.0) == "SLDEM2015_512_30S_00S_315_360"
    assert get_sldem2015_tile_name(-45.0, 0.0) == "SLDEM2015_512_60S_30S_000_045"
    assert get_sldem2015_tile_name(60.0, 180.0) == "SLDEM2015_512_30N_60N_180_225"

    with pytest.raises(ValueError, match="out of SLDEM2015 tile bounds"):
        get_sldem2015_tile_name(65.0, 0.0)
    with pytest.raises(ValueError, match="out of SLDEM2015 tile bounds"):
        get_sldem2015_tile_name(-65.0, 0.0)
    with pytest.raises(ValueError, match="NaN"):
        get_sldem2015_tile_name(float("nan"), 0.0)


def test_fetch_dem_for_region_validation_and_mock(tmp_path: Path):
    with pytest.raises(ValueError, match="min_lat"):
        fetch_dem_for_region(20.0, 10.0, 0.0, 10.0, out_dir=tmp_path)
    with pytest.raises(ValueError, match="min_lon"):
        fetch_dem_for_region(0.0, 10.0, 50.0, 10.0, out_dir=tmp_path)

    mock_aux_resp = MagicMock()
    mock_aux_resp.content = b"<PAMDataset><PAMRasterBand><Scale>0.5</Scale></PAMRasterBand></PAMDataset>"
    mock_aux_resp.raise_for_status = MagicMock()

    mock_jp2_resp = MagicMock()
    mock_jp2_resp.__enter__.return_value = mock_jp2_resp
    mock_jp2_resp.headers = {"Content-Length": "8"}
    mock_jp2_resp.iter_content.return_value = [b"JP2_DATA"]
    mock_jp2_resp.raise_for_status = MagicMock()

    with patch("luna.io.dem.requests.Session") as mock_session_cls:
        sess = mock_session_cls.return_value
        sess.get.side_effect = [mock_aux_resp, mock_jp2_resp]
        paths = fetch_dem_for_region(10.0, 12.0, 300.0, 302.0, out_dir=tmp_path)
        assert len(paths) == 1
        assert paths[0].name == "SLDEM2015_512_00N_30N_270_315.JP2"
        assert paths[0].read_bytes() == b"JP2_DATA"
        assert (tmp_path / "SLDEM2015_512_00N_30N_270_315.JP2.aux.xml").exists()


def test_lunar_dem_guard_and_queries(tmp_path: Path):
    import rasterio
    from rasterio.transform import from_bounds

    # 1. Missing sidecar (scale=1.0 on SLDEM2015*.JP2) raises RuntimeError
    fake_jp2 = tmp_path / "SLDEM2015_512_00N_30N_270_315.JP2"
    fake_jp2.touch()

    mock_ds_bad = MagicMock()
    mock_ds_bad.scales = (1.0,)
    mock_ds_bad.offsets = (0.0,)
    mock_ds_bad.nodata = -32768

    with patch("rasterio.open", return_value=mock_ds_bad):
        with pytest.raises(RuntimeError, match="without GDAL sidecar"):
            LunarDEM([fake_jp2])

    # 2. Real GeoTIFF tiles straddling boundary lon=315.0 and lat=0.0, with nodata handling
    tile1_path = tmp_path / "tile_270_315.tif"
    tile2_path = tmp_path / "tile_315_360.tif"

    data1 = np.full((30, 45), -4000, dtype=np.int16)
    data1[0, 0] = -32768  # nodata at top-left (lon=270.0..271.0, lat=29.0..30.0)
    data1[:, -1] = -32768  # nodata along right edge (lon=315.0) in tile 1

    data2 = np.full((30, 45), -2000, dtype=np.int16)

    for p, left, right, arr in [
        (tile1_path, 270.0, 315.0, data1),
        (tile2_path, 315.0, 360.0, data2),
    ]:
        transform = from_bounds(left, 0.0, right, 30.0, 45, 30)
        with rasterio.open(
            p,
            "w",
            driver="GTiff",
            height=30,
            width=45,
            count=1,
            dtype="int16",
            crs="EPSG:4326",
            transform=transform,
            nodata=-32768,
        ) as dst:
            dst.scales = (0.5,)
            dst.offsets = (0.0,)
            dst.write(arr, 1)

    with LunarDEM([tile1_path, tile2_path]) as dem:
        # Interior of tile 1 (-4000 * 0.5 = -2000.0 m)
        assert dem.get_elevation_m(300.0, 15.0) == pytest.approx(-2000.0)
        # Negative longitude wrapping (-60.0 == 300.0)
        assert dem.get_elevation_m(-60.0, 15.0) == pytest.approx(-2000.0)
        # Exact bottom boundary lat=0.0 on tile 1 works without out-of-bounds nodata error
        assert dem.get_elevation_m(300.0, 0.0) == pytest.approx(-2000.0)
        # Exact shared boundary lon=315.0 falls through tile 1's nodata edge to tile 2 (-1000.0 m)
        assert dem.get_elevation_m(315.0, 15.0) == pytest.approx(-1000.0)
        # Exact right boundary lon=360.0 on tile 2
        assert dem.get_elevation_m(359.9, 15.0) == pytest.approx(-1000.0)

        # Nodata pixel raises ValueError in get_elevation_m
        with pytest.raises(ValueError, match="DEM nodata"):
            dem.get_elevation_m(270.2, 29.8)

        # Uncovered coordinate raises ValueError
        with pytest.raises(ValueError, match="not covered"):
            dem.get_elevation_m(10.0, 14.1)

        # 2D query_grid across tile boundary, nodata, and uncovered coordinates
        lons = np.array([[300.0, 315.0], [270.2, 10.0]])
        lats = np.array([[15.0, 15.0], [29.8, 15.0]])
        grid = dem.query_grid(lons, lats)
        assert grid.shape == (2, 2)
        assert grid[0, 0] == pytest.approx(-2000.0)
        assert grid[0, 1] == pytest.approx(-1000.0)
        assert np.isnan(grid[1, 0])  # nodata
        assert np.isnan(grid[1, 1])  # uncovered

        with pytest.raises(ValueError, match="must match"):
            dem.query_grid(np.zeros((2, 2)), np.zeros((2, 3)))

    # 3. default_dem raises FileNotFoundError on empty directory
    empty_dir = tmp_path / "empty_dem"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="no SLDEM JP2 tiles"):
        default_dem(empty_dir)
