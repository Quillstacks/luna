import pytest

import luna
from luna._compat import (
    is_package_available,
    require_extra,
)


def test_package_availability():
    assert is_package_available("sys") is True
    assert is_package_available("nonexistent_package_12345") is False


def test_require_extra_raises_actionable_error():
    with pytest.raises(ImportError) as excinfo:
        require_extra("ml", "TestFeature", ["nonexistent_package_abc"])

    err_msg = str(excinfo.value)
    assert "'TestFeature' requires optional dependency extra [ml]" in err_msg
    assert "pip install 'luna[ml]'" in err_msg


def test_luna_top_level_exports():
    assert hasattr(luna, "__version__")
    assert hasattr(luna, "fetch_nac")
    assert hasattr(luna, "read_nac")
    assert hasattr(luna, "ground_to_image")
    assert hasattr(luna, "image_to_ground")
    assert hasattr(luna, "PDSIndex")
    assert hasattr(luna, "LinearProjection")
    assert hasattr(luna, "LunarDEM")
    assert hasattr(luna, "fetch_dem_for_region")
    assert hasattr(luna, "get_sldem2015_tile_name")
    assert hasattr(luna, "default_dem")
    assert hasattr(luna, "MissingKernelError")
    assert hasattr(luna, "LunaConfig")
    assert hasattr(luna, "LunaError")


def test_luna_lazy_getattr_unknown_attribute():
    with pytest.raises(AttributeError, match="module 'luna' has no attribute 'nonexistent_symbol'"):
        _ = getattr(luna, "nonexistent_symbol")

