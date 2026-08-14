from datetime import datetime, timezone
from pathlib import Path
import pytest
import spiceypy as sp

from luna.io.spice_project import (
    furnish_kernels,
    furnish_kernels_for_date,
    DEFAULT_KERNEL_ROOT,
)


def test_furnish_kernels_for_date():
    # Test targeted kernel loading for a specific 2012 date
    dt_2012 = datetime(2012, 4, 15, 12, 0, 0, tzinfo=timezone.utc)
    count = furnish_kernels_for_date(dt_2012, DEFAULT_KERNEL_ROOT)
    
    assert count > 0
    # Base kernels (.tls, .tsc, .tpc, .bpc, .tf, .ti, de421.bsp) + 2012 SPK/CK
    # Verify SPICE pool has essential variables loaded
    radii = sp.bodvrd("MOON", "RADII", 3)[1]
    assert len(radii) == 3
    assert 1737.0 < radii[0] < 1738.5

    # Test loading another date in 2010 clears and loads targeted kernels
    dt_2010 = datetime(2010, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    count_2010 = furnish_kernels_for_date(dt_2010, DEFAULT_KERNEL_ROOT)
    assert count_2010 > 0


def test_furnish_kernels_batch():
    # Test batch furnish
    count = furnish_kernels(DEFAULT_KERNEL_ROOT)
    assert count > 0
