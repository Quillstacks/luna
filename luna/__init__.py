"""LUNA — Planetary Photogrammetry, Feature Extraction, and Lunar Pit Detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import LunaConfig
from .exceptions import (
    DeltaError,
    DeltaFullError,
    IndexError,
    IndexNotFoundError,
    IngestError,
    LunaError,
    MissingKernelError,
    NACNotFoundError,
    RefinementError,
    SearchError,
)
from .io.coverage import select_coverage_nacs
from .io.dem import (
    LunarDEM,
    default_dem,
    fetch_dem_for_region,
    get_sldem2015_tile_name,
)
from .io.kernel_fetch import ensure_kernels_for_date
from .io.nac_reader import NACImage, read_nac
from .io.pds_fetch import (
    DEFAULT_HEADERS,
    DEFAULT_USER_AGENT,
    fetch_nac,
    get_pds_session,
)
from .io.pds_index import PDSIndex
from .io.projection import LinearProjection, get_image_of_roi
from .io.spice_project import (
    BODY_FRAME,
    ensure_kernels_for_label,
    furnish_kernels,
    furnish_kernels_for_date,
    ground_to_image,
    image_to_ground,
    lonlat_to_pixel_spice,
)

__version__ = "0.0.1"

# Type checking guards for static analysis / IDE autocompletion
if TYPE_CHECKING:
    from .metrics import MetricsReport
    from .models.essa import RefinedHit
    from .pipeline import CandidateHit, LunaPipeline

__all__ = [
    "__version__",
    # Core Photogrammetry, DEM & I/O
    "fetch_nac",
    "read_nac",
    "get_pds_session",
    "DEFAULT_USER_AGENT",
    "DEFAULT_HEADERS",
    "PDSIndex",
    "LinearProjection",
    "get_image_of_roi",
    "ground_to_image",
    "image_to_ground",
    "lonlat_to_pixel_spice",
    "furnish_kernels",
    "furnish_kernels_for_date",
    "ensure_kernels_for_date",
    "ensure_kernels_for_label",
    "BODY_FRAME",
    "select_coverage_nacs",
    "NACImage",
    "LunarDEM",
    "fetch_dem_for_region",
    "get_sldem2015_tile_name",
    "default_dem",
    # Config & Exceptions
    "LunaConfig",
    "LunaError",
    "IngestError",
    "IndexError",
    "IndexNotFoundError",
    "SearchError",
    "DeltaError",
    "DeltaFullError",
    "NACNotFoundError",
    "RefinementError",
    "MissingKernelError",
    # High-level Pipeline / ML (Lazy loaded via PEP 562)
    "LunaPipeline",
    "CandidateHit",
    "RefinedHit",
    "MetricsReport",
]


def __getattr__(name: str) -> Any:
    """Lazy module loader for heavy pipeline and ML components (PEP 562)."""
    if name in ("LunaPipeline", "CandidateHit"):
        from ._compat import require_extra

        require_extra("ml", name, ["torch", "transformers", "peft"])
        from .pipeline import CandidateHit, LunaPipeline

        return LunaPipeline if name == "LunaPipeline" else CandidateHit

    if name == "RefinedHit":
        from ._compat import require_extra

        require_extra("ml", name, ["torch"])
        from .models.essa import RefinedHit

        return RefinedHit

    if name == "MetricsReport":
        from .metrics import MetricsReport

        return MetricsReport

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
