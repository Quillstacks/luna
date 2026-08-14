"""Compatibility and optional dependency utilities for LUNA."""

from __future__ import annotations

import importlib.util
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def is_package_available(package_name: str) -> bool:
    """Check if a top-level package is installed without importing it."""
    return importlib.util.find_spec(package_name) is not None


def has_ml() -> bool:
    """Check if ML dependencies (torch, torchvision, transformers, peft) are installed."""
    return (
        is_package_available("torch")
        and is_package_available("transformers")
        and is_package_available("peft")
    )


def has_geo() -> bool:
    """Check if geospatial dependencies (shapely, pyproj) are installed."""
    return is_package_available("shapely")


def has_analysis() -> bool:
    """Check if manifold analysis dependencies (umap, sklearn) are installed."""
    return is_package_available("umap") and is_package_available("sklearn")


def has_dashboard() -> bool:
    """Check if dashboard dependencies (dash, datashader) are installed."""
    return is_package_available("dash") and is_package_available("datashader")


def require_extra(extra_name: str, feature_name: str, required_packages: list[str]) -> None:
    """Raise a clear, actionable ImportError if required dependencies for a feature are missing.

    Args:
        extra_name: Name of the pip extra (e.g., 'ml', 'geo', 'analysis', 'dashboard').
        feature_name: Name of the class or feature requiring the extra.
        required_packages: List of package names that must be present.

    Raises:
        ImportError: If any of the required packages are not available.
    """
    missing = [pkg for pkg in required_packages if not is_package_available(pkg)]
    if missing:
        missing_str = ", ".join(repr(p) for p in missing)
        raise ImportError(
            f"'{feature_name}' requires optional dependency extra [{extra_name}]. "
            f"Missing package(s): {missing_str}. "
            f"Install with: pip install 'luna[{extra_name}]' or pip install 'luna[all]'"
        )
