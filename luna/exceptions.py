"""
luna.exceptions
~~~~~~~~~~~~~~~
Exception hierarchy for the Luna pipeline.
"""

from __future__ import annotations


class LunaError(Exception):
    """Base exception for all Luna pipeline errors."""
    pass


class IngestError(LunaError):
    """Raised when tile embedding or ingestion fails."""
    pass


class IndexError(LunaError):
    """Raised when Pithos index operations fail."""
    pass


class IndexNotFoundError(IndexError):
    """Raised when a required index file is not found."""
    pass


class SearchError(LunaError):
    """Raised when vector search operations fail."""
    pass


class DeltaError(LunaError):
    """Raised when delta buffer operations fail."""
    pass


class DeltaFullError(DeltaError):
    """Raised when the delta buffer is full and needs to be flushed."""
    pass


class NACNotFoundError(LunaError):
    """Raised when a NAC image is not found on disk or PDS fetch fails."""
    pass


class RefinementError(LunaError):
    """Raised when ESSA refinement fails."""
    pass
