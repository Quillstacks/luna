"""Shared data models and structural interfaces for NAC tile retrieval.

Tile-based image retrieval works by slicing a NAC frame into fixed-size
sub-images (tiles), embedding each tile into a latent vector space, and
querying a vector store to find the closest known Product-ID match.

This module defines the two core data containers:

- ``TileMetadata`` — provenance record that maps a tile back to its
  source image and pixel position within it.
- ``SearchResult``  — ranked lookup result pairing a similarity score
  with the matched product ID and the originating tile's metadata.

It also defines the two structural ``Protocol`` interfaces that every
concrete implementation must satisfy:

- ``EmbeddingModel`` — anything that turns a batch of raw tile arrays
  into a matrix of embedding vectors (e.g. a CLIP or Matryoshka encoder).
- ``VectorStore``    — anything that accepts a batch of query vectors and
  returns ranked ``SearchResult`` lists (e.g. a FAISS or Qdrant wrapper).

Coding against these protocols rather than concrete classes keeps the
retrieval pipeline fully swappable: switch encoders or backends without
touching call sites.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
import numpy as np
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TileMetadata:
    """Provenance record that locates a tile within its source NAC image.

    Stored alongside every embedding so that a vector-store hit can be
    traced back to the exact pixel region it came from.

    Attributes:
        product_id: PDS product identifier of the parent NAC frame
                    (e.g. ``"M102285549RE"``).
        x_offset:   Sample (column) index of the tile's top-left corner
                    within the full-resolution image, in pixels.
        y_offset:   Line (row) index of the tile's top-left corner
                    within the full-resolution image, in pixels.
        width:      Tile width in pixels.
        height:     Tile height in pixels.
        lon:        Longitude.
        lat:        Latitude.
    """
    product_id: str
    x_offset: int
    y_offset: int
    width: int
    height: int
    lon: float = 0.0
    lat: float = 0.0


@dataclass(frozen=True)
class SearchResult:
    """Single ranked result returned by a ``VectorStore`` lookup.

    Attributes:
        score:      Similarity score between the query and the stored
                    embedding (e.g. cosine similarity in ``[-1, 1]``).
                    Higher values indicate a closer match.
        matched_id: Product ID of the known NAC frame whose embedding
                    was nearest to the query (e.g. ``"M102285549RE"``).
        metadata:   ``TileMetadata`` of the matched tile, providing the
                    exact pixel coordinates within ``matched_id``'s image.
    """
    score: float
    matched_id: str
    metadata: TileMetadata


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingModel(Protocol):
    """Structural interface for tile embedding models.

    Any object implementing this protocol can be used as the encoding
    step of the retrieval pipeline. Implementations may wrap a neural
    network (CLIP, DINOv2, …), a classical feature extractor, or any
    other mapping from raw pixel data to a fixed-size vector.

    The ``@runtime_checkable`` decorator allows ``isinstance`` guards
    at the ingestion boundary without importing concrete encoder classes.
    """

    def encode(self, tiles: np.ndarray) -> np.ndarray:
        """Embed a batch of tiles into a matrix of feature vectors.

        Args:
            tiles: Pixel data with shape ``(B, H, W)`` or ``(B, H, W, C)``,
                   dtype ``uint8`` or ``float32``.

        Returns:
            Embedding matrix of shape ``(B, D)`` where *D* is the model's
            output dimensionality.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Structural interface for approximate-nearest-neighbour backends.

    Implementations wrap a concrete ANN library (FAISS, Qdrant, Weaviate,
    …) and translate raw similarity hits into typed ``SearchResult`` lists.
    The batch signature allows the pipeline to amortise round-trip latency
    when querying multiple tiles at once.

    The ``@runtime_checkable`` decorator allows ``isinstance`` guards at
    the retrieval boundary without importing concrete store classes.
    """

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[list[SearchResult]]:
        """Query the store with a batch of embedding vectors.

        Args:
            query_vector: Query matrix of shape ``(B, D)`` — *B* queries,
                          each of dimension *D* (must match the index).
            top_k:        Number of nearest neighbours to return per query.

        Returns:
            A list of *B* result lists, each containing up to ``top_k``
            ``SearchResult`` objects sorted by descending score.
        """
        ...

    def upsert(self, vectors: np.ndarray, metadata: list[TileMetadata]) -> None:
        """
        Insert or update a batch of vectors and their associated metadata.

        If a vector with the same identifier already exists, it will be updated.
        Otherwise, it will be inserted.

        Args:
            vectors:
                Array of shape (B, D), where B is the number of vectors and
                D is the embedding dimension.
            metadata:
                List of length B containing metadata objects associated with
                each vector.

        Raises:
            ValueError:
                If the number of vectors and metadata entries does not match.
        """
        ...