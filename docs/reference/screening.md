# `luna.screening` Reference

`luna.screening` defines the vector-search screening protocols, tile ingestion pipelines, and the native Pithos SIMD index interface.

---

## Screening Protocols & Metadata (`luna.screening.protocols`)

::: luna.screening.protocols
    options:
      members:
        - TileMetadata
        - SearchResult
        - EmbeddingModel
        - VectorStore

---

## Pithos Native SIMD Engine (`luna.screening.pithos`)

::: luna.screening.pithos
    options:
      members:
        - PithosMIDB
        - binarize

---

## Data Ingestion & Tile Pipeline (`luna.screening.candidate_gen`)

::: luna.screening.candidate_gen
    options:
      members:
        - DataIngestor

