# `luna.screening` Reference

`luna.screening` implements the lock-free streaming ring buffer engine, memory-mapped I/O stripe transformers, and screening protocols for high-throughput tile ingestion.

---

## Streaming Engine & Ring Buffer (`luna.screening.engine`)

::: luna.screening.engine
    options:
      members:
        - DisruptorEngine
        - MappedStripe
        - push_stripe_to_ring
        - NACTransformer

---

## Data Ingestion & Tile Pipeline (`luna.screening.candidate_gen`)

::: luna.screening.candidate_gen
    options:
      members:
        - DataIngestor
        - ScreenerEngine

---

## Screening Protocols & Metadata (`luna.screening.protocols`)

::: luna.screening.protocols
    options:
      members:
        - TileMetadata
        - SearchResult
        - EmbeddingModel
        - VectorStore


