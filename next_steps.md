
# luna — Next Steps (Library Readiness)

Sortiert nach Priorität. Alle Punkte sind **non-breaking** — additiv oder Opt-in.

---

## 1. Metrics API  *(nächste Iteration)*

**Neue Datei:** `luna/metrics.py` — `MetricsReport` frozen dataclass.

### Felder

| Gruppe | Felder |
|---|---|
| Ingest | `ingest_s`, `ingest_tiles_per_s`, `index_compile_s`, `index_size_bytes` |
| Pithos Search | `index_load_s`, `knn_search_s`, `knn_ms_per_query`, `n_queries`, `index_vectors`, `index_tiers` |
| Kandidaten | `n_candidates_raw`, `nms_s`, `n_candidates_nms` |
| ESSA | `essa_s`, `essa_hits_in`, `essa_hits_out`, `essa_refinement_ratio` |
| Total | `total_scan_s`, `total_s` |

### API

```python
# Default — kein Breaking Change
hits = pipeline.scan("M1118880788RC")

# Opt-in
hits, metrics = pipeline.scan("M1118880788RC", metrics=True)
refined, metrics = pipeline.refine(hits, metrics=True)

print(metrics)        # Rich-Tabelle
metrics.to_dict()     # JSON-serialisierbar für Benchmark-DB
```

`scan()` / `refine()` bekommen `metrics: bool = False`. Intern wird der bestehende
`trace`-Dict bei `metrics=True` immer aktiviert und am Ende zu `MetricsReport`
assembliert. `MetricsReport` kommt in `__all__` in `luna/__init__.py`.

**`fun_with_luna.py`:** neues `--metrics` Flag, druckt `MetricsReport` nach dem Run.

---

## 2. Pithos Delta-API exponieren  *(non-breaking, additiv)*

Der Pithos-Singleton hat 8 native Funktionen die noch nicht in `pithos.py` gebunden
sind. Sie bilden zusammen eine **Online-Update-API**: Vektoren können inkrementell
in einen Delta-Buffer eingefügt/gelöscht werden ohne den Hauptindex neu zu
kompilieren, und der Delta-Buffer kann auf Disk gesichert und wiederhergestellt werden.

### Derzeit in `pithos.py` nicht implementiert

| C-Funktion | Bedeutung |
|---|---|
| `vdb_create_delta_buffer(thread, name, capacity)` | Erstellt einen In-Memory-Delta-Buffer der Kapazität `capacity` für den Index `name` |
| `vdb_insert(thread, name, id, vector*)` | Fügt einen float32-Vektor in den Delta-Buffer ein |
| `vdb_delete_from_delta(thread, name, id)` | Löscht eine ID aus dem Delta-Buffer |
| `vdb_delta_size(thread, name)` | Anzahl Einträge im Delta-Buffer |
| `vdb_needs_flush(thread, name)` | Gibt 1 zurück wenn der Delta-Buffer voll ist und geleert werden soll |
| `vdb_search_merged(thread, name, query*, k, ids*, dists*)` | KNN über Haupt-Index + Delta-Buffer gemeinsam |
| `vdb_backup_delta(thread, name, path)` | Schreibt den Delta-Buffer als Datei auf Disk |
| `vdb_restore_delta(thread, name, path, capacity)` | Lädt einen gesicherten Delta-Buffer von Disk |
| `vdb_get_tier_address(thread, name, tier, offset*, size*)` | Gibt Byte-Offset und Größe eines Matryoshka-Tiers zurück (für Low-Level-Inspection) |
| `vdb_transform_and_quantize(thread, name, vectors*, out_ids*)` | Transformiert und quantisiert float32-Vektoren ohne Search (für externes Preprocessing) |

### Neue Methoden in `PithosMIDB`

```python
db = PithosMIDB()

# Delta-Buffer
db.create_delta_buffer("my_index", capacity=1000)
db.insert("my_index", id=7046, vector=np.array([...], dtype=np.float32))
db.delete_from_delta("my_index", id=42)
db.delta_size("my_index")     # → int
db.needs_flush("my_index")    # → bool

# Merged Search (Haupt-Index + Delta)
ids, dists = db.search_merged("my_index", query, k=10)

# Backup / Restore
db.backup_delta("my_index", path="/path/to/delta.bin")
db.restore_delta("my_index", path="/path/to/delta.bin", capacity=1000)

# Low-Level Inspection
offset, size = db.get_tier_address("my_index", tier=0)
quantized_ids = db.transform_and_quantize("my_index", vectors)
```

### Warum das für luna wichtig ist

- **Inkrementelle Ingestion**: Neues NAC-Bild einlaufen lassen → nur Diff in den
  Delta-Buffer schreiben, kein Full-Recompile des Hauptindex
- **Backup**: Delta-Buffer sichern zwischen Pipeline-Runs → kein Datenverlust bei
  Crash nach Ingestion aber vor Kompilierung
- **Tier-Inspection**: Matryoshka-Tier-Layout inspizieren für Debugging und
  Kalibrierung des Cascade-Prunings (`set_energy_budget`)

---

## 3. Exception-Hierarchie  *(non-breaking, additiv)*

**Neue Datei:** `luna/exceptions.py`

```
LunaError                     (base)
  ├── IngestError             (Tile-Embedding fehlgeschlagen)
  ├── IndexError              (Pithos compile/load/search fehlgeschlagen)
  │     └── IndexNotFoundError
  ├── SearchError             (vdb_batch_search / vdb_search_merged fehlgeschlagen)
  ├── DeltaError              (Delta-Buffer Operation fehlgeschlagen)
  │     └── DeltaFullError   (vdb_needs_flush == 1, Flush-Pflicht)
  ├── NACNotFoundError        (IMG nicht auf Disk, PDS-Fetch fehlgeschlagen)
  └── RefinementError         (ESSA inference fehlgeschlagen)
```

Alle `raise RuntimeError(...)` in `pithos.py`, `pithos_store.py`, `pipeline.py`
werden durch spezifische Klassen ersetzt. Code mit `except Exception` bricht nicht.

---

## 4. Öffentliche API bereinigen  *(additiv)*

`luna/__init__.py` exportiert derzeit nur `LunaPipeline` und `CandidateHit`.
Vollständige öffentliche API:

```python
from luna import (
    LunaPipeline,
    CandidateHit,       # Phase-1-Treffer
    RefinedHit,         # Phase-2-Treffer
    MetricsReport,      # Performance-Metrics
    LunaError,          # Base Exception
    LunaConfig,         # Konfigurationsobjekt (→ Punkt 5)
)
```

Dazu: `luna/py.typed` (leere Datei) für mypy/pyright-Support.

---

## 5. `LunaConfig` — Konfigurationsobjekt  *(additiv, opt-in)*

Aktuell kommen alle Hyperparameter als Modul-Globals aus `luna/config.py`.
Das verhindert zwei `LunaPipeline`-Instanzen mit verschiedenen Settings.

```python
cfg = LunaConfig(
    tile_size      = 256,
    stride         = 192,
    search_k       = 200,
    index_dir      = Path("custom/indices"),
    pithos_tiers   = [64, 128, 256, 384],
    energy_budget  = 0.85,   # Cascade-Pruning Tau
)
pipeline = LunaPipeline.from_pretrained("...", config=cfg)
```

`LunaConfig` ist ein `@dataclass` mit Defaults aus den bestehenden Konstanten.
`from_pretrained()` ohne `config=` ist identisch zum Status quo.

---

## 6. Progress Callbacks statt tqdm-Hardcoding  *(additiv, opt-in)*

```python
def on_progress(step: str, current: int, total: int) -> None:
    ...

pipeline.scan("M1118880788RC", on_progress=on_progress)
```

Default `on_progress=None` → tqdm wie bisher. Mit Callback: tqdm deaktiviert,
Callback pro Batch. Für Jupyter-Nutzung und Server-Umgebungen unverzichtbar.

---

## 7. Typing mit `overload`  *(non-breaking, Hygiene)*

`scan()` und `refine()` ändern ihren Rückgabetyp abhängig von `metrics=`.
Das muss für mypy korrekt annotiert sein:

```python
from typing import overload, Literal

@overload
def scan(self, ..., metrics: Literal[False] = ...) -> list[CandidateHit]: ...
@overload
def scan(self, ..., metrics: Literal[True]) -> tuple[list[CandidateHit], MetricsReport]: ...
```

---

## 8. Packaging: native Library einbinden  *(wichtig für pip install)*

`third_party/pithos/libpithos-macos-aarch64.dylib` wird bei `pip install` nicht
mitgeliefert, weil `pyproject.toml` kein `package_data` deklariert.

```toml
[tool.setuptools.package-data]
luna = [
    "../third_party/pithos/*.dylib",
    "../third_party/pithos/*.so",
    "py.typed",
]
```

Platform-Detection in `pithos.py` ist schon vorhanden. `pyproject.toml` bekommt
zusätzlich `markers` für macOS/Linux-spezifische Deps.

---

## 9. Tests ausbauen  *(additiv)*

`luna/tests/test_smoke.py` existiert. Ergänzen:

| Test | Was |
|---|---|
| `test_pithos_singleton` | Zwei `PithosMIDB()` geben dasselbe Objekt |
| `test_pithos_binarize` | `binarize()` → `(N,6) int64` |
| `test_delta_roundtrip` | `create_delta_buffer` → `insert` → `delta_size` → `backup` → `restore` |
| `test_metrics_report` | `MetricsReport.to_dict()` ist JSON-serialisierbar |
| `test_scan_no_metrics` | `scan()` ohne `metrics` → `list` |
| `test_scan_with_metrics` | `scan(..., metrics=True)` → `tuple` |
| `test_exception_types` | `IndexNotFoundError` ist Subclass von `LunaError` |
| `test_config_defaults` | `LunaConfig()` hat identische Werte wie aktuelle `config.py`-Konstanten |

---

## 10. `__repr__` / `_repr_html_` auf Dataclasses  *(trivial, additiv)*

`CandidateHit`, `RefinedHit`, `MetricsReport` bekommen `_repr_html_()` für
saubere Jupyter-Ausgabe (Mini-Tabelle).

---

## Nicht jetzt

- **Async** (`async def scan(...)`) — zu viel Umbau, kein klarer Bedarf
- **Multi-NAC parallelism** — Pithos-Singleton ist Single-Thread; erst wenn GraalVM
  Multi-Isolate-Support landed
- **REST API / CLI** — außerhalb Library-Scope

---

## Empfohlene Reihenfolge

```
1. Metrics API              → sofort nützlich, Grundlage für Benchmarks
2. Pithos Delta-API         → Backup/Restore für robuste Ingestion
3. Exception-Hierarchie     → Library-Qualität, wenig Aufwand
4. Public API / py.typed    → schnell, große Wirkung für Nutzer
5. LunaConfig               → ermöglicht sauberere Tests + Jupyter-Use
6. Progress Callbacks       → Jupyter/Server-Nutzung
7. Typing overloads         → Hygiene
8. Packaging                → wenn externe Nutzer kommen
9. Tests                    → parallel zu allem anderen
```
