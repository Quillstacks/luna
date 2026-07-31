"""
luna.screening.pithos
~~~~~~~~~~~~~~~~~~~~~
Python singleton wrapper for the Pithos AOT-compiled Model-Isomorphic
Database (MIDB) native library.

The native shared library is bundled under:
    third_party/pithos/libpithos-macos-aarch64.dylib   (macOS / Apple Silicon)
    third_party/pithos/libpithos-linux-x86_64.so        (Linux / DGX Spark)

Key difference from the old PithosEngine:
  - All public methods accept **raw float32 vectors** (N, 384).
    The native library handles quantization internally.
  - The class is a **Singleton** — one GraalVM isolate per process.
  - ``binarize()`` is kept as a static helper for any code that still
    needs the PolarQuant-Hadamard transform externally.
"""
from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

from luna.exceptions import IndexError, SearchError, DeltaError, DeltaFullError

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate bundled native library
# ---------------------------------------------------------------------------
_THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party" / "pithos"


def _find_lib(use_cuda: bool = False) -> Path:
    system = platform.system()
    
    if system == "Linux":
        if use_cuda:
            candidates = [
                _THIRD_PARTY / "libpithos-linux-cuda-aarch64.so",
                _THIRD_PARTY / "libpithos-linux-x86_64-cuda.so",
                _THIRD_PARTY / "libpithos-cuda.so",
                _THIRD_PARTY / "libpithos-linux-aarch64.so",
                _THIRD_PARTY / "libpithos-linux-x86_64.so",
                _THIRD_PARTY / "libpithos.so",
            ]
        else:
            candidates = [
                _THIRD_PARTY / "libpithos-linux-aarch64.so",
                _THIRD_PARTY / "libpithos-linux-cuda-aarch64.so",
                _THIRD_PARTY / "libpithos-linux-x86_64.so",
                _THIRD_PARTY / "libpithos.so",
                _THIRD_PARTY / "libpithos-cuda.so",
                _THIRD_PARTY / "libpithos-linux-x86_64-cuda.so",
            ]
    elif system == "Darwin":
        candidates = [
            _THIRD_PARTY / "libpithos-macos-aarch64.dylib",
            _THIRD_PARTY / "libpithos.dylib",
        ]
    else:
        candidates = [
            _THIRD_PARTY / "libpithos-linux-x86_64.so",
            _THIRD_PARTY / "libpithos.so",
        ]
    
    for p in candidates:
        if p.exists():
            return p
    
    raise FileNotFoundError(
        f"Pithos native library not found. Expected one of:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


# ---------------------------------------------------------------------------
# PolarQuant-Hadamard binarization  (kept for external use / legacy compat)
# ---------------------------------------------------------------------------
_DINO_DIM = 384
_rng = np.random.default_rng(42)
_D_SIGN = _rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=_DINO_DIM)


def _build_hadamard_384() -> np.ndarray:
    """Build the 384×384 Kronecker-Hadamard matrix H12 ⊗ H32."""

    def silvester(n: int) -> np.ndarray:
        if n == 1:
            return np.array([[1]], dtype=np.float32)
        h = silvester(n // 2)
        return np.block([[h, h], [h, -h]])

    H32 = silvester(32)
    H12 = np.array([
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [ 1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1],
        [ 1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1],
        [ 1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1],
        [ 1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1],
        [ 1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1],
        [ 1,  1, -1, -1, -1,  1, -1, -1,  1, -1,  1,  1],
        [ 1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,  1],
        [ 1,  1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1],
        [ 1, -1,  1,  1,  1, -1, -1, -1,  1, -1, -1,  1],
        [ 1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1, -1],
    ], dtype=np.float32)
    return np.kron(H12, H32)   # (384, 384) float32


_H384: np.ndarray = _build_hadamard_384()
_SCALE: float = float(np.sqrt(_DINO_DIM))

# ---------------------------------------------------------------------------
# GraalVM isolate handle types (opaque)
# ---------------------------------------------------------------------------


class _GraalIsolate(ctypes.Structure):
    pass


class _GraalIsolateThread(ctypes.Structure):
    pass


# ---------------------------------------------------------------------------
# Stderr suppressor — silences GraalVM startup chatter
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _suppress_stderr():
    sys.stderr.flush()
    err_fd = sys.stderr.fileno()
    saved = os.dup(err_fd)
    null_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(null_fd, err_fd)
    os.close(null_fd)
    try:
        yield
    finally:
        os.dup2(saved, err_fd)
        os.close(saved)


# ---------------------------------------------------------------------------
# Matryoshka tier layout for DINOv3 384-dim embeddings
# ---------------------------------------------------------------------------
#  Pithos cascades through these prefix lengths for early-exit pruning.
MOON_TIERS = np.array([64, 128, 256, 384], dtype=np.int32)
MOON_ID     = 1           # Planet registry byte for the Moon
MOON_RADIUS = 1_737_400   # Mean Moon radius in metres


# ---------------------------------------------------------------------------
# PithosMIDB — Singleton FFI wrapper
# ---------------------------------------------------------------------------

class PithosMIDB:
    """
    AOT-compiled Pithos Model-Isomorphic Database (MIDB) native library
    interface — Singleton.

    One GraalVM isolate is created per process.  All methods accept raw
    **float32** vectors; quantization is handled natively.

    Attributes
    ----------
    lib    : ctypes.CDLL
    isolate: POINTER(_GraalIsolate)
    thread : POINTER(_GraalIsolateThread)
    """

    _instance: "PithosMIDB | None" = None

    def __new__(cls, lib_path: str | Path | None = None, use_cuda: bool = False) -> "PithosMIDB":
        if cls._instance is None:
            instance = super().__new__(cls)
            resolved = Path(lib_path) if lib_path is not None else _find_lib(use_cuda=use_cuda)
            instance._init_ffi(resolved, use_cuda=use_cuda)
            cls._instance = instance
        else:
            if use_cuda and not cls._instance.use_cuda:
                cls._instance._enable_cuda_dynamically()
        return cls._instance

    def __init__(self, lib_path: str | Path | None = None, use_cuda: bool = False) -> None:
        """No-op: prevents re-initialization of the singleton state."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_ffi(self, lib_path: Path, use_cuda: bool = False) -> None:
        """Load the shared library and register all C-API signatures."""
        log.info("Loading Pithos native library from %s …", lib_path)
        self.lib     = ctypes.CDLL(str(lib_path))
        self.isolate = ctypes.POINTER(_GraalIsolate)()
        self.thread  = ctypes.POINTER(_GraalIsolateThread)()

        self._configure_signatures()

        with _suppress_stderr():
            status = self.lib.graal_create_isolate(
                None,
                ctypes.byref(self.isolate),
                ctypes.byref(self.thread),
            )
        if status != 0:
            raise IndexError(f"graal_create_isolate failed (code {status}).")

        with _suppress_stderr():
            status = self.lib.vdb_init(self.thread)
        if status != 0:
            raise IndexError(f"vdb_init failed (code {status}).")

        self.use_cuda = False
        if use_cuda and getattr(self, "_has_cuda_api", False):
            with _suppress_stderr():
                cuda_status = self.lib.vdb_cuda_init(self.thread, 0)
                if cuda_status == 0:
                    self.use_cuda = True
                    log.info("Pithos CUDA acceleration enabled successfully (device 0).")
                else:
                    log.warning("vdb_cuda_init failed (code %d). Falling back to CPU.", cuda_status)

        log.info("Pithos isolate initialized successfully.")

    def _enable_cuda_dynamically(self) -> None:
        """Attempt to enable CUDA runtime dynamically on an active isolate thread."""
        if getattr(self, "_has_cuda_api", False) and not self.use_cuda:
            with _suppress_stderr():
                cuda_status = self.lib.vdb_cuda_init(self.thread, 0)
                if cuda_status == 0:
                    self.use_cuda = True
                    log.info("Pithos CUDA acceleration enabled dynamically (device 0).")
                else:
                    log.warning("Dynamic vdb_cuda_init failed (code %d). Remaining on CPU.", cuda_status)

    def _configure_signatures(self) -> None:
        """Register ctypes argtypes / restype for every exported symbol."""
        lib = self.lib
        P  = ctypes.c_void_p                                   # IsolateThread*
        PP_Iso = ctypes.POINTER(ctypes.POINTER(_GraalIsolate))
        PP_Thr = ctypes.POINTER(ctypes.POINTER(_GraalIsolateThread))

        lib.graal_create_isolate.argtypes = [ctypes.c_void_p, PP_Iso, PP_Thr]
        lib.graal_create_isolate.restype  = ctypes.c_int

        lib.graal_tear_down_isolate.argtypes = [P]
        lib.graal_tear_down_isolate.restype  = ctypes.c_int

        lib.vdb_init.argtypes = [P]
        lib.vdb_init.restype  = ctypes.c_int

        # vdb_compile_index_file(thread, path, planet_id, planet_radius,
        #   dimension, tiers*, n_tiers, ids*, vectors*, n_records, q_mode)
        lib.vdb_compile_index_file.argtypes = [
            P,                  # thread
            ctypes.c_char_p,    # path
            ctypes.c_byte,      # planet_id
            ctypes.c_longlong,  # planet_radius
            ctypes.c_int,       # dimension
            ctypes.c_void_p,    # tiers*  (int32[])
            ctypes.c_int,       # n_tiers
            ctypes.c_void_p,    # ids*    (int64[])
            ctypes.c_void_p,    # vectors* (float32[])
            ctypes.c_int,       # n_records
            ctypes.c_int,       # q_mode
        ]
        lib.vdb_compile_index_file.restype = ctypes.c_int

        # vdb_compile_index_file_ext(thread, path, planet_id, planet_radius,
        #   dimension, tiers*, n_tiers, ids*, vectors*, n_records, q_mode, write_fp16)
        lib.vdb_compile_index_file_ext.argtypes = [
            P,                  # thread
            ctypes.c_char_p,    # path
            ctypes.c_byte,      # planet_id
            ctypes.c_longlong,  # planet_radius
            ctypes.c_int,       # dimension
            ctypes.c_void_p,    # tiers*  (int32[])
            ctypes.c_int,       # n_tiers
            ctypes.c_void_p,    # ids*    (int64[])
            ctypes.c_void_p,    # vectors* (float32[])
            ctypes.c_int,       # n_records
            ctypes.c_int,       # q_mode
            ctypes.c_int,       # write_fp16
        ]
        lib.vdb_compile_index_file_ext.restype = ctypes.c_int

        lib.vdb_load_index.argtypes = [P, ctypes.c_char_p, ctypes.c_char_p]
        lib.vdb_load_index.restype  = ctypes.c_int

        lib.vdb_load_index_with_weights.argtypes = [
            P, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int
        ]
        lib.vdb_load_index_with_weights.restype = ctypes.c_int

        # vdb_batch_search(thread, name, queries_f32*, n_queries, k, ids*, dists*)
        lib.vdb_batch_search.argtypes = [
            P, ctypes.c_char_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.vdb_batch_search.restype = ctypes.c_int

        lib.vdb_query_planetary_grid.argtypes = [
            P, ctypes.c_char_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_void_p,
        ]
        lib.vdb_query_planetary_grid.restype = ctypes.c_longlong

        lib.vdb_get_info.argtypes = [
            P, ctypes.c_char_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.vdb_get_info.restype = ctypes.c_int

        lib.vdb_drop_index.argtypes = [P, ctypes.c_char_p]
        lib.vdb_drop_index.restype  = ctypes.c_int

        lib.vdb_size.argtypes = [P, ctypes.c_char_p]
        lib.vdb_size.restype  = ctypes.c_longlong

        lib.vdb_set_chunk_size.argtypes = [P, ctypes.c_char_p, ctypes.c_longlong]
        lib.vdb_set_chunk_size.restype  = ctypes.c_int

        lib.vdb_set_energy_budget.argtypes = [P, ctypes.c_char_p, ctypes.c_double]
        lib.vdb_set_energy_budget.restype  = ctypes.c_int

        lib.vdb_close.argtypes = [P]
        lib.vdb_close.restype  = ctypes.c_int

        # Delta Buffer API
        lib.vdb_create_delta_buffer.argtypes = [P, ctypes.c_char_p, ctypes.c_int]
        lib.vdb_create_delta_buffer.restype  = ctypes.c_int

        lib.vdb_insert.argtypes = [
            P, ctypes.c_char_p,
            ctypes.c_longlong,      # id (int64)
            ctypes.c_void_p,        # vector (float32*)
        ]
        lib.vdb_insert.restype  = ctypes.c_int

        lib.vdb_delete_from_delta.argtypes = [P, ctypes.c_char_p, ctypes.c_longlong]
        lib.vdb_delete_from_delta.restype  = ctypes.c_int

        lib.vdb_delta_size.argtypes = [P, ctypes.c_char_p]
        lib.vdb_delta_size.restype  = ctypes.c_longlong

        lib.vdb_needs_flush.argtypes = [P, ctypes.c_char_p]
        lib.vdb_needs_flush.restype  = ctypes.c_int

        lib.vdb_search_merged.argtypes = [
            P, ctypes.c_char_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.vdb_search_merged.restype = ctypes.c_int

        lib.vdb_backup_delta.argtypes = [P, ctypes.c_char_p, ctypes.c_char_p]
        lib.vdb_backup_delta.restype  = ctypes.c_int

        lib.vdb_restore_delta.argtypes = [
            P, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int
        ]
        lib.vdb_restore_delta.restype  = ctypes.c_int

        lib.vdb_get_tier_address.argtypes = [
            P, ctypes.c_char_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.vdb_get_tier_address.restype  = ctypes.c_int

        lib.vdb_transform_and_quantize.argtypes = [
            P, ctypes.c_char_p,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.vdb_transform_and_quantize.restype = ctypes.c_int

        # CUDA API registrations (optional fallback)
        try:
            lib.vdb_cuda_init.argtypes = [P, ctypes.c_int]
            lib.vdb_cuda_init.restype  = ctypes.c_int

            lib.vdb_cuda_shutdown.argtypes = [P]
            lib.vdb_cuda_shutdown.restype  = ctypes.c_int

            lib.vdb_cuda_is_available.argtypes = [P]
            lib.vdb_cuda_is_available.restype  = ctypes.c_int

            lib.vdb_cuda_batch_search.argtypes = [
                P, ctypes.c_char_p,
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p,
            ]
            lib.vdb_cuda_batch_search.restype = ctypes.c_int

            lib.vdb_cuda_query_planetary_grid.argtypes = [
                P, ctypes.c_char_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_void_p,
            ]
            lib.vdb_cuda_query_planetary_grid.restype = ctypes.c_longlong
            
            self._has_cuda_api = True
        except AttributeError:
            log.warning("CUDA API functions not found in libpithos.")
            self._has_cuda_api = False

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def binarize(embeddings: np.ndarray) -> np.ndarray:
        """
        Convert float32 DINOv3 embeddings → (N, 6) int64 binary vectors
        using PolarQuant-Hadamard preconditioning.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, 384), dtype float32

        Returns
        -------
        np.ndarray, shape (N, 6), dtype int64
        """
        emb = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        emb = emb / norms
        emb = emb * _D_SIGN
        emb = (emb @ _H384.T) / _SCALE
        bits   = (emb >= 0).astype(np.uint8)
        packed = np.packbits(bits, axis=1)
        packed = np.ascontiguousarray(packed)
        return packed.view(np.int64)   # (N, 6) int64

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(
        self,
        file_path: str | Path,
        ids: np.ndarray,
        vectors: np.ndarray,
        planet_id: int     = MOON_ID,
        planet_radius: int = MOON_RADIUS,
        dimension: int     = _DINO_DIM,
        tiers: np.ndarray  = MOON_TIERS,
        q_mode: int        = 0,
        use_fp16: bool     = False,
    ) -> None:
        """
        Compile raw float32 embeddings into an off-heap Pithos index file.

        Parameters
        ----------
        file_path     : Destination ``.bin`` base path (created/overwritten).
        ids           : shape (N,) int64 — sequential record IDs.
        vectors       : shape (N, dimension) float32 — raw embeddings.
        planet_id     : Planet registry byte.  Default: ``MOON_ID`` = 1.
        planet_radius : Mean radius in metres.  Default: ``MOON_RADIUS``.
        dimension     : Vector dimension.  Default: 384.
        tiers         : Matryoshka cascade breakpoints.  Default: [64,128,256,384].
        q_mode        : 0 = 1-bit sign (default), 1 = 2-bit ternary, 2 = float32.
        use_fp16      : Use FP16 precision for index. Default: False.
        """
        n = len(ids)
        if vectors.shape[0] != n:
            raise ValueError(f"ids/vectors length mismatch: {n} vs {vectors.shape[0]}")

        ids_c     = np.ascontiguousarray(ids,     dtype=np.int64)
        vectors_c = np.ascontiguousarray(vectors, dtype=np.float32)
        tiers_c   = np.ascontiguousarray(tiers,   dtype=np.int32)

        with _suppress_stderr():
            status = self.lib.vdb_compile_index_file_ext(
                self.thread,
                str(file_path).encode(),
                ctypes.c_byte(planet_id),
                ctypes.c_longlong(planet_radius),
                ctypes.c_int(dimension),
                tiers_c.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(len(tiers_c)),
                ids_c.ctypes.data_as(ctypes.c_void_p),
                vectors_c.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(n),
                ctypes.c_int(q_mode),
                ctypes.c_int(1 if use_fp16 else 0),
            )
        if status != 0:
            raise IndexError(f"vdb_compile_index_file_ext failed (code {status}).")

    def load_index(
        self,
        index_name: str,
        file_path: str | Path,
        weights: np.ndarray | None = None,
        lora_dim: int = 0,
    ) -> None:
        """
        Memory-map a compiled Pithos index file off-heap.

        Parameters
        ----------
        index_name : Identifier tag assigned to this index.
        file_path  : Base file path of the compiled index.
        weights    : Optional (dim, lora_dim) float32 projection weights.
        lora_dim   : Rank of the weights matrix.
        """
        name_b = index_name.encode()
        path_b = str(file_path).encode()
        if weights is not None:
            w_c = np.ascontiguousarray(weights, dtype=np.float32)
            with _suppress_stderr():
                status = self.lib.vdb_load_index_with_weights(
                    self.thread, name_b, path_b,
                    w_c.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(lora_dim),
                )
        else:
            with _suppress_stderr():
                status = self.lib.vdb_load_index(self.thread, name_b, path_b)
        if status != 0:
            raise IndexError(f"vdb_load_index failed (code {status}).")

    def drop_index(self, index_name: str) -> None:
        """Unmap and release an off-heap index."""
        with _suppress_stderr():
            status = self.lib.vdb_drop_index(self.thread, index_name.encode())
        if status != 0:
            raise IndexError(f"vdb_drop_index failed (code {status}).")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def batch_search(
        self,
        index_name: str,
        queries: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parallel KNN search over the named index.

        Parameters
        ----------
        queries : shape (N, 384) float32 — raw query embeddings.
        k       : Number of nearest neighbours per query.

        Returns
        -------
        ids       : shape (N, k) int64
        distances : shape (N, k) int32  — Hamming distances (lower = better).
        """
        n         = queries.shape[0]
        queries_c = np.ascontiguousarray(queries, dtype=np.float32)
        out_ids   = np.empty(n * k, dtype=np.int64)
        out_dists = np.empty(n * k, dtype=np.int32)

        with _suppress_stderr():
            if getattr(self, "use_cuda", False) and getattr(self, "_has_cuda_api", False):
                status = self.lib.vdb_cuda_batch_search(
                    self.thread,
                    index_name.encode(),
                    queries_c.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(n),
                    ctypes.c_int(k),
                    out_ids.ctypes.data_as(ctypes.c_void_p),
                    out_dists.ctypes.data_as(ctypes.c_void_p),
                )
            else:
                status = self.lib.vdb_batch_search(
                    self.thread,
                    index_name.encode(),
                    queries_c.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(n),
                    ctypes.c_int(k),
                    out_ids.ctypes.data_as(ctypes.c_void_p),
                    out_dists.ctypes.data_as(ctypes.c_void_p),
                )
        if status != 0:
            raise SearchError(f"vdb_batch_search failed (code {status}).")

        return out_ids.reshape(n, k), out_dists.reshape(n, k)

    def query_planetary_grid(
        self,
        index_name: str,
        queries: np.ndarray,
        families: np.ndarray,
        thresholds: np.ndarray,
        voting_mask: np.ndarray,
    ) -> int:
        """
        Multi-Family Resonant Voting scan over the named index.

        Parameters
        ----------
        queries     : shape (N, 384) float32.
        families    : shape (N,) int32 — family ID (0–7) per query.
        thresholds  : shape (N,) int32 — Hamming cutoff per query.
        voting_mask : shape (total_records,) uint8 — written in-place.

        Returns
        -------
        int — number of resonant tiles (voting_mask != 0).
        """
        queries_c = np.ascontiguousarray(queries, dtype=np.float32)
        with _suppress_stderr():
            if getattr(self, "use_cuda", False) and getattr(self, "_has_cuda_api", False):
                return int(self.lib.vdb_cuda_query_planetary_grid(
                    self.thread,
                    index_name.encode(),
                    queries_c.ctypes.data_as(ctypes.c_void_p),
                    families.ctypes.data_as(ctypes.c_void_p),
                    thresholds.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(queries.shape[0]),
                    voting_mask.ctypes.data_as(ctypes.c_void_p),
                ))
            else:
                return int(self.lib.vdb_query_planetary_grid(
                    self.thread,
                    index_name.encode(),
                    queries_c.ctypes.data_as(ctypes.c_void_p),
                    families.ctypes.data_as(ctypes.c_void_p),
                    thresholds.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(queries.shape[0]),
                    voting_mask.ctypes.data_as(ctypes.c_void_p),
                ))

    # ------------------------------------------------------------------
    # Delta Buffer API
    # ------------------------------------------------------------------

    def create_delta_buffer(self, index_name: str, capacity: int) -> None:
        """Create an in-memory delta buffer for the named index."""
        with _suppress_stderr():
            status = self.lib.vdb_create_delta_buffer(
                self.thread, index_name.encode(), ctypes.c_int(capacity)
            )
        if status != 0:
            raise DeltaError(f"vdb_create_delta_buffer failed (code {status}).")

    def insert(self, index_name: str, id: int, vector: np.ndarray) -> None:
        """Insert a float32 vector into the delta buffer for the named index."""
        vec_c = np.ascontiguousarray(vector, dtype=np.float32)
        with _suppress_stderr():
            status = self.lib.vdb_insert(
                self.thread,
                index_name.encode(),
                ctypes.c_longlong(id),
                vec_c.ctypes.data_as(ctypes.c_void_p),
            )
        if status != 0:
            raise DeltaError(f"vdb_insert failed (code {status}).")

    def delete_from_delta(self, index_name: str, id: int) -> None:
        """Delete an entry from the delta buffer for the named index."""
        with _suppress_stderr():
            status = self.lib.vdb_delete_from_delta(
                self.thread, index_name.encode(), ctypes.c_longlong(id)
            )
        if status != 0:
            raise DeltaError(f"vdb_delete_from_delta failed (code {status}).")

    def delta_size(self, index_name: str) -> int:
        """Return the number of entries in the delta buffer for the named index."""
        with _suppress_stderr():
            return int(self.lib.vdb_delta_size(self.thread, index_name.encode()))

    def needs_flush(self, index_name: str) -> bool:
        """Check if the delta buffer for the named index needs to be flushed."""
        with _suppress_stderr():
            return bool(self.lib.vdb_needs_flush(self.thread, index_name.encode()))

    def search_merged(
        self, index_name: str, queries: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """KNN search over main index + delta buffer together."""
        n = queries.shape[0]
        queries_c = np.ascontiguousarray(queries, dtype=np.float32)
        out_ids = np.empty(n * k, dtype=np.int64)
        out_dists = np.empty(n * k, dtype=np.int32)

        with _suppress_stderr():
            status = self.lib.vdb_search_merged(
                self.thread,
                index_name.encode(),
                queries_c.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(n),
                ctypes.c_int(k),
                out_ids.ctypes.data_as(ctypes.c_void_p),
                out_dists.ctypes.data_as(ctypes.c_void_p),
            )
        if status != 0:
            raise SearchError(f"vdb_search_merged failed (code {status}).")

        return out_ids.reshape(n, k), out_dists.reshape(n, k)

    def backup_delta(self, index_name: str, path: str | Path) -> None:
        """Backup the delta buffer to disk."""
        path_b = str(path).encode()
        with _suppress_stderr():
            status = self.lib.vdb_backup_delta(
                self.thread, index_name.encode(), path_b
            )
        if status != 0:
            raise DeltaError(f"vdb_backup_delta failed (code {status}).")

    def restore_delta(
        self, index_name: str, path: str | Path, capacity: int
    ) -> None:
        """Restore a delta buffer from disk."""
        path_b = str(path).encode()
        with _suppress_stderr():
            status = self.lib.vdb_restore_delta(
                self.thread, index_name.encode(), path_b, ctypes.c_int(capacity)
            )
        if status != 0:
            raise DeltaError(f"vdb_restore_delta failed (code {status}).")

    def get_tier_address(
        self, index_name: str, tier: int
    ) -> Tuple[int, int]:
        """Return byte offset and size of a Matryoshka tier."""
        offset = ctypes.c_longlong(0)
        size = ctypes.c_longlong(0)
        with _suppress_stderr():
            status = self.lib.vdb_get_tier_address(
                self.thread,
                index_name.encode(),
                ctypes.c_int(tier),
                ctypes.byref(offset),
                ctypes.byref(size),
            )
        if status != 0:
            raise DeltaError(f"vdb_get_tier_address failed (code {status}).")
        return int(offset.value), int(size.value)

    def transform_and_quantize(
        self, index_name: str, vectors: np.ndarray
    ) -> np.ndarray:
        """Transform and quantize float32 vectors without search."""
        n = vectors.shape[0]
        vectors_c = np.ascontiguousarray(vectors, dtype=np.float32)
        out_ids = np.empty(n, dtype=np.int64)

        with _suppress_stderr():
            status = self.lib.vdb_transform_and_quantize(
                self.thread,
                index_name.encode(),
                vectors_c.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(n),
                out_ids.ctypes.data_as(ctypes.c_void_p),
            )
        if status != 0:
            raise DeltaError(f"vdb_transform_and_quantize failed (code {status}).")
        return out_ids

    # ------------------------------------------------------------------
    # Metadata & tuning
    # ------------------------------------------------------------------

    def size(self, index_name: str) -> int:
        """Return the number of vectors in the named index."""
        with _suppress_stderr():
            return int(self.lib.vdb_size(self.thread, index_name.encode()))

    def set_chunk_size(self, index_name: str, chunk_size: int) -> None:
        """Tune the L1-cache chunk size for the parallel scan."""
        with _suppress_stderr():
            self.lib.vdb_set_chunk_size(
                self.thread, index_name.encode(), ctypes.c_longlong(chunk_size)
            )

    def set_energy_budget(self, index_name: str, tau: float) -> None:
        """Set the spectral energy threshold (tau) for cascade pruning."""
        with _suppress_stderr():
            self.lib.vdb_set_energy_budget(
                self.thread, index_name.encode(), ctypes.c_double(tau)
            )

    def get_info(self, index_name: str) -> dict:
        """Return layout metadata for the named index."""
        name_b     = index_name.encode()
        dim        = ctypes.c_int(0)
        size       = ctypes.c_longlong(0)
        planet_id  = ctypes.c_byte(0)
        radius     = ctypes.c_longlong(0)
        tiers_cnt  = ctypes.c_int(0)
        with _suppress_stderr():
            status = self.lib.vdb_get_info(
                self.thread, name_b,
                ctypes.byref(dim), ctypes.byref(size),
                ctypes.byref(planet_id), ctypes.byref(radius),
                ctypes.byref(tiers_cnt),
            )
        if status != 0:
            raise IndexError(f"vdb_get_info failed (code {status}).")
        return {
            "dimension":     dim.value,
            "size":          size.value,
            "planet_id":     planet_id.value,
            "planet_radius": radius.value,
            "tiers_count":   tiers_cnt.value,
        }

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Close all open indexes and clean up the isolate thread.

        Resets the singleton so a subsequent instantiation creates a fresh
        GraalVM context.  ``graal_tear_down_isolate`` is intentionally
        skipped to avoid macOS safepoint spin-wait hangs on exit.
        """
        if self.thread:
            with _suppress_stderr():
                self.lib.vdb_close(self.thread)
            self.thread  = None
            self.isolate = None
        PithosMIDB._instance = None
