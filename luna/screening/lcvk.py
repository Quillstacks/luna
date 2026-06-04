"""
luna.screening.lcvk
~~~~~~~~~~~~~~~~~~~
Python wrapper for the LCVK AOT-compiled native vector kernel.

The native shared libraries are bundled under:
    third_party/lcvk/lunar_core.dylib   (macOS / Apple Silicon)
    third_party/lcvk/liblunar_core.so   (Linux / DGX Spark)

No dependency on the lcvk source repository is required at runtime.
"""
from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Resolve bundled native library path
# ---------------------------------------------------------------------------
_THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party" / "lcvk"

def _find_lib() -> Path:
    system = platform.system()
    candidates = (
        [_THIRD_PARTY / "lunar_core.dylib", _THIRD_PARTY / "liblunar_core.so"]
        if system == "Darwin"
        else [_THIRD_PARTY / "liblunar_core.so", _THIRD_PARTY / "lunar_core.dylib"]
    )
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"LCVK native library not found. Expected one of:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


# ---------------------------------------------------------------------------
# Hadamard preconditioning for binarization (PolarQuant-H384)
#
# DINOv3 embeddings have correlated dimensions — raw sign-binarization
# produces heavily skewed bit distributions that collapse the Hamming
# separation between classes. Applying a Kronecker-Hadamard rotation
# (H12 ⊗ H32) followed by diagonal sign preconditioning decorrelates the
# bit planes and restores the expected ~19-bit Hamming gap.
# ---------------------------------------------------------------------------
_DINO_DIM = 384

# Isolated RNG — does not affect the caller's global numpy random state.
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

    # Paley-type H12 — all rows mutually orthogonal, entries ±1
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
# ctypes struct stubs (opaque GraalVM handles)
# ---------------------------------------------------------------------------
class _GraalIsolate(ctypes.Structure):
    pass

class _GraalIsolateThread(ctypes.Structure):
    pass


# ---------------------------------------------------------------------------
# LcvkEngine
# ---------------------------------------------------------------------------
class LcvkEngine:
    """
    Python interface to the LCVK AOT-compiled Hamming-distance vector kernel.

    Usage (search)::

        from luna.screening.lcvk import LcvkEngine

        with LcvkEngine() as engine:
            engine.load_index("lunar", "/path/to/lunar.bin")
            queries = LcvkEngine.binarize(float32_embeddings)
            ids, distances = engine.batch_search("lunar", queries, k=100)

    Usage (index build)::

        with LcvkEngine() as engine:
            ids = np.arange(n, dtype=np.int64)
            vecs = LcvkEngine.binarize(embeddings)
            engine.build_index("/path/to/out.bin", planet_id=1,
                               planet_radius=1_737_400, ids=ids, vectors=vecs)
    """

    #: Moon mean radius in metres — passed to vdb_compile_index_file.
    MOON_RADIUS: int = 1_737_400
    #: Planet-ID byte for the Moon in the LCVK planet registry.
    MOON_ID: int = 1

    def __init__(self, lib_path: str | os.PathLike | None = None) -> None:
        path = Path(lib_path) if lib_path is not None else _find_lib()
        if not path.exists():
            raise FileNotFoundError(f"LCVK native library not found at: {path}")

        self.lib = ctypes.CDLL(str(path))
        self._isolate = ctypes.POINTER(_GraalIsolate)()
        self._thread  = ctypes.POINTER(_GraalIsolateThread)()

        self._configure_signatures()

        status = self.lib.graal_create_isolate(
            None, ctypes.byref(self._isolate), ctypes.byref(self._thread)
        )
        if status != 0:
            raise RuntimeError("Failed to allocate GraalVM isolate thread.")

        status = self.lib.vdb_init(self._thread)
        if status != 0:
            raise RuntimeError("Failed to initialize LCVK DB engine.")

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def binarize(embeddings: np.ndarray) -> np.ndarray:
        """
        Convert float32 DINOv3 embeddings → (N, 6) int64 binary vectors.

        Applies PolarQuant-Hadamard preconditioning before sign-binarization
        to decorrelate bit planes and preserve the inter-class Hamming gap.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, 384), dtype float32
            Raw or L2-normalised DINOv3 output vectors.

        Returns
        -------
        np.ndarray, shape (N, 6), dtype int64
            Packed binary vectors ready for LCVK registers.
        """
        emb = np.asarray(embeddings, dtype=np.float32)

        # 1. L2 normalise (guard against zero-norm rows)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        emb = emb / norms

        # 2. Diagonal sign preconditioning
        emb = emb * _D_SIGN

        # 3. Kronecker-Hadamard rotation — decorrelates bit planes
        emb = (emb @ _H384.T) / _SCALE  # (N, 384)

        # 4. Sign binarization → uint8 bits (0 / 1)
        bits = (emb >= 0).astype(np.uint8)  # (N, 384)

        # 5. Pack 8 bits → 1 byte, then view 8 bytes → 1 int64
        #    np.packbits default order = big-endian, consistent at query time.
        packed = np.packbits(bits, axis=1)           # (N, 48) uint8
        packed = np.ascontiguousarray(packed)
        return packed.view(np.int64)                 # (N, 6) int64

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(
        self,
        file_path: str | os.PathLike,
        planet_id: int,
        planet_radius: int,
        ids: np.ndarray,
        vectors: np.ndarray,
    ) -> None:
        """
        Compile a PLAN binary index file from binary vectors.

        Maps directly to ``vdb_compile_index_file`` in CApi.java:

            compileIndexFile(thread, path, planetId, planetRadius, ids, vectors, numRecords)

        Parameters
        ----------
        file_path : path-like
            Destination ``.bin`` file (created / overwritten).
        planet_id : int
            Planet registry byte (use ``LcvkEngine.MOON_ID = 1`` for the Moon).
        planet_radius : int
            Mean planetary radius in metres (use ``LcvkEngine.MOON_RADIUS``).
        ids : np.ndarray, shape (N,) int64
            Record IDs corresponding to each vector row.
        vectors : np.ndarray, shape (N, 6) int64
            Packed binary vectors produced by ``LcvkEngine.binarize()``.
        """
        n = len(ids)
        if vectors.shape != (n, 6):
            raise ValueError(
                f"vectors must be shape (N, 6), got {vectors.shape} for N={n}"
            )

        ids_c     = np.ascontiguousarray(ids,     dtype=np.int64)
        vectors_c = np.ascontiguousarray(vectors, dtype=np.int64)

        status = self.lib.vdb_compile_index_file(
            self._thread,
            str(file_path).encode(),
            ctypes.c_byte(planet_id),
            ctypes.c_longlong(planet_radius),
            ids_c.ctypes.data_as(ctypes.c_void_p),
            vectors_c.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(n),
        )
        if status != 0:
            raise RuntimeError(f"vdb_compile_index_file failed (code {status}).")

    def load_index(self, index_name: str, file_path: str | os.PathLike) -> None:
        """Memory-map a PLAN binary index file off-heap."""
        status = self.lib.vdb_load_index(
            self._thread,
            index_name.encode(),
            str(file_path).encode(),
        )
        if status != 0:
            raise RuntimeError(f"vdb_load_index failed (code {status}).")

    def drop_index(self, index_name: str) -> None:
        """Unmap and release an off-heap index."""
        status = self.lib.vdb_drop_index(self._thread, index_name.encode())
        if status != 0:
            raise RuntimeError(f"vdb_drop_index failed (code {status}).")

    def batch_search(
        self, index_name: str, queries: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parallel KNN Hamming search.

        Parameters
        ----------
        queries : np.ndarray, shape (N, 6) int64
            Packed binary query vectors (output of ``binarize``).
        k : int
            Number of nearest neighbours per query.

        Returns
        -------
        ids : np.ndarray, shape (N, k) int64
        distances : np.ndarray, shape (N, k) int32
            Hamming distances — lower is better.
        """
        n = queries.shape[0]
        queries_c = np.ascontiguousarray(queries, dtype=np.int64)
        out_ids   = np.empty(n * k, dtype=np.int64)
        out_dists = np.empty(n * k, dtype=np.int32)

        status = self.lib.vdb_batch_search(
            self._thread,
            index_name.encode(),
            queries_c.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(n),
            ctypes.c_int(k),
            out_ids.ctypes.data_as(ctypes.c_void_p),
            out_dists.ctypes.data_as(ctypes.c_void_p),
        )
        if status != 0:
            raise RuntimeError(f"vdb_batch_search failed (code {status}).")

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
        Single-pass Multi-Family Resonant Voting scan.

        Parameters
        ----------
        queries : shape (N, 6) int64
        families : shape (N,) int32 — family ID per query
        thresholds : shape (N,) int32 — Hamming threshold per query
        voting_mask : shape (total_records,) uint8 — modified in-place

        Returns
        -------
        int — number of resonant tiles (voting_mask != 0)
        """
        return self.lib.vdb_query_planetary_grid(
            self._thread,
            index_name.encode(),
            queries.ctypes.data_as(ctypes.c_void_p),
            families.ctypes.data_as(ctypes.c_void_p),
            thresholds.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(queries.shape[0]),
            voting_mask.ctypes.data_as(ctypes.c_void_p),
        )

    def set_chunk_size(self, index_name: str, chunk_size: int) -> None:
        """Tune the L1-cache chunk size for the parallel scan."""
        self.lib.vdb_set_chunk_size(self._thread, index_name.encode(), chunk_size)

    def size(self, index_name: str) -> int:
        """Return the number of vectors in the named index."""
        return self.lib.vdb_size(self._thread, index_name.encode())

    def close(self) -> None:
        """Tear down the GraalVM isolate and release off-heap memory."""
        if self._thread:
            self.lib.vdb_close(self._thread)
            self.lib.graal_tear_down_isolate(self._thread)
            self._thread  = None
            self._isolate = None

    def __enter__(self) -> "LcvkEngine":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _configure_signatures(self) -> None:
        lib = self.lib
        PP_Isolate = ctypes.POINTER(ctypes.POINTER(_GraalIsolate))
        PP_Thread  = ctypes.POINTER(ctypes.POINTER(_GraalIsolateThread))
        P_Thread   = ctypes.c_void_p

        lib.graal_create_isolate.argtypes = [ctypes.c_void_p, PP_Isolate, PP_Thread]
        lib.graal_create_isolate.restype  = ctypes.c_int

        lib.graal_tear_down_isolate.argtypes = [P_Thread]
        lib.graal_tear_down_isolate.restype  = ctypes.c_int

        lib.vdb_init.argtypes = [P_Thread]
        lib.vdb_init.restype  = ctypes.c_int

        lib.vdb_load_index.argtypes = [P_Thread, ctypes.c_char_p, ctypes.c_char_p]
        lib.vdb_load_index.restype  = ctypes.c_int

        lib.vdb_batch_search.argtypes = [
            P_Thread, ctypes.c_char_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.vdb_batch_search.restype = ctypes.c_int

        lib.vdb_query_planetary_grid.argtypes = [
            P_Thread, ctypes.c_char_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_void_p,
        ]
        lib.vdb_query_planetary_grid.restype = ctypes.c_longlong

        # Signature mirrors CApi.java exactly:
        #   compileIndexFile(thread, path, planetId, planetRadius, ids, vectors, numRecords)
        lib.vdb_compile_index_file.argtypes = [
            P_Thread,           # IsolateThread thread
            ctypes.c_char_p,    # CCharPointer   path
            ctypes.c_byte,      # byte           planetId
            ctypes.c_longlong,  # long           planetRadius
            ctypes.c_void_p,    # CLongPointer   ids
            ctypes.c_void_p,    # CLongPointer   vectors  (N×6 int64, row-major)
            ctypes.c_int,       # int            numRecords
        ]
        lib.vdb_compile_index_file.restype = ctypes.c_int

        lib.vdb_drop_index.argtypes = [P_Thread, ctypes.c_char_p]
        lib.vdb_drop_index.restype  = ctypes.c_int

        lib.vdb_set_chunk_size.argtypes = [P_Thread, ctypes.c_char_p, ctypes.c_longlong]
        lib.vdb_set_chunk_size.restype  = ctypes.c_int

        lib.vdb_size.argtypes = [P_Thread, ctypes.c_char_p]
        lib.vdb_size.restype  = ctypes.c_longlong

        lib.vdb_close.argtypes = [P_Thread]
        lib.vdb_close.restype  = ctypes.c_int
