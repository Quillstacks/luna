# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.stdint cimport uint8_t, uint32_t, uint64_t, int16_t, int32_t
from libc.stdlib cimport malloc, free
from libcpp.atomic cimport atomic
from libc.string cimport memcpy
from luna.screening.engine.disruptor cimport DisruptorEngine, NACSlot
import numpy as np

cdef extern from "<stdint.h>":
    uint64_t UINT64_MAX

cdef extern from *:
    """
    #if defined(__x86_64__) || defined(__i386__)
        #include <immintrin.h>
        #define cpu_relax() _mm_pause()
    #elif defined(__aarch64__) || defined(__arm__)
        #include <arm_acle.h>
        #define cpu_relax() __yield()
    #else
        #define cpu_relax() ((void)0)
    #endif
    """
    void cpu_relax() nogil


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEF TILE_SIZE     = 256      # must match luna.config.TILE_SIZE
DEF PDS3_OFFSET   = 5064     # PDS3 header size in bytes
DEF STRIDE        = 192      # Overlap
DEF LROC_VALID_MIN = -32752  # LROC sensor: everything below is null or saturation artifact


cdef struct GPUFeederBatch:
    uint8_t* tensor_data      # Contiguous UINT8 buffer for PyTorch [B, 1, H, W]
    int32_t* tile_offsets
    int      current_tiles
    int      max_tiles


# ─────────────────────────────────────────────────────────────────────────────
# CONSUMER
# ─────────────────────────────────────────────────────────────────────────────

cdef class NACTransformer:

    cdef:
        DisruptorEngine engine
        uint64_t        my_sequence
        GPUFeederBatch  batch
        int             consumer_id
        atomic[int]     is_running
        atomic[int]     batch_ready

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def __cinit__(self, DisruptorEngine engine, int max_batch_size=64, int consumer_id=0):
        self.engine      = engine
        self.my_sequence = 0
        self.consumer_id = consumer_id

        self.is_running.store(1)
        self.batch_ready.store(0)

        # Allocate feeder buffer: max_batch_size × 1ch × TILE_SIZE² × 1 byte
        self.batch.max_tiles     = max_batch_size
        self.batch.current_tiles = 0
        self.batch.tensor_data   = <uint8_t*>malloc(
            max_batch_size * TILE_SIZE * TILE_SIZE * sizeof(uint8_t)
        )
        if not self.batch.tensor_data:
            raise MemoryError("Failed to allocate GPU feeder buffer.")
        self.batch.tile_offsets = <int32_t*>malloc(max_batch_size * 2 * sizeof(int32_t))

    def stop(self):
        self.is_running.store(0)

    def __dealloc__(self):
        if self.batch.tensor_data is not NULL:
            free(self.batch.tensor_data)
        if self.batch.tile_offsets is not NULL:
            free(self.batch.tile_offsets)

    # ── Main loop (GIL-free) ──────────────────────────────────────────────────

    def run_forever(self):
        """
        Intended to run in a dedicated thread.
        Drops the GIL immediately and processes slots at C speed.
        """
        cdef uint64_t available_cursor
        cdef uint64_t seq

        with nogil:
            while self.is_running.load() == 1:
                available_cursor = self.engine.cursor.value.load()

                if available_cursor != UINT64_MAX and available_cursor >= self.my_sequence:
                    # Drain all available slots in one pass (batching effect)
                    for seq in range(self.my_sequence, available_cursor + 1):
                        self._process_slot(seq)

                    # Release backpressure: signal progress to the producer
                    self.engine.consumer_seqs[self.consumer_id].value.store(available_cursor)
                    self.my_sequence = available_cursor + 1

                else:
                    cpu_relax()

    # ── Slot processing ───────────────────────────────────────────────────────

    cdef void _process_slot(self, uint64_t sequence) noexcept nogil:
        cdef NACSlot* slot = self.engine.get_slot(sequence)

        # Skip PDS3 header; interpret payload as signed 16-bit (little-endian)
        cdef int16_t* img_data = <int16_t*>(slot.raw_ptr + PDS3_OFFSET)

        cdef uint32_t x, y
        for y in range(0, slot.height - TILE_SIZE + 1, STRIDE):
            for x in range(0, slot.width - TILE_SIZE + 1, STRIDE):
                self._norm_and_feed(img_data, slot.width, x, y)

                if self.batch.current_tiles >= self.batch.max_tiles:
                    self._flush_to_gpu()

        if self.batch.current_tiles > 0:
            self._flush_to_gpu()

    cdef void _norm_and_feed(
        self,
        int16_t*  raw_image,
        uint32_t  img_width,
        uint32_t  start_x,
        uint32_t  start_y
    ) noexcept nogil:
        """
        BLOCK 1 & 2 — TILING + NORMALIZATION
        Reads a TILE_SIZE² region from the 16-bit signed image,
        computes a local min/max (excluding LROC invalid pixels),
        and writes contrast-stretched UINT8 values into the feeder buffer.
        """
        cdef int        tx, ty
        cdef int32_t    pixel
        cdef uint64_t   src_idx
        cdef uint8_t*   batch_ptr = (
            self.batch.tensor_data + self.batch.current_tiles * TILE_SIZE * TILE_SIZE
        )
        cdef int16_t local_min = 32767
        cdef int16_t local_max = -32768

        for ty in range(TILE_SIZE):
            for tx in range(TILE_SIZE):
                src_idx = (start_y + ty) * img_width + (start_x + tx)
                pixel   = raw_image[src_idx]

                if pixel >= LROC_VALID_MIN:
                    if pixel < local_min: local_min = pixel
                    if pixel > local_max: local_max = pixel

        if local_max < local_min:
            local_min = 0
            local_max = 1
            
        cdef float diff = <float>(local_max - local_min)
        if diff < 1e-6: 
            diff = 1.0
            
        cdef int dst_idx = 0
        cdef float normalized

        for ty in range(TILE_SIZE):
            for tx in range(TILE_SIZE):
                src_idx = (start_y + ty) * img_width + (start_x + tx)
                pixel   = raw_image[src_idx]

                if pixel < LROC_VALID_MIN:
                    batch_ptr[dst_idx] = 0
                else:
                    normalized = (pixel - local_min) / diff
                    if normalized < 0.0: normalized = 0.0
                    if normalized > 1.0: normalized = 1.0
                    batch_ptr[dst_idx] = <uint8_t>(normalized * 255.0)

                dst_idx += 1

        self.batch.tile_offsets[self.batch.current_tiles * 2] = <int32_t>start_x
        self.batch.tile_offsets[self.batch.current_tiles * 2 + 1] = <int32_t>start_y

        self.batch.current_tiles += 1

    # ── Python bridge ─────────────────────────────────────────────────────────

    def get_current_batch_with_offsets(self):
        cdef int n = self.batch.current_tiles if self.batch.current_tiles > 0 else self.batch.max_tiles

        img_arr = np.empty((n, TILE_SIZE, TILE_SIZE), dtype=np.uint8)
        offset_arr = np.empty((n, 2), dtype=np.int32)

        cdef uint8_t[:, :, ::1] img_view = img_arr
        cdef int32_t[:, ::1] offset_view = offset_arr

        cdef Py_ssize_t img_bytes = n * TILE_SIZE * TILE_SIZE * sizeof(uint8_t)
        memcpy(&img_view[0, 0, 0], self.batch.tensor_data, img_bytes)

        cdef Py_ssize_t offset_bytes = n * 2 * sizeof(int32_t)
        memcpy(&offset_view[0, 0], self.batch.tile_offsets, offset_bytes)

        return img_arr, offset_arr

    @property
    def is_batch_ready(self):
        return self.batch_ready.load() == 1

    @is_batch_ready.setter
    def is_batch_ready(self, int value):
        self.batch_ready.store(value)

    def trigger_final_flush(self):
        """Called from Python when the image is fully consumed, to flush any remaining tiles."""
        if self.batch.current_tiles > 0:
            self.batch_ready.store(1)

    cdef void _flush_to_gpu(self) noexcept nogil:
        """
        BLOCK 3 — EMBEDDER FEEDER
        Signals Python that a full batch is ready in RAM for GPU upload,
        then busy-waits until Python resets the flag.
        The is_running guard prevents a deadlock on shutdown.
        """
        self.batch_ready.store(1)

        while self.batch_ready.load() == 1 and self.is_running.load() == 1:
            cpu_relax()

        self.batch.current_tiles = 0