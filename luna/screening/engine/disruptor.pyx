# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

from libc.stdint cimport uint64_t, uint32_t, uint8_t    # type: ignore[import]
from libc.stdlib cimport malloc, free                   # type: ignore[import]
from libcpp.atomic cimport atomic                       # type: ignore[import]

cdef extern from "<stdint.h>":
    uint64_t UINT64_MAX

# ─────────────────────────────────────────────────────────────────────────────
# 1. HARDWARE SYMPATHY LAYER — False-Sharing Immunity
# ─────────────────────────────────────────────────────────────────────────────

cdef struct PaddedSequence:
    atomic[uint64_t] value
    uint64_t[7] padding          # Pad to 64-byte cache line


# ─────────────────────────────────────────────────────────────────────────────
# 2. ZERO-COPY PAYLOAD
# ─────────────────────────────────────────────────────────────────────────────

cdef struct NACSlot:
    uint8_t*  raw_ptr            # Pointer into 550 MB mmap region
    uint32_t  width
    uint32_t  height
    uint64_t  stripe_id
    uint64_t  timestamp


# ─────────────────────────────────────────────────────────────────────────────
# 3. DISRUPTOR CORE
# ─────────────────────────────────────────────────────────────────────────────

cdef class DisruptorEngine:

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def __cinit__(self, int size=1024, int num_consumers=1):
        if (size & (size - 1)) != 0:
            raise ValueError("Ring size must be a power of two (e.g. 512, 1024, 2048).")

        self.ring_size     = size
        self.mask_limit    = size - 1   # Replaces expensive modulo with bitwise AND
        self.num_consumers = num_consumers

        self.ring_buffer   = <NACSlot*>malloc(size * sizeof(NACSlot))
        self.consumer_seqs = <PaddedSequence*>malloc(num_consumers * sizeof(PaddedSequence))

        self.cursor.value.store(UINT64_MAX)
        self.next_producer.value.store(0)
        for i in range(num_consumers):
            self.consumer_seqs[i].value.store(UINT64_MAX)

    def __dealloc__(self):
        if self.ring_buffer is not NULL:
            free(self.ring_buffer)
        if self.consumer_seqs is not NULL:
            free(self.consumer_seqs)

    # ── Producer API (nogil-ready) ─────────────────────────────────────────────

    cdef uint64_t claim_next(self) noexcept nogil:
        """
        Atomically claim the next sequence slot.
        """
        cdef uint64_t current_claim = self.next_producer.value.fetch_add(1)
        cdef uint64_t min_consumer
        cdef int i

        while True:
            min_consumer = self.consumer_seqs[0].value.load()
            for i in range(1, self.num_consumers):
                if self.consumer_seqs[i].value.load() < min_consumer:
                    min_consumer = self.consumer_seqs[i].value.load()

            if (current_claim - min_consumer) <= <uint64_t>self.ring_size:
                break
            

        return current_claim

    cdef void commit(self, uint64_t sequence) noexcept nogil:
        """
        Publish a slot to consumers.
        Busy-spins until all preceding sequences have been committed,
        preserving strict ordering on the cursor.
        """
        cdef uint64_t expected = sequence - 1
        while self.cursor.value.load() != expected:
            pass

        self.cursor.value.store(sequence)

    cdef NACSlot* get_slot(self, uint64_t sequence) noexcept nogil:
        """
        Map a sequence number to a ring slot via bitmask (orders of magnitude
        faster than modulo for power-of-two ring sizes).
        """
        return &self.ring_buffer[sequence & self.mask_limit]