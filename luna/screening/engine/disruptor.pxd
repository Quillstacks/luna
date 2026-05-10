# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport uint64_t, uint32_t, uint8_t
from libcpp.atomic cimport atomic

cdef struct PaddedSequence:
    atomic[uint64_t] value
    uint64_t[7] padding

cdef struct NACSlot:
    uint8_t* raw_ptr
    uint32_t  width
    uint32_t  height
    uint64_t  stripe_id
    uint64_t  timestamp

cdef class DisruptorEngine:
    cdef:
        int            ring_size
        int            mask_limit
        NACSlot* ring_buffer
        PaddedSequence cursor
        PaddedSequence next_producer
        PaddedSequence* consumer_seqs
        int             num_consumers

    cdef uint64_t claim_next(self) noexcept nogil
    cdef void commit(self, uint64_t sequence) noexcept nogil
    cdef NACSlot* get_slot(self, uint64_t sequence) noexcept nogil