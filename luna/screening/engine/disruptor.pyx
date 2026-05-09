from libc.stdint cimport uint64_t, uint8_t  # type: ignore[import]

cdef struct Sequence:
    uint64_t value
    uint64_t[7] padding

cdef struct NACSlot:
    uint8_t* data_ptr
    uint64_t stripe_id
    int width
    int height
    uint64_t timestamp

cdef class DisruptorCore:
    cdef NACSlot* ring_buffer
    cdef Sequence cursor
    cdef Sequence next_free
    cdef int buffer_size

    def __cinit__(self, int size=8):
        self.buffer_size = size