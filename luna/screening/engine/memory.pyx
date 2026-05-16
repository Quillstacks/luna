# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdint cimport uint8_t, uint32_t, uint64_t
from luna.screening.engine.disruptor cimport DisruptorEngine, NACSlot


# ─────────────────────────────────────────────────────────────────────────────
# POSIX C-APIs — Bypass Python I/O entirely
# ─────────────────────────────────────────────────────────────────────────────

cdef extern from "<sys/mman.h>" nogil:
    void* mmap(void* addr, size_t length, int prot, int flags, int fd, uint64_t offset)
    int   munmap(void* addr, size_t length)
    int   PROT_READ
    int   MAP_PRIVATE
    void* MAP_FAILED

cdef extern from "<fcntl.h>" nogil:
    int open(const char* pathname, int flags)
    int O_RDONLY

cdef extern from "<unistd.h>" nogil:
    int close(int fd)

cdef extern from "<sys/stat.h>" nogil:
    struct stat:
        uint64_t st_size
    int fstat(int fd, stat* statbuf)


# ─────────────────────────────────────────────────────────────────────────────
# ZERO-COPY STRIPE
# ─────────────────────────────────────────────────────────────────────────────

cdef class MappedStripe:
    """
    Maps a 550 MB NAC image directly into virtual address space.
    No malloc, no GC — pure DMA.
    """
    cdef:
        int             fd
        public uint64_t size
        public uint8_t* raw_ptr

    def __cinit__(self, str filepath):
        cdef bytes      py_path = filepath.encode('utf-8')
        cdef const char* c_path = py_path

        # 1. Open file via C API
        self.fd = open(c_path, O_RDONLY)
        if self.fd < 0:
            raise IOError(f"[OS ERROR] Failed to open: {filepath}")

        # 2. Determine file size
        cdef stat statbuf
        if fstat(self.fd, &statbuf) < 0:
            close(self.fd)
            raise IOError(f"[OS ERROR] Failed to stat: {filepath}")
        self.size = statbuf.st_size

        # 3. Memory-map the file — kernel pages on demand, zero copy
        self.raw_ptr = <uint8_t*>mmap(NULL, self.size, PROT_READ, MAP_PRIVATE, self.fd, 0)
        if self.raw_ptr == <uint8_t*>MAP_FAILED:
            close(self.fd)
            raise MemoryError(f"[OS ERROR] mmap failed for {filepath}")

    def __dealloc__(self):
        # Return the mapping and fd to the OS on object death
        if self.raw_ptr != NULL and self.raw_ptr != <uint8_t*>MAP_FAILED:
            munmap(self.raw_ptr, self.size)
        if self.fd >= 0:
            close(self.fd)


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCER — Proof of Concept
# ─────────────────────────────────────────────────────────────────────────────

def push_stripe_to_ring(
    DisruptorEngine engine,
    MappedStripe    stripe,
    uint32_t        width,
    uint32_t        height,
    uint64_t        stripe_id
):
    """
    Python-facing entry point for initial integration testing.
    In production this runs entirely inside a nogil C loop.
    """
    cdef uint64_t  seq
    cdef NACSlot*  slot

    # 1. Claim slot — lock-free busy-spin if ring is full
    seq  = engine.claim_next()
    slot = engine.get_slot(seq)

    # 2. Write metadata only — pixel data stays in the mmap region
    slot.raw_ptr   = stripe.raw_ptr
    slot.width     = width
    slot.height    = height
    slot.stripe_id = stripe_id
    slot.timestamp = 0  # inject TSC here

    # 3. Publish — slot is now visible to consumers
    engine.commit(seq)

    return seq