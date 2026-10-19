// The RAM tier of the routed experts: a pool of expert slots in ordinary memory (huge pages when the kernel gives
// them, pinned for DMA), filled from the model files with O_DIRECT reads (the page cache would hold a second copy and
// let any other I/O evict the first), an LRU over the slots. A slot is one expert's [gate | up | down], contiguous.
// Reads are either demanded (fetch: blocks until its slots are filled) or prefetched (prefetch: queued behind the
// demanded ones, returns at once); a slot with reads in flight is never evicted, and a fetch of one waits for it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace bl {

class ExpertRam {
public:
    // where one expert's three parts are in the files: an O_DIRECT descriptor and byte range per part
    struct Src {
        int      fd[3];
        uint64_t off[3];
        size_t   len[3];
    };

    // bytes: the pool size (rounded down to whole slots); per: one expert's bytes; pin: cudaHostRegister the pool
    ExpertRam(size_t bytes, size_t per, int io_threads, bool pin);
    ~ExpertRam();
    ExpertRam(const ExpertRam &) = delete;
    ExpertRam & operator=(const ExpertRam &) = delete;

    int    slots() const { return n_; }
    size_t bytes() const { return static_cast<size_t>(n_) * per_; }
    bool   pinned() const { return pinned_; }

    bool contains(int key) const { return where_.count(key) != 0; }   // no LRU update (may still be in flight)
    // makes every key resident, reading the missing ones from disk in parallel; out[i] = its slot. Keys of one call
    // never evict each other (n_keys <= slots). Blocks until all reads are done, prefetches of these keys included.
    void fetch(const int * keys, const Src * src, int n_keys, const uint8_t ** out);
    // fetch in two halves: start() queues the reads and returns; ready[i]: key i is usable now (resident, no read
    // or copy in flight); slots[i] is what wait() takes. The out pointers stay valid until the next start/fetch/prefetch.
    void start(const int * keys, const Src * src, int n_keys, const uint8_t ** out, int * slots, char * ready);
    void wait(const int * slots, int n);
    // starts reading the keys that are not resident, behind any demanded reads; returns at once
    void prefetch(const int * keys, const Src * src, int n_keys);

    // device copies (an exclusive VRAM tier above): `ready` is recorded after the copy that uses the slot. Until it
    // completes the slot is not evicted, and a fetch of its key waits for it.
    // a slot for `key` that a device-to-host copy fills (an expert leaving VRAM); nullptr if the key is resident
    uint8_t * insert(int key, cudaEvent_t ready);
    // forgets `key` (it moved into VRAM); its slot is the next to reuse once `ready` (the copy out of it) completes
    void drop(int key, cudaEvent_t ready);

    // statistics
    uint64_t hits = 0, reads = 0;            // fetch: found resident / read on demand
    uint64_t prefetched = 0, prefetch_used = 0, prefetch_waits = 0;   // prefetch reads / later fetched / fetched in flight
    double   read_s = 0;                     // wall time fetch waits for reads

private:
    struct Io;
    int victim();   // the least recently used slot with no reads or copies in flight, taken off the LRU
    bool held(int slot);   // a device copy of the slot still running
    template <typename V> void add_tasks(V & tasks, int fd, uint64_t off, size_t len, uint8_t * dst, int slot);
    uint8_t * pool_ = nullptr;
    size_t    pool_bytes_ = 0, pool_bytes_pinned_ = 0, per_ = 0, chunk_ = 1, chunk_bytes_ = 0;
    int       n_ = 0;
    bool      pinned_ = false;
    std::vector<int>  key_;   // per slot, -1 free
    std::vector<char> pf_;    // per slot: filled by a prefetch, not fetched since
    std::vector<cudaEvent_t> hold_;   // per slot: the device copy it waits for (nullptr: none)
    std::list<int>    lru_;   // slots, most recently used first
    std::unordered_map<int, std::list<int>::iterator> where_;
    std::unique_ptr<Io> io_;
};

}  // namespace bl
