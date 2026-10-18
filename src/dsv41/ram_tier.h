// The RAM tier of the routed experts: a pool of expert slots in ordinary memory (huge pages when the kernel gives
// them, pinned for DMA), filled from the model files with O_DIRECT reads (the page cache would hold a second copy and
// let any other I/O evict the first), an LRU over the slots. A slot is one expert's [gate | up | down], contiguous.
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

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

    bool contains(int key) const { return where_.count(key) != 0; }   // no LRU update
    // the resident copy of `key` (moved to the LRU front), or nullptr
    const uint8_t * find(int key);
    // makes every key resident, reading the missing ones from disk in parallel; out[i] = its slot. Keys of one call
    // never evict each other (n_keys <= slots). Blocks until all reads are done.
    void fetch(const int * keys, const Src * src, int n_keys, const uint8_t ** out);

    // statistics
    uint64_t hits = 0, reads = 0;
    double   read_s = 0;   // wall time inside fetch's reads

private:
    struct Io;
    uint8_t * pool_ = nullptr;
    size_t    pool_bytes_ = 0, pool_bytes_pinned_ = 0, per_ = 0, chunk_ = 1;
    int       n_ = 0;
    bool      pinned_ = false;
    std::vector<int> key_;   // per slot, -1 free
    std::list<int>   lru_;   // slots, most recently used first
    std::unordered_map<int, std::list<int>::iterator> where_;
    std::unique_ptr<Io> io_;
};

}  // namespace bl
