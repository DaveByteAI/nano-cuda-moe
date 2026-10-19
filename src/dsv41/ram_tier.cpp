#include "ram_tier.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <cuda_runtime.h>

namespace bl {

namespace {
constexpr size_t kAlign = 4096;   // O_DIRECT: offset, length and buffer aligned to the logical block (4 KiB covers it)
constexpr size_t kPinChunk = size_t{1} << 30;
}  // namespace

// a fixed pool of threads, each with an aligned bounce buffer: a task reads the aligned superset of one part into
// it and copies the wanted bytes into the slot. Two queues: demanded reads go first, prefetches when none is waiting.
struct ExpertRam::Io {
    struct Task { int fd; uint64_t off; size_t len; uint8_t * dst; int slot; };
    std::vector<std::thread> threads;
    std::mutex mu;
    std::condition_variable cv, done_cv;
    std::deque<Task> demand, background;
    std::vector<int> inflight;   // per slot: tasks queued or running
    bool   quit = false;
    std::string error;
    size_t bounce_bytes;

    Io(int n, size_t max_part, int n_slots)
        : inflight(n_slots, 0), bounce_bytes((max_part + 2 * kAlign + kAlign - 1) / kAlign * kAlign) {
        for (int i = 0; i < n; ++i) threads.emplace_back([this] { loop(); });
    }
    ~Io() {
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        for (auto & t : threads) t.join();
    }
    void loop() {
        void * buf = nullptr;
        if (posix_memalign(&buf, kAlign, bounce_bytes) != 0) buf = nullptr;
        while (true) {
            Task t;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || !demand.empty() || !background.empty(); });
                if (quit) break;
                std::deque<Task> & q = demand.empty() ? background : demand;
                t = q.front();
                q.pop_front();
            }
            std::string err;
            if (!buf) err = "bounce buffer allocation failed";
            else {
                const uint64_t a = t.off / kAlign * kAlign;
                const size_t span = (t.off + t.len - a + kAlign - 1) / kAlign * kAlign;
                size_t got = 0;
                while (got < span) {
                    const ssize_t r = ::pread(t.fd, static_cast<uint8_t *>(buf) + got, span - got, static_cast<off_t>(a + got));
                    if (r < 0) { err = std::string("pread: ") + std::strerror(errno); break; }
                    if (r == 0) break;   // the end of the file: the last part's tail is short of a block
                    got += static_cast<size_t>(r);
                }
                if (err.empty() && got < t.off - a + t.len) err = "short read";
                if (err.empty()) std::memcpy(t.dst, static_cast<uint8_t *>(buf) + (t.off - a), t.len);
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                if (!err.empty() && error.empty()) error = err;
                if (--inflight[t.slot] == 0) done_cv.notify_all();
            }
        }
        std::free(buf);
    }
    void enqueue(const std::vector<Task> & tasks, bool bg) {
        if (tasks.empty()) return;
        {
            std::lock_guard<std::mutex> lk(mu);
            for (const Task & t : tasks) {
                ++inflight[t.slot];
                (bg ? background : demand).push_back(t);
            }
        }
        cv.notify_all();
    }
    bool busy(int slot) {
        std::lock_guard<std::mutex> lk(mu);
        return inflight[slot] != 0;
    }
    // blocks until no read of these slots is queued or running
    void promote(const int * slots, int n) {
        std::lock_guard<std::mutex> lk(mu);
        bool moved = false;
        for (int i = 0; i < n; ++i) {
            const int s = slots[i];
            if (!inflight[s]) continue;
            for (auto it = background.begin(); it != background.end();) {
                if (it->slot == s) { demand.push_front(*it); it = background.erase(it); moved = true; }
                else ++it;
            }
        }
        if (moved) cv.notify_all();
    }
    void wait(const std::vector<int> & slots) {
        std::unique_lock<std::mutex> lk(mu);
        done_cv.wait(lk, [&] {
            for (int s : slots) if (inflight[s]) return false;
            return true;
        });
        if (!error.empty()) {
            const std::string e = error;
            error.clear();
            throw std::runtime_error("expert read: " + e);
        }
    }
};

ExpertRam::ExpertRam(size_t bytes, size_t per, int io_threads, bool pin) : per_(per) {
    n_ = static_cast<int>(bytes / per);
    if (n_ <= 0) return;
    pool_bytes_ = static_cast<size_t>(n_) * per;
    void * p = ::mmap(nullptr, pool_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("mmap of the RAM expert tier failed");
    ::madvise(p, pool_bytes_, MADV_HUGEPAGE);
    pool_ = static_cast<uint8_t *>(p);
    if (pin) {   // in chunks of whole slots (a copy may not span two registrations), ~1 GB each
        pinned_ = true;
        chunk_ = std::max<size_t>(1, kPinChunk / per) * per;
        for (size_t o = 0; o < pool_bytes_; o += chunk_) {
            const size_t len = std::min(chunk_, pool_bytes_ - o);
            if (cudaHostRegister(pool_ + o, len, cudaHostRegisterDefault) != cudaSuccess) {
                cudaGetLastError();
                std::fprintf(stderr, "dsv41: RAM tier pinned up to %.1f GB only; the rest copies to VRAM unpinned\n", o / 1e9);
                pinned_ = o > 0;
                pool_bytes_pinned_ = o;
                break;
            }
            pool_bytes_pinned_ = o + len;
        }
    }
    key_.assign(n_, -1);
    pf_.assign(n_, 0);
    hold_.assign(n_, nullptr);
    for (int i = 0; i < n_; ++i) lru_.push_back(i);
    // BL_DSV41_IO_CHUNK_KB: split each part's read (default 1536: two threads stay busy to the end of one expert;
    // tests/bench_dsv41_ram.cpp)
    chunk_bytes_ = size_t{1536} * 1024;
    if (const char * e = std::getenv("BL_DSV41_IO_CHUNK_KB")) chunk_bytes_ = static_cast<size_t>(std::atol(e)) * 1024;
    io_ = std::make_unique<Io>(std::max(io_threads, 1), per, n_);
}

ExpertRam::~ExpertRam() {
    io_.reset();
    if (pool_) {
        for (size_t o = 0; o < pool_bytes_pinned_; o += chunk_) cudaHostUnregister(pool_ + o);
        ::munmap(pool_, pool_bytes_);
    }
}

// one part as reads of at most chunk_bytes_ each (so two threads stay busy to the end of one expert)
template <typename V> void ExpertRam::add_tasks(V & tasks, int fd, uint64_t off, size_t len, uint8_t * dst, int slot) {
    const size_t ch = chunk_bytes_ ? chunk_bytes_ : len;
    for (size_t o = 0; o < len; o += ch) tasks.push_back({fd, off + o, std::min(ch, len - o), dst + o, slot});
}

bool ExpertRam::held(int slot) {
    if (!hold_[slot]) return false;
    const cudaError_t e = cudaEventQuery(hold_[slot]);
    if (e == cudaErrorNotReady) return true;
    if (e != cudaSuccess) throw std::runtime_error(std::string("RAM tier copy: ") + cudaGetErrorString(e));
    hold_[slot] = nullptr;
    return false;
}

int ExpertRam::victim() {
    for (auto it = lru_.rbegin(); it != lru_.rend(); ++it) {
        if (io_->busy(*it) || held(*it)) continue;   // a prefetch still filling it / a device copy using it
        const int slot = *it;
        lru_.erase(std::next(it).base());
        if (key_[slot] >= 0) where_.erase(key_[slot]);
        key_[slot] = -1;
        pf_[slot] = 0;
        hold_[slot] = nullptr;
        return slot;
    }
    throw std::runtime_error("RAM tier: every slot has a read in flight");
}

void ExpertRam::fetch(const int * keys, const Src * src, int n_keys, const uint8_t ** out) {
    std::vector<int> slots(n_keys);
    std::vector<char> ready(n_keys);
    start(keys, src, n_keys, out, slots.data(), ready.data());
    wait(slots.data(), n_keys);
}

void ExpertRam::wait(const int * slot, int n) {
    std::vector<int> slots(slot, slot + n);
    for (int s : slots)
        if (hold_[s]) {   // filled by a device copy still running
            if (cudaEventSynchronize(hold_[s]) != cudaSuccess) throw std::runtime_error("RAM tier copy failed");
            hold_[s] = nullptr;
        }
    const auto t0 = std::chrono::steady_clock::now();
    io_->wait(slots);
    read_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertRam::start(const int * keys, const Src * src, int n_keys, const uint8_t ** out, int * slots, char * ready) {
    if (n_keys > n_) throw std::runtime_error("RAM tier: more experts at once than slots");
    std::vector<Io::Task> tasks;
    for (int i = 0; i < n_keys; ++i) {
        auto it = where_.find(keys[i]);
        if (it != where_.end()) {
            ++hits;
            const int slot = *it->second;
            lru_.splice(lru_.begin(), lru_, it->second);
            if (pf_[slot]) { ++prefetch_used; pf_[slot] = 0; prefetch_waits += io_->busy(slot); }
            ready[i] = !io_->busy(slot) && !held(slot);
            slots[i] = slot;
            out[i] = pool_ + static_cast<size_t>(slot) * per_;
            continue;
        }
        // the least recently used idle slot: never a key of this call, those were all just moved to the front
        const int slot = victim();
        key_[slot] = keys[i];
        lru_.push_front(slot);
        where_[keys[i]] = lru_.begin();
        uint8_t * dst = pool_ + static_cast<size_t>(slot) * per_;
        slots[i] = slot;
        ready[i] = 0;
        out[i] = dst;
        size_t o = 0;
        for (int k = 0; k < 3; ++k) {
            add_tasks(tasks, src[i].fd[k], src[i].off[k], src[i].len[k], dst + o, slot);
            o += src[i].len[k];
        }
        ++reads;
    }
    io_->enqueue(tasks, false);
    // queued prefetches of these keys move ahead of the other background reads
    io_->promote(slots, n_keys);
}

void ExpertRam::prefetch(const int * keys, const Src * src, int n_keys) {
    std::vector<Io::Task> tasks;
    for (int i = 0; i < n_keys; ++i) {
        if (where_.count(keys[i])) continue;
        const int slot = victim();
        key_[slot] = keys[i];
        pf_[slot] = 1;
        lru_.push_front(slot);
        where_[keys[i]] = lru_.begin();
        uint8_t * dst = pool_ + static_cast<size_t>(slot) * per_;
        size_t o = 0;
        for (int k = 0; k < 3; ++k) {
            add_tasks(tasks, src[i].fd[k], src[i].off[k], src[i].len[k], dst + o, slot);
            o += src[i].len[k];
        }
        ++prefetched;
    }
    io_->enqueue(tasks, true);
}

}  // namespace bl

namespace bl {

uint8_t * ExpertRam::insert(int key, cudaEvent_t ready) {
    if (n_ <= 0 || where_.count(key)) return nullptr;
    const int slot = victim();
    key_[slot] = key;
    hold_[slot] = ready;
    lru_.push_front(slot);
    where_[key] = lru_.begin();
    return pool_ + static_cast<size_t>(slot) * per_;
}

void ExpertRam::drop(int key, cudaEvent_t ready) {
    auto it = where_.find(key);
    if (it == where_.end()) return;
    const int slot = *it->second;
    lru_.erase(it->second);
    where_.erase(it);
    key_[slot] = -1;
    pf_[slot] = 0;
    hold_[slot] = ready;
    lru_.push_back(slot);
}

}  // namespace bl
