#include "ram_tier.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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
// it and copies the wanted bytes into the slot
struct ExpertRam::Io {
    struct Task { int fd; uint64_t off; size_t len; uint8_t * dst; };
    std::vector<std::thread> threads;
    std::mutex mu;
    std::condition_variable cv, done_cv;
    std::vector<Task> queue;
    size_t next = 0;
    int    pending = 0;
    bool   quit = false;
    std::string error;
    size_t bounce_bytes;

    Io(int n, size_t max_part) : bounce_bytes((max_part + 2 * kAlign + kAlign - 1) / kAlign * kAlign) {
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
                cv.wait(lk, [&] { return quit || next < queue.size(); });
                if (quit) break;
                t = queue[next++];
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
                if (--pending == 0) done_cv.notify_all();
            }
        }
        std::free(buf);
    }
    void run(std::vector<Task> && tasks) {
        if (tasks.empty()) return;
        std::unique_lock<std::mutex> lk(mu);
        queue = std::move(tasks);
        next = 0;
        pending = static_cast<int>(queue.size());
        error.clear();
        cv.notify_all();
        done_cv.wait(lk, [&] { return pending == 0; });
        if (!error.empty()) throw std::runtime_error("expert read: " + error);
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
    for (int i = 0; i < n_; ++i) lru_.push_back(i);
    io_ = std::make_unique<Io>(std::max(io_threads, 1), per);
}

ExpertRam::~ExpertRam() {
    io_.reset();
    if (pool_) {
        for (size_t o = 0; o < pool_bytes_pinned_; o += chunk_) cudaHostUnregister(pool_ + o);
        ::munmap(pool_, pool_bytes_);
    }
}

const uint8_t * ExpertRam::find(int key) {
    auto it = where_.find(key);
    if (it == where_.end()) return nullptr;
    ++hits;
    lru_.splice(lru_.begin(), lru_, it->second);
    return pool_ + static_cast<size_t>(*it->second) * per_;
}

void ExpertRam::fetch(const int * keys, const Src * src, int n_keys, const uint8_t ** out) {
    if (n_keys > n_) throw std::runtime_error("RAM tier: more experts at once than slots");
    std::vector<Io::Task> tasks;
    for (int i = 0; i < n_keys; ++i) {
        if (const uint8_t * p = find(keys[i])) { out[i] = p; continue; }
        // the LRU tail: never a key of this call, those were all just moved to the front
        const int slot = lru_.back();
        lru_.pop_back();
        if (key_[slot] >= 0) where_.erase(key_[slot]);
        key_[slot] = keys[i];
        lru_.push_front(slot);
        where_[keys[i]] = lru_.begin();
        uint8_t * dst = pool_ + static_cast<size_t>(slot) * per_;
        out[i] = dst;
        size_t o = 0;
        for (int k = 0; k < 3; ++k) {
            tasks.push_back({src[i].fd[k], src[i].off[k], src[i].len[k], dst + o});
            o += src[i].len[k];
        }
        ++reads;
    }
    const auto t0 = std::chrono::steady_clock::now();
    io_->run(std::move(tasks));
    read_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace bl
