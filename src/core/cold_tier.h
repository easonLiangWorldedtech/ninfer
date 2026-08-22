#pragma once

// Cross-GPU VRAM cold tier: raw byte arenas resident on secondary devices that
// temporarily store parked sequence-state images, plus a pinned host staging
// buffer. The tier owns no state semantics; the target runtime serializes and
// restores images. Transfers stage through host memory (consumer GPUs expose
// no P2P) and are synchronous at the engine boundary.

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <span>
#include <vector>

namespace ninfer {

struct ColdTierConfig {
    // Explicit target device index. -1 spans every non-primary device.
    int device         = -1;
    int primary_device = 0;
    // Arena bytes per device; 0 selects device free memory minus 256 MiB.
    std::size_t capacity_bytes = 0;
    // Smallest per-device arena size accepted in automatic sizing.
    std::size_t min_capacity = 512ULL * 1024ULL * 1024ULL;
    // Pinned staging buffer bytes.
    std::size_t staging_bytes = 64ULL * 1024ULL * 1024ULL;
};

class ColdTier {
public:
    struct Handle {
        std::uint64_t id   = 0;
        std::int32_t arena = -1;
        std::size_t offset = 0;
        std::size_t size   = 0;
    };

    struct Stats {
        std::size_t arena_count       = 0;
        std::size_t entry_count       = 0;
        std::size_t used_bytes        = 0;
        std::size_t evictions         = 0;
        std::size_t rejected_oversize = 0;
    };

    static std::optional<ColdTier> create(const ColdTierConfig& config);

    ColdTier(ColdTier&& other) noexcept;
    ColdTier& operator=(ColdTier&& other) noexcept;
    ~ColdTier();

    // Allocates one region of the arena set, evicting least-recently-used
    // entries across arenas until the request fits. Returns nullopt when the
    // request exceeds every single arena.
    [[nodiscard]] std::optional<Handle> allocate(std::size_t size,
                                                 std::vector<std::uint64_t>* evicted);
    void free_entry(const Handle& handle);
    void touch(const Handle& handle);

    [[nodiscard]] void* device_ptr(const Handle& handle) const;
    [[nodiscard]] int device_of(const Handle& handle) const;

    [[nodiscard]] cudaStream_t staging_stream() const noexcept { return staging_stream_; }

    [[nodiscard]] void* staging() const noexcept { return staging_; }

    [[nodiscard]] std::size_t staging_bytes() const noexcept { return staging_bytes_; }

    [[nodiscard]] std::size_t total_capacity_bytes() const noexcept { return capacity_; }

    [[nodiscard]] int primary_device() const noexcept { return primary_device_; }

    [[nodiscard]] Stats stats() const;

private:
    struct FreeRange {
        std::size_t offset = 0;
        std::size_t size   = 0;
    };

    struct Arena {
        int device           = -1;
        void* ptr            = nullptr;
        std::size_t capacity = 0;
        std::map<std::uint64_t, FreeRange> live;
        std::vector<FreeRange> free;
    };

    ColdTier() = default;

    static std::size_t free_device_memory(int device);
    void coalesce_free(std::size_t arena);
    void evict_oldest(std::vector<std::uint64_t>* evicted);

    int primary_device_          = 0;
    void* staging_               = nullptr;
    cudaStream_t staging_stream_ = nullptr;
    std::size_t staging_bytes_   = 0;
    std::size_t capacity_        = 0;
    std::size_t used_            = 0;
    std::uint64_t next_id_       = 1;
    std::vector<Arena> arenas_;
    std::deque<std::uint64_t> lru_;
    std::size_t evictions_         = 0;
    std::size_t rejected_oversize_ = 0;
};

// Synchronous staged bulk transfer through a pinned host buffer. The source
// spans (all on src_device) and destination spans (all on dst_device) must
// sum to the same total; the transfer is chunked by staging_bytes. The
// ambient CUDA device is restored on return. Throws on size or CUDA errors.
void staged_span_transfer(int src_device, cudaStream_t src_stream,
                          std::span<const const void*> src_ptrs,
                          std::span<const std::size_t> src_sizes, int dst_device,
                          cudaStream_t dst_stream, std::span<const void*> dst_ptrs,
                          std::span<const std::size_t> dst_sizes, void* staging,
                          std::size_t staging_bytes);

} // namespace ninfer
