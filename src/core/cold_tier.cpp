#include "core/cold_tier.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr std::size_t kAutoHeadroomBytes = 256ULL * 1024ULL * 1024ULL;

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

struct DeviceScope {
    int previous = 0;
    bool changed = false;

    explicit DeviceScope(int device) {
        check_cuda(cudaGetDevice(&previous), "cold tier device query");
        if (previous != device) {
            check_cuda(cudaSetDevice(device), "cold tier set device");
            changed = true;
        }
    }

    ~DeviceScope() {
        if (changed) { check_cuda(cudaSetDevice(previous), "cold tier restore device"); }
    }
};

std::size_t span_total(std::span<const std::size_t> sizes) {
    std::size_t total = 0;
    for (std::size_t size : sizes) { total += size; }
    return total;
}

} // namespace

std::size_t ColdTier::free_device_memory(int device) {
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    DeviceScope scope(device);
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cold tier memory query");
    return free_bytes;
}

std::optional<ColdTier> ColdTier::create(const ColdTierConfig& config) {
    ColdTier tier;
    tier.primary_device_ = config.primary_device;

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cold tier device count");
    std::vector<int> candidates;
    for (int device = 0; device < device_count; ++device) {
        if (config.device >= 0) {
            if (device == config.device) { candidates.push_back(device); }
        } else if (device != config.primary_device) {
            candidates.push_back(device);
        }
    }
    if (candidates.empty()) { return std::nullopt; }

    for (int device : candidates) {
        const std::size_t free_bytes = free_device_memory(device);
        std::size_t capacity         = config.capacity_bytes;
        if (capacity == 0) {
            capacity = free_bytes > kAutoHeadroomBytes ? free_bytes - kAutoHeadroomBytes : 0;
        }
        if (capacity < config.min_capacity || capacity > free_bytes) { continue; }
        DeviceScope scope(device);
        void* arena = nullptr;
        check_cuda(cudaMalloc(&arena, capacity), "cold tier arena allocation");
        Arena added;
        added.device   = device;
        added.ptr      = arena;
        added.capacity = capacity;
        added.free.push_back(FreeRange{.offset = 0, .size = capacity});
        tier.arenas_.push_back(std::move(added));
        tier.capacity_ += capacity;
    }
    if (tier.arenas_.empty()) { return std::nullopt; }

    void* staging = nullptr;
    check_cuda(cudaHostAlloc(&staging, config.staging_bytes, cudaHostAllocDefault),
               "cold tier staging allocation");
    check_cuda(cudaStreamCreate(&tier.staging_stream_), "cold tier staging stream");
    tier.staging_       = staging;
    tier.staging_bytes_ = config.staging_bytes;
    return tier;
}

ColdTier::ColdTier(ColdTier&& other) noexcept
    : primary_device_(other.primary_device_), staging_(other.staging_),
      staging_stream_(other.staging_stream_), staging_bytes_(other.staging_bytes_),
      capacity_(other.capacity_), used_(other.used_), next_id_(other.next_id_),
      arenas_(std::move(other.arenas_)), lru_(std::move(other.lru_)), evictions_(other.evictions_),
      rejected_oversize_(other.rejected_oversize_) {
    other.staging_        = nullptr;
    other.staging_stream_ = nullptr;
    other.capacity_       = 0;
}

ColdTier& ColdTier::operator=(ColdTier&& other) noexcept {
    if (this != &other) {
        this->~ColdTier();
        new (this) ColdTier(std::move(other));
    }
    return *this;
}

ColdTier::~ColdTier() {
    for (Arena& arena : arenas_) {
        if (arena.ptr != nullptr) {
            DeviceScope scope(arena.device);
            check_cuda(cudaFree(arena.ptr), "cold tier arena free");
        }
    }
    if (staging_ != nullptr) { cudaFreeHost(staging_); }
    if (staging_stream_ != nullptr) { cudaStreamDestroy(staging_stream_); }
}

void ColdTier::coalesce_free(std::size_t arena_index) {
    Arena& arena = arenas_[arena_index];
    std::sort(arena.free.begin(), arena.free.end(),
              [](const FreeRange& a, const FreeRange& b) { return a.offset < b.offset; });
    std::vector<FreeRange> merged;
    for (const FreeRange& range : arena.free) {
        if (!merged.empty() && merged.back().offset + merged.back().size == range.offset) {
            merged.back().size += range.size;
        } else {
            merged.push_back(range);
        }
    }
    arena.free = std::move(merged);
}

void ColdTier::evict_oldest(std::vector<std::uint64_t>* evicted) {
    if (lru_.empty()) { return; }
    const std::uint64_t id = lru_.front();
    lru_.pop_front();
    for (std::size_t arena_index = 0; arena_index < arenas_.size(); ++arena_index) {
        const auto it = arenas_[arena_index].live.find(id);
        if (it == arenas_[arena_index].live.end()) { continue; }
        arenas_[arena_index].free.push_back(it->second);
        arenas_[arena_index].live.erase(it);
        used_ -= it->second.size;
        coalesce_free(arena_index);
        break;
    }
    ++evictions_;
    if (evicted != nullptr) { evicted->push_back(id); }
}

std::optional<ColdTier::Handle> ColdTier::allocate(std::size_t size,
                                                   std::vector<std::uint64_t>* evicted) {
    if (size == 0) { throw std::invalid_argument("cold tier allocation size is zero"); }
    if (size > capacity_) {
        ++rejected_oversize_;
        return std::nullopt;
    }

    auto fits = [&](std::size_t arena_index) {
        return std::any_of(arenas_[arena_index].free.begin(), arenas_[arena_index].free.end(),
                           [size](const FreeRange& range) { return range.size >= size; });
    };
    auto any_fits = [&] {
        for (std::size_t arena_index = 0; arena_index < arenas_.size(); ++arena_index) {
            if (fits(arena_index)) { return true; }
        }
        return false;
    };

    if (!any_fits()) {
        for (std::size_t arena_index = 0; arena_index < arenas_.size(); ++arena_index) {
            coalesce_free(arena_index);
        }
        while (!any_fits() && !lru_.empty()) { evict_oldest(evicted); }
    }
    if (!any_fits()) {
        ++rejected_oversize_;
        return std::nullopt;
    }

    std::size_t chosen = 0;
    for (std::size_t arena_index = 0; arena_index < arenas_.size(); ++arena_index) {
        if (fits(arena_index)) {
            chosen = arena_index;
            break;
        }
    }
    Arena& arena    = arenas_[chosen];
    const auto slot = std::find_if(arena.free.begin(), arena.free.end(),
                                   [size](const FreeRange& range) { return range.size >= size; });
    const std::size_t offset = slot->offset;
    slot->offset += size;
    slot->size -= size;
    if (slot->size == 0) { arena.free.erase(slot); }
    coalesce_free(chosen);

    const std::uint64_t id = next_id_++;
    arena.live.emplace(id, FreeRange{.offset = offset, .size = size});
    lru_.push_back(id);
    used_ += size;
    return Handle{
        .id = id, .arena = static_cast<std::int32_t>(chosen), .offset = offset, .size = size};
}

void ColdTier::free_entry(const Handle& handle) {
    if (handle.arena < 0 || static_cast<std::size_t>(handle.arena) >= arenas_.size()) {
        throw std::invalid_argument("cold tier handle names a missing arena");
    }
    Arena& arena  = arenas_[handle.arena];
    const auto it = arena.live.find(handle.id);
    if (it == arena.live.end()) { throw std::invalid_argument("cold tier handle is not live"); }
    arena.free.push_back(it->second);
    arena.live.erase(it);
    used_ -= handle.size;
    const auto lru_it = std::find(lru_.begin(), lru_.end(), handle.id);
    if (lru_it != lru_.end()) { lru_.erase(lru_it); }
    coalesce_free(static_cast<std::size_t>(handle.arena));
}

void ColdTier::touch(const Handle& handle) {
    if (handle.arena < 0 || static_cast<std::size_t>(handle.arena) >= arenas_.size() ||
        !arenas_[handle.arena].live.count(handle.id)) {
        throw std::invalid_argument("cold tier handle is not live");
    }
    const auto it = std::find(lru_.begin(), lru_.end(), handle.id);
    if (it == lru_.end()) { return; }
    lru_.erase(it);
    lru_.push_back(handle.id);
}

void* ColdTier::device_ptr(const Handle& handle) const {
    if (handle.arena < 0 || static_cast<std::size_t>(handle.arena) >= arenas_.size() ||
        !arenas_[handle.arena].live.count(handle.id)) {
        throw std::invalid_argument("cold tier handle is not live");
    }
    return static_cast<char*>(arenas_[handle.arena].ptr) + handle.offset;
}

int ColdTier::device_of(const Handle& handle) const {
    if (handle.arena < 0 || static_cast<std::size_t>(handle.arena) >= arenas_.size()) {
        throw std::invalid_argument("cold tier handle names a missing arena");
    }
    return arenas_[handle.arena].device;
}

ColdTier::Stats ColdTier::stats() const {
    return Stats{
        .arena_count       = arenas_.size(),
        .entry_count       = lru_.size(),
        .used_bytes        = used_,
        .evictions         = evictions_,
        .rejected_oversize = rejected_oversize_,
    };
}

void staged_span_transfer(int src_device, cudaStream_t src_stream,
                          std::span<const void* const> src_ptrs,
                          std::span<const std::size_t> src_sizes, int dst_device,
                          cudaStream_t dst_stream, std::span<void*> dst_ptrs,
                          std::span<const std::size_t> dst_sizes, void* staging,
                          std::size_t staging_bytes) {
    if (src_ptrs.size() != src_sizes.size() || dst_ptrs.size() != dst_sizes.size() ||
        src_ptrs.empty() || dst_ptrs.empty() || staging == nullptr || staging_bytes == 0) {
        throw std::invalid_argument("staged span transfer inputs are inconsistent");
    }
    const std::size_t total = span_total(src_sizes);
    if (total == 0 || total != span_total(dst_sizes)) {
        throw std::invalid_argument("staged span transfer totals disagree");
    }

    struct Piece {
        std::size_t begin  = 0; // Offset within the transfer.
        std::size_t length = 0;
        const void* src    = nullptr;
        void* dst          = nullptr;
    };

    std::vector<Piece> pieces;
    {
        std::size_t offset = 0;
        for (std::size_t i = 0; i < src_sizes.size(); ++i) {
            pieces.push_back(Piece{
                .begin = offset, .length = src_sizes[i], .src = src_ptrs[i], .dst = dst_ptrs[i]});
            offset += src_sizes[i];
        }
    }

    int saved_device = 0;
    check_cuda(cudaGetDevice(&saved_device), "staged transfer device query");
    try {
        std::size_t offset = 0;
        while (offset < total) {
            const std::size_t chunk = std::min(staging_bytes, total - offset);
            {
                DeviceScope scope(src_device);
                for (const Piece& piece : pieces) {
                    if (piece.begin + piece.length <= offset) { continue; }
                    if (piece.begin >= offset + chunk) { break; }
                    const std::size_t piece_begin = std::max(piece.begin, offset) - piece.begin;
                    const std::size_t piece_len   = std::min(
                        piece.length - piece_begin, offset + chunk - std::max(piece.begin, offset));
                    check_cuda(cudaMemcpyAsync(static_cast<char*>(staging) +
                                                   (std::max(piece.begin, offset) - offset),
                                               static_cast<const char*>(piece.src) + piece_begin,
                                               piece_len, cudaMemcpyDeviceToHost, src_stream),
                               "staged transfer device to host");
                }
                check_cuda(cudaStreamSynchronize(src_stream), "staged transfer source sync");
            }
            {
                DeviceScope scope(dst_device);
                for (const Piece& piece : pieces) {
                    if (piece.begin + piece.length <= offset) { continue; }
                    if (piece.begin >= offset + chunk) { break; }
                    const std::size_t piece_begin = std::max(piece.begin, offset) - piece.begin;
                    const std::size_t piece_len   = std::min(
                        piece.length - piece_begin, offset + chunk - std::max(piece.begin, offset));
                    check_cuda(cudaMemcpyAsync(static_cast<char*>(piece.dst) + piece_begin,
                                               static_cast<const char*>(staging) +
                                                   (std::max(piece.begin, offset) - offset),
                                               piece_len, cudaMemcpyHostToDevice, dst_stream),
                               "staged transfer host to device");
                }
                check_cuda(cudaStreamSynchronize(dst_stream), "staged transfer destination sync");
            }
            offset += chunk;
        }
    } catch (...) {
        if (saved_device != 0) {
            check_cuda(cudaSetDevice(saved_device), "staged transfer restore");
        }
        throw;
    }
    if (saved_device != 0) { check_cuda(cudaSetDevice(saved_device), "staged transfer restore"); }
}

} // namespace ninfer
