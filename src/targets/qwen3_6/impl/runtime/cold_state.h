#pragma once

// Cross-GPU cold state cache for one family Program: when a retained lane is
// evicted, its serialized state image (paged KV, GDN slots, hidden rows,
// prefix identity) is swapped into the ColdTier arena set on secondary
// devices; when a later request presents the same prefix, the image is
// swapped back and the request continues from the parked frontier instead of
// re-prefilling.

#include "core/cold_tier.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

class ColdStateCache {
public:
    // One parked lane state image plus the host-side metadata needed to match
    // a later prompt against it without touching the arena.
    struct Entry {
        std::uint64_t id            = 0;
        ColdTier::Handle handle     = {};
        std::vector<TokenId> ledger;
        qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
        std::int32_t rope_delta     = 0;
        std::uint32_t execution_frontier = 0;
        std::uint32_t ledger_frontier    = 0;
        std::uint32_t text_kv_valid      = 0;
        std::uint32_t mtp_kv_valid       = 0;
        std::vector<std::int32_t> text_pages;
        std::vector<std::int32_t> backend_pages;
        bool tail_hidden_valid = false;
        RewriteCheckpoint rewrite_checkpoint;
        std::size_t image_bytes = 0;
        std::chrono::steady_clock::time_point parked_at;
    };

    struct Stats {
        std::uint64_t parks         = 0;
        std::uint64_t restores      = 0;
        std::uint64_t park_failures = 0;
        ColdTier::Stats tier;
    };

    explicit ColdStateCache(ColdTier tier);

    // Serializes the retained lane into the arena set. Never throws; returns
    // false when the lane is not parkable or the image does not fit.
    bool park_lane(const ProgramImplCore& program, std::uint32_t lane);

    // Id of the parked entry whose full ledger prefix-matches the prompt, or
    // 0. Longest match wins; ties resolve to the most recently parked entry.
    [[nodiscard]] std::uint64_t find_parked(const PreparedPromptData& prompt) const;

    // Swaps the entry image back into an empty lane and re-establishes the
    // retained sequence state so the planner sees a resident prefix.
    // Throws on an invalid lane, allocation failure, or transfer error.
    void restore_parked(std::uint64_t entry_id, ProgramImplCore& program, std::uint32_t lane);

    [[nodiscard]] Stats stats() const;

private:
    struct ImageSpan {
        const void* device_ptr = nullptr;
        std::size_t bytes      = 0;
    };

    std::vector<ImageSpan> image_spans_for_lane(const ProgramImplCore& program,
                                                std::uint32_t lane) const;

    mutable std::mutex mutex_;
    ColdTier tier_;
    std::vector<Entry> entries_;
    std::uint64_t parks_         = 0;
    std::uint64_t restores_      = 0;
    std::uint64_t park_failures_ = 0;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
