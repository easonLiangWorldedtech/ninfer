#include "targets/qwen3_6/impl/runtime/cold_state.h"

#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

struct SpanSet {
    std::vector<const void*> ptrs;
    std::vector<std::size_t> sizes;
    std::size_t total = 0;
};

SpanSet to_span_set(const std::vector<ColdStateCache::ImageSpan>& spans) {
    SpanSet out;
    out.ptrs.reserve(spans.size());
    out.sizes.reserve(spans.size());
    for (const auto& span : spans) {
        if (span.device_ptr == nullptr || span.bytes == 0) {
            throw std::logic_error("cold cache image span is empty");
        }
        out.ptrs.push_back(span.device_ptr);
        out.sizes.push_back(span.bytes);
        out.total += span.bytes;
    }
    return out;
}

std::vector<std::int32_t> page_vector(std::span<const std::int32_t> pages) {
    return {pages.begin(), pages.end()};
}

} // namespace

ColdStateCache::ColdStateCache(ColdTier tier) : tier_(std::move(tier)) {}

std::vector<ColdStateCache::ImageSpan>
ColdStateCache::image_spans_for_lane(const ProgramImplCore& program, std::uint32_t lane) const {
    const SequenceState& sequence = program.sequences[lane];
    std::vector<ImageSpan> spans;
    if (!sequence.kv) { throw std::logic_error("cold cache lane has no KV allocation bundle"); }

    const auto collect_pool = [&](const qwen3_6::PagedKVCache& cache,
                                  std::span<const std::int32_t> pages) {
        const PagedKVPool& pool = cache.pool();
        for (std::size_t plane_index = 0; plane_index < pool.plane_count(); ++plane_index) {
            const Tensor& plane = pool.plane(plane_index);
            if (plane.nb[3] <= 0) { throw std::logic_error("cold cache plane is not page-major"); }
            const std::size_t stride = static_cast<std::size_t>(plane.nb[3]);
            const char* base         = static_cast<const char*>(plane.data);
            for (std::int32_t page : pages) {
                if (page < 0 || page >= static_cast<std::int32_t>(pool.page_group_count())) {
                    throw std::out_of_range("cold cache page id is out of range");
                }
                spans.push_back({base + static_cast<std::size_t>(page) * stride, stride});
            }
        }
    };
    collect_pool(program.decoder->text_kv, sequence.kv->text.page_ids());
    if (program.speculative_backend == SpeculativeBackend::Mtp && sequence.kv->backend) {
        collect_pool(*program.decoder->mtp_cache(), sequence.kv->backend->page_ids());
    }

    const LinearAttentionStatePool& linear = program.decoder->linear_attention;
    const auto append_slot                 = [&](std::int32_t slot) {
        for (std::uint32_t layer = 0; layer < linear.layer_count(); ++layer) {
            const Tensor conv      = linear.conv_slot(layer, slot);
            const Tensor recurrent = linear.recurrent_slot(layer, slot);
            spans.push_back({conv.data, conv.bytes()});
            spans.push_back({recurrent.data, recurrent.bytes()});
        }
    };
    append_slot(LinearStateSlots::current_state_slot(lane, program.max_concurrency));
    if (sequence.rewrite_checkpoint.valid) {
        append_slot(LinearStateSlots::rewrite_checkpoint_state_slot(lane, program.max_concurrency));
    }
    if (sequence.tail_hidden_valid) {
        spans.push_back({sequence.tail_hidden.data, sequence.tail_hidden.bytes()});
    }
    if (sequence.rewrite_checkpoint.valid) {
        spans.push_back(
            {sequence.rewrite_checkpoint_hidden.data, sequence.rewrite_checkpoint_hidden.bytes()});
    }
    return spans;
}

bool ColdStateCache::park_lane(const ProgramImplCore& program, std::uint32_t lane) {
    try {
        std::lock_guard lock(mutex_);
        if (lane >= program.max_concurrency) { return false; }
        const SequenceState& sequence = program.sequences[lane];
        if (!sequence.retained || !sequence.kv || sequence.execution_frontier == 0 ||
            sequence.ledger.empty() || !sequence.tail_hidden_valid) {
            return false;
        }
        if (program.speculative_backend == SpeculativeBackend::DFlash) { return false; }
        const std::size_t prefix_count = sequence.ledger.size();
        if (prefix_count != sequence.execution_frontier ||
            prefix_count != sequence.ledger_frontier) {
            return false;
        }
        if (program.speculative_backend == SpeculativeBackend::Mtp &&
            (!sequence.kv->backend || sequence.kv->backend->page_ids().empty() ||
             sequence.mtp_kv_valid + 1 < sequence.execution_frontier)) {
            return false;
        }

        const std::vector<ImageSpan> spans = image_spans_for_lane(program, lane);
        const SpanSet source               = to_span_set(spans);

        Entry entry;
        entry.ledger             = sequence.ledger;
        entry.prefix_identity    = sequence.prefix_identity;
        entry.rope_delta         = sequence.rope_delta;
        entry.execution_frontier = sequence.execution_frontier;
        entry.ledger_frontier    = sequence.ledger_frontier;
        entry.text_kv_valid      = sequence.text_kv_valid;
        entry.mtp_kv_valid       = sequence.mtp_kv_valid;
        entry.text_pages         = page_vector(sequence.kv->text.page_ids());
        if (sequence.kv->backend) {
            entry.backend_pages = page_vector(sequence.kv->backend->page_ids());
        }
        entry.tail_hidden_valid  = sequence.tail_hidden_valid;
        entry.rewrite_checkpoint = sequence.rewrite_checkpoint;
        entry.image_bytes        = source.total;
        entry.parked_at          = std::chrono::steady_clock::now();

        std::vector<std::uint64_t> evicted_ids;
        std::optional<ColdTier::Handle> handle = tier_.allocate(source.total, &evicted_ids);
        if (!handle) { return false; }
        for (std::uint64_t evicted : evicted_ids) {
            const auto it = std::find_if(entries_.begin(), entries_.end(),
                                         [evicted](const Entry& e) { return e.id == evicted; });
            if (it != entries_.end()) { entries_.erase(it); }
        }

        std::vector<const void*> destination_ptrs{tier_.device_ptr(*handle)};
        try {
            staged_span_transfer(
                program.device.device, program.device.stream,
                std::span<const const void*>(source.ptrs.data(), source.ptrs.size()),
                std::span<const std::size_t>(source.sizes.data(), source.sizes.size()),
                program.device.device, program.device.stream,
                std::span<const void*>(destination_ptrs.data(), destination_ptrs.size()),
                std::span<const std::size_t>{handle->size}, tier_.staging(), tier_.staging_bytes());
        } catch (...) {
            tier_.free_entry(*handle);
            throw;
        }
        entry.id     = handle->id;
        entry.handle = *handle;
        entries_.push_back(std::move(entry));
        ++parks_;
        return true;
    } catch (...) {
        std::lock_guard lock(mutex_);
        ++park_failures_;
        return false;
    }
}

std::uint64_t ColdStateCache::find_parked(const PreparedPromptData& prompt) const {
    std::lock_guard lock(mutex_);
    if (!prompt.identity.reusable || prompt.token_ids.empty()) { return 0; }
    const Entry* best = nullptr;
    for (const Entry& entry : entries_) {
        if (entry.ledger.empty() || entry.ledger.size() > prompt.token_ids.size()) { continue; }
        if (!qwen3_6::detail::prefix_matches(prompt, entry.ledger, entry.prefix_identity,
                                             entry.ledger.size())) {
            continue;
        }
        if (best == nullptr || entry.ledger.size() > best->ledger.size() ||
            (entry.ledger.size() == best->ledger.size() && entry.parked_at > best->parked_at)) {
            best = &entry;
        }
    }
    return best != nullptr ? best->id : 0;
}

void ColdStateCache::restore_parked(std::uint64_t entry_id, ProgramImplCore& program,
                                    std::uint32_t lane) {
    if (lane >= program.max_concurrency) {
        throw std::out_of_range("cold cache restore lane is out of range");
    }
    std::lock_guard lock(mutex_);
    const auto found = std::find_if(entries_.begin(), entries_.end(),
                                    [entry_id](const Entry& e) { return e.id == entry_id; });
    if (found == entries_.end()) { throw std::invalid_argument("cold cache entry is not parked"); }
    const Entry parked = *found;

    SequenceState& sequence = program.sequences[lane];
    RequestControl& request = program.requests[lane];
    if (request.lifecycle != Lifecycle::Empty || sequence.retained || sequence.kv) {
        throw std::logic_error("cold cache restore requires an empty lane");
    }

    // Fresh allocation with the parked mapped page count, materialized so the
    // physical pages are addressable before the image is swapped back.
    program.reserve_sequence_kv(sequence, static_cast<std::uint32_t>(parked.text_pages.size()),
                                static_cast<std::uint32_t>(parked.backend_pages.size()));
    program.materialize_sequence_kv(sequence, parked.text_kv_valid,
                                    parked.mtp_kv_valid > 0 ? parked.mtp_kv_valid : 0U);

    if (sequence.kv->text.page_ids().size() != parked.text_pages.size() ||
        (sequence.kv->backend ? sequence.kv->backend->page_ids().size() : 0U) !=
            parked.backend_pages.size()) {
        sequence.kv.reset();
        throw std::logic_error("cold cache restore allocation does not match the parked image");
    }
    const std::vector<ImageSpan> spans = image_spans_for_lane(program, lane);
    SpanSet destination                = to_span_set(spans);
    if (destination.total != parked.image_bytes) {
        sequence.kv.reset();
        throw std::logic_error("cold cache restore image size disagrees with the parked entry");
    }
    std::vector<const void*> source_ptrs{tier_.device_ptr(parked.handle)};
    try {
        staged_span_transfer(
            tier_.device_of(parked.handle), tier_.staging_stream(),
            std::span<const const void*>(source_ptrs.data(), source_ptrs.size()),
            std::span<const std::size_t>{parked.handle.size}, program.device.device,
            program.device.stream,
            std::span<const void*>(destination.ptrs.data(), destination.ptrs.size()),
            std::span<const std::size_t>(destination.sizes.data(), destination.sizes.size()),
            tier_.staging(), tier_.staging_bytes());
    } catch (...) {
        sequence.kv.reset();
        throw;
    }

    sequence.lane                    = lane;
    sequence.ledger                  = parked.ledger;
    sequence.ledger_frontier         = parked.ledger_frontier;
    sequence.execution_frontier      = parked.execution_frontier;
    sequence.prefix_identity         = parked.prefix_identity;
    sequence.rope_delta              = parked.rope_delta;
    sequence.text_kv_valid           = parked.text_kv_valid;
    sequence.mtp_kv_valid            = parked.mtp_kv_valid;
    sequence.dflash_context_frontier = 0;
    sequence.mtp_draft_count         = 0;
    sequence.tail_hidden_valid       = parked.tail_hidden_valid;
    sequence.rewrite_checkpoint      = parked.rewrite_checkpoint;
    sequence.retained                = true;

    tier_.free_entry(parked.handle);
    entries_.erase(found);
    ++restores_;
}

ColdStateCache::Stats ColdStateCache::stats() const {
    std::lock_guard lock(mutex_);
    return Stats{
        .parks         = parks_,
        .restores      = restores_,
        .park_failures = park_failures_,
        .tier          = tier_.stats(),
    };
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
