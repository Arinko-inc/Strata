// sycl/src/core/verify_deferred.cpp - L1 (Arinko-inc): a remote stage worker's deferred windows on SYCL (the B70 end;
// strata-mesh IMPL-PLAN-2026-11-26-batch-drafts.md 3.5).  The CUDA tree's init_deferred_commit / capture_deferred /
// batch_commit_rows (src/core/verify.cpp "STRATA_PIPED_MTP: commits after the verdict") ported the SYCL way
// (dpct::experimental::begin_recording / end_recording, command_graph executables, copy_i32_from_mapped; the model is
// this tree's capture_commit_batch, with the saved rows dc_* in place of the window scratch), and
// run_slot_rows_deferred as the CUDA tree's (src/core/verify_deferred.cpp).
//
// What a deferred window is (strata-mesh 3.2/3.4): its window runs now with NO commit graph; right behind it the save
// graph copies the rows its commit replays (each GDN layer's conv input / delta-rule input / gate / beta, each QSA
// layer's indexer key, the PLE history) out of the window scratch - which the next window overwrites - into the saved
// rows [save_base, save_base + S); its commit, much later (batch_commit_rows, once the verdict is known), replays
// from the saved rows only the accepted prefix.  Only THIS stage's layers' rows are saved (the CUDA tree sizes them
// for the whole model; here the end has 13 GDN layers of 36).
//
// The skeleton (2026-11-26) fixed the declarations (sycl/include/strata/core/verify.hpp) and the two hooks in
// verify.cpp (stage_batch skips capture_commit_batch while staging_deferred_; ~Verifier calls free_deferred).
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/verify.hpp"

#include "strata/core/layout.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;   // (the same as verify.cpp's: the GDN output norm and the indexer key norm)

bool dfr_env_on(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr && e[0] != 0 && e[0] != '0';
}

// verify.cpp's mapped(): host USM the window's graphs read (uncached when the window polls it).  Its helpers are in
// verify.cpp's own anonymous namespace, so the same choice is made here (window_host_uncached()).
bool dfr_mapped(size_t bytes, void** h, void** d) {
    static const bool uncached = !dfr_env_on("STRATA_VERIFY_STEPPED") || dfr_env_on("STRATA_VERIFY_HOST_UNCACHED");
    *h = uncached ? strata::host_malloc_polled(bytes, dpct::get_in_order_queue())
                  : sycl::malloc_host(bytes, dpct::get_in_order_queue());
    if (*h == nullptr) return false;
    std::memset(*h, 0, bytes);
    *d = *h;
    return true;
}

const WeightRef* dfr_need(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* r = v.get(suffix);
    if (r == nullptr && err.empty()) err = v.name(suffix) + " is missing";
    return r;
}

}  // namespace

// this stage's layers [lb_, le_) (le_ < 0: to the last): the GDN / QSA layers before it and in it
namespace {
struct StageLayers {
    int64_t gdn0 = 0, qsa0 = 0;   // the stage's first GDN / QSA layer's global ordinal
    int64_t n_gdn = 0, n_qsa = 0;
};
StageLayers dfr_stage_layers(const ModelGeometry& g, int64_t lb, int64_t le) {
    StageLayers s;
    if (le < 0) le = g.n_layers;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        int64_t& before = is_qsa_layer(g, l) ? s.qsa0 : s.gdn0;
        int64_t& inside = is_qsa_layer(g, l) ? s.n_qsa : s.n_gdn;
        if (l < lb) ++before;
        else if (l < le) ++inside;
    }
    return s;
}
}  // namespace

bool Verifier::init_deferred_commit(int rows, std::string& err) try {
    const OnDevice on_device(device_);
    if (slots_.empty()) { err = "verify: init_deferred_commit before init_slots"; return false; }
    if (rows < 1) { err = "verify: init_deferred_commit needs at least one row"; return false; }
    if (dc_rows_ > 0) { err = "verify: init_deferred_commit twice"; return false; }
    const ModelGeometry& g = *g_;
    const int64_t le = le_ < 0 ? g.n_layers : le_;
    const StageLayers sl = dfr_stage_layers(g, lb_, le_);
    const uint64_t R = (uint64_t) rows, C = (uint64_t) g.ssm_conv_channels, HV = (uint64_t) g.ssm_v_heads;
    const uint64_t ID = (uint64_t) g.idx_key_dim;
    const uint64_t nG = (uint64_t) std::max<int64_t>(sl.n_gdn, 1), nQ = (uint64_t) std::max<int64_t>(sl.n_qsa, 1);
    const uint64_t HS = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
    const uint64_t CB = 2 + (uint64_t) max_t_, nS = (uint64_t) slots_.size();
    // the PLE history rows only on the stage that holds layer 1 (the end does not: one float stands in)
    const uint64_t hist_rows = ple_stage() ? R * HS : 1;
    const uint64_t floats[6] = {nG * R * C, nG * R * C, nG * R * HV, nG * R * HV, nQ * R * ID, hist_rows};
    uint64_t bytes = (nS * CB * 4 + 255) & ~255ull;
    for (uint64_t f : floats) bytes += (f * 4 + 255) & ~255ull;
    void* arena = nullptr;
    if (DPCT_CHECK_ERROR(arena = (void*) sycl::malloc_device(bytes, dpct::get_in_order_queue())) != 0 ||
        arena == nullptr) {
        err = "verify: the saved commit rows (" + std::to_string(bytes >> 20) + " MiB) do not fit";
        return false;
    }
    dc_arena_ = arena;
    uint8_t* p = (uint8_t*) dc_arena_;
    commitd_ = (int32_t*) p;
    p += (nS * CB * 4 + 255) & ~255ull;
    float** dst[6] = {&dc_qkv_, &dc_h_, &dc_gate_, &dc_beta_, &dc_idx_, &dc_hist_};
    for (int i = 0; i < 6; ++i) {
        *dst[i] = (float*) p;
        p += (floats[i] * 4 + 255) & ~255ull;
    }
    if (!dfr_mapped((size_t) (nS * CB * 4 + 16), (void**) &h_commitd_, (void**) &m_commitd_)) {
        sycl::free(dc_arena_, dpct::get_in_order_queue());
        dc_arena_ = nullptr;
        err = "verify: the deferred commit staging failed";
        return false;
    }
    dc_rows_ = rows;
    std::fprintf(stderr, "strata verify: deferred batch windows (layers [%lld, %lld): %lld GDN + %lld QSA, %d rows, "
                         "%.1f MiB saved rows)\n", (long long) lb_, (long long) le, (long long) sl.n_gdn,
                 (long long) sl.n_qsa, rows, (double) bytes / 1048576.0);
    return true;
}
catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

bool Verifier::capture_deferred(const int* rows, int S, int hbase, std::string& err) try {
    using namespace strata::kernels;
    const auto key = bkey(rows, S, hbase);
    // Graphs of layouts whose window graph (exec_bm_, keyed with hbase 0: run_slot_rows) the LRU limit evicted: drop
    // them, once, before making a new one (the window graph's own eviction waits for the queue; so does this).  A
    // layout still held by a window is the most recently used: not evicted unless 64 others came since.
    {
        std::vector<std::vector<int>> stale;
        auto window_key = [](std::vector<int> k) { k[0] = 0; return k; };
        for (const auto& kv : save_bm_)
            if (kv.first != key && exec_bm_.find(window_key(kv.first)) == exec_bm_.end()) stale.push_back(kv.first);
        for (const auto& kv : dcommit_bm_)
            if (kv.first != key && exec_bm_.find(window_key(kv.first)) == exec_bm_.end() &&
                std::find(stale.begin(), stale.end(), kv.first) == stale.end())
                stale.push_back(kv.first);
        if (!stale.empty()) {
            if (DPCT_CHECK_ERROR(cs_->wait()) != 0) {
                err = "verify: synchronizing before deferred graph eviction failed";
                return false;
            }
            for (const auto& k : stale) {
                auto s = save_bm_.find(k);
                if (s != save_bm_.end()) {
                    if (s->second) delete (s->second);
                    save_bm_.erase(s);
                }
                auto c = dcommit_bm_.find(k);
                if (c != dcommit_bm_.end()) {
                    if (c->second) delete (c->second);
                    dcommit_bm_.erase(c);
                }
            }
        }
    }
    dpct::experimental::command_graph_exec_ptr& sex = save_bm_[key];
    dpct::experimental::command_graph_exec_ptr& cex = dcommit_bm_[key];
    if (sex != nullptr && cex != nullptr) return true;
    const ModelGeometry& g = *g_;
    const QsaShapes s = qsa_shapes(g);
    const StageLayers sl = dfr_stage_layers(g, lb_, le_);
    const int64_t le = le_ < 0 ? g.n_layers : le_;
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ZV = g.ssm_value_dim, ID = g.idx_key_dim, MT = max_t_;
    const int64_t R = dc_rows_, CB = 2 + MT, nQ = g.n_qsa_layers();
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const bool ple = ss_->ple.ready() && ple_stage();
    auto finish = [&](dpct::experimental::command_graph_exec_ptr& ex, bool ok, const char* what) -> bool {
        dpct::experimental::command_graph_ptr graph = nullptr;
        const dpct::err0 ce = DPCT_CHECK_ERROR(dpct::experimental::end_recording(cs_, &graph));
        if (!ok || ce != 0) {
            if (graph) delete (graph);
            if (ok) err = std::string(what) + dpct::get_error_string_dummy(ce);
            return false;
        }
        try {
            ex = new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(
                graph->finalize());
        } catch (const sycl::exception& x) {
            delete (graph);
            err = std::string(what) + x.what();
            return false;
        }
        delete (graph);
        return true;
    };
    // the save: the window's rows [0, S) of the scratch the commit replays -> this stage's saved rows [hbase, +S)
    if (sex == nullptr) {
        if (DPCT_CHECK_ERROR(dpct::experimental::begin_recording(cs_)) != 0) {
            err = "verify: begin save capture failed";
            return false;
        }
        bool ok = true;
        try {
            int64_t qsa_index = sl.qsa0, gdn_index = sl.gdn0;   // (the scratch is indexed by the global ordinal)
            auto copy = [&](float* dst, const float* src, int64_t n) { cs_->memcpy(dst, src, (size_t) n * sizeof(float)); };
            for (int64_t l = lb_; l < le; ++l) {
                if (!is_qsa_layer(g, l)) {
                    const int64_t gr = gdn_index - sl.gdn0;   // (the saved rows: this stage's layers only)
                    copy(dc_qkv_ + (size_t) (gr * R + hbase) * C, qkv_L_ + (size_t) gdn_index * MT * C, S * C);
                    copy(dc_h_ + (size_t) (gr * R + hbase) * C, h_L_ + (size_t) gdn_index * MT * C, S * C);
                    copy(dc_gate_ + (size_t) (gr * R + hbase) * HV, gate_L_ + (size_t) gdn_index * MT * HV, S * HV);
                    copy(dc_beta_ + (size_t) (gr * R + hbase) * HV, beta_L_ + (size_t) gdn_index * MT * HV, S * HV);
                    ++gdn_index;
                } else {
                    const int64_t qr = qsa_index - sl.qsa0;
                    copy(dc_idx_ + (size_t) (qr * R + hbase) * ID, idx_raw_L_ + (size_t) qsa_index * MT * ID, S * ID);
                    ++qsa_index;
                }
            }
            if (ple) copy(dc_hist_ + (size_t) hbase * HS, hist_snap_, S * HS);
        } catch (const std::exception& e) {
            err = std::string("verify deferred save: ") + e.what();
            ok = false;
        }
        if (!finish(sex, ok, "verify: save capture: ")) {
            if (ok && err.empty()) err = "verify: save capture failed";
            return false;
        }
    }
    // the commit: as capture_commit_batch, replaying the saved rows; the accepted prefixes come from commitd_
    if (cex == nullptr) {
        if (DPCT_CHECK_ERROR(dpct::experimental::begin_recording(cs_)) != 0) {
            err = "verify: begin deferred commit capture failed";
            return false;
        }
        bool ok = true;
        try {
            for (int t = 0; t < S; ++t)
                if (t == 0 || rows[t] != rows[t - 1])
                    copy_i32_from_mapped(commitd_ + (size_t) rows[t] * CB, m_commitd_ + (size_t) rows[t] * CB, CB, cs_);
            int64_t qsa_index = sl.qsa0, gdn_index = sl.gdn0;
            for (int64_t l = lb_; l < le && ok; ++l) {
                const LayerView v(*wt_, l);
                if (!is_qsa_layer(g, l)) {
                    const WeightRef* wnm = dfr_need(v, "ssm_norm.weight", err);
                    if (!wnm) { ok = false; break; }
                    for (int t = 0; t < S;) {
                        const int first = t;
                        while (t < S && rows[t] == rows[first]) ++t;
                        SessionState& sx = *slots_[(size_t) rows[first]];
                        const int32_t* keep = commitd_ + (size_t) rows[first] * CB;
                        float* state = sx.gdn_state + (size_t) (gdn_index - sx.gdn_ord0) * gdn_floats;
                        float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                        const int64_t row = (gdn_index - sl.gdn0) * R + hbase + first;
                        gdn_conv_commit(conv, dc_qkv_ + (size_t) row * C, (int) C, keep, cs_);
                        // z only feeds the norm's output, which the commit discards (y_dummy_): the window scratch will do
                        gdn_step_norm_multi(state, dc_h_ + (size_t) row * C, (int) C, dc_gate_ + (size_t) row * HV,
                                            dc_beta_ + (size_t) row * HV, z_ + (size_t) first * ZV,
                                            (const float*) wnm->data, EPS, y_dummy_ + (size_t) first * ZV,
                                            (int) g.ssm_k_heads, (int) HV, t - first, keep, cs_,
                                            t - first > 1 ? t - first : 0);   // a 1-row group keeps the 0.1.39 kernel
                    }
                    ++gdn_index;
                } else {
                    const WeightRef* wikn = dfr_need(v, "indexer.k_norm.weight", err);
                    if (!wikn) { ok = false; break; }
                    for (int t = 0; t < S;) {
                        const int first = t;
                        while (t < S && rows[t] == rows[first]) ++t;
                        const QsaState& st = slots_[(size_t) rows[first]]->qsa_states[qsa_index];
                        copy_from_mapped(st.idx_tail, tail_snap_b_ + ((size_t) rows[first] * nQ + qsa_index) * TS, TS, cs_);
                        const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                        for (int u = first; u < t; ++u)
                            native_qsa_indexer_append(dc_idx_ + (size_t) ((qsa_index - sl.qsa0) * R + hbase + u) * ID,
                                                      commitd_ + (size_t) rows[first] * CB + 2 + (u - first),
                                                      slot_pos_base(rows[first]), (const float*) wikn->data, EPS, ib, s,
                                                      st.max_cells, rope_scaling(), cs_);
                    }
                    ++qsa_index;
                }
            }
            if (ok && ple)
                for (int t = 0; t < S;) {
                    const int first = t;
                    while (t < S && rows[t] == rows[first]) ++t;
                    copy_indexed(slots_[(size_t) rows[first]]->ple_hist, dc_hist_ + (size_t) (hbase + first) * HS, HS,
                                 commitd_ + (size_t) rows[first] * CB + 1, HS, cs_);
                }
        } catch (const std::exception& e) {
            err = std::string("verify deferred commit: ") + e.what();
            ok = false;
        }
        if (!finish(cex, ok, "verify: deferred commit capture: ")) {
            if (ok && err.empty()) err = "verify: deferred commit capture failed";
            return false;
        }
    }
    // (how many layouts are held: a worker's log shows it; each is two executable graphs)
    std::fprintf(stderr, "strata verify: captured the deferred window's save and commit graphs over slots %d..%d "
                         "(%zu layouts held)\n", rows[0], rows[S - 1], save_bm_.size());
    return true;
}
catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

bool Verifier::run_slot_rows_deferred(const int* rows, int S, int save_base, const int32_t* tokens, const int64_t* pos,
                                      PoolMultiFn pool, void* user, int32_t* out, std::string& err) try {
    const OnDevice on_device(device_);
    if (dc_rows_ < 1) { err = "verify: run_slot_rows_deferred without init_deferred_commit"; return false; }
    if (S < 1 || S > max_t_ || save_base < 0 || save_base + S > dc_rows_) {
        err = "verify: a deferred window outside the saved rows";
        return false;
    }
    // the window without its commit graph (stage_batch skips capture_commit_batch while staging_deferred_), over the
    // hand-off rows [0, S); it returns with the window finished on the GPU and its rows in the hand-off
    staging_deferred_ = true;
    bool ran = false;
    try {
        ran = run_slot_rows(rows, S, tokens, pos, pool, user, out, err);
    } catch (...) {
        staging_deferred_ = false;
        throw;
    }
    staging_deferred_ = false;
    if (!ran) return false;
    if (!capture_deferred(rows, S, save_base, err)) return false;
    // the rows the commit replays, kept for it: queued behind the window, not waited for - the queue orders it before
    // this stage's next window (which overwrites the scratch) and before the commit
    const auto it = save_bm_.find(bkey(rows, S, save_base));
    if (it == save_bm_.end() || it->second == nullptr) { err = "verify: a deferred window lost its save graph"; return false; }
    const dpct::err0 le = DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph(*it->second));
    if (le != 0) {
        err = std::string("verify: deferred save launch: ") + dpct::get_error_string_dummy(le);
        return false;
    }
    return true;
}
catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

bool Verifier::batch_commit_rows(const int* rows, int S, int hbase, const int* keep, const int32_t* tokens,
                                 const int64_t* pos, std::string& err) try {
    const OnDevice on_device(device_);
    auto it = dcommit_bm_.find(bkey(rows, S, hbase));
    if (it == dcommit_bm_.end() || it->second == nullptr) {
        // 合流 D: the doorbell variant (#871: ar_off_, the key's last entry) flipped between the window and this commit
        // (refresh_ar).  The commit reads no doorbell: the other variant's graph, captured with the window, is the
        // same commit (the CUDA tree's batch_commit_rows does the same)
        std::vector<int> other = bkey(rows, S, hbase);
        other.back() = other.back() != 0 ? 0 : 1;
        it = dcommit_bm_.find(other);
    }
    if (it == dcommit_bm_.end() || it->second == nullptr) { err = "verify: a deferred commit without its window"; return false; }
    const int64_t CB = 2 + max_t_;
    // Each slot's accepted prefix.  The graph reads it from mapped memory when it runs; this slot's entry is written
    // again only at its next verdict, which comes after its next window has passed this stage - behind this commit.
    for (int t = 0; t < S;) {
        const int first = t;
        while (t < S && rows[t] == rows[first]) ++t;
        const int n = keep[rows[first]];
        if (n < 1 || n > t - first) { err = "verify: accepted prefix is outside its slot group"; return false; }
        int32_t* c = h_commitd_ + (size_t) rows[first] * CB;
        c[0] = n;
        c[1] = n - 1;
        for (int64_t j = 0; j < max_t_; ++j) c[2 + j] = j < n ? (int32_t) pos[first + j] : -1;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const dpct::err0 le = DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph(*it->second));
    if (le != 0) {
        err = std::string("verify: deferred commit launch: ") + dpct::get_error_string_dummy(le);
        return false;
    }
    if (ple_stage())   // the host's side of the commit (the hash's last two tokens), the accepted rows only
        for (int t = 0; t < S;) {
            const int first = t;
            while (t < S && rows[t] == rows[first]) ++t;
            SessionState& sx = *slots_[(size_t) rows[first]];
            for (int u = first; u < first + keep[rows[first]]; ++u) {
                sx.ple_prev[0] = sx.ple_prev[1];
                sx.ple_prev[1] = tokens[u];
            }
        }
    return true;
}
catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void Verifier::free_deferred() {
    // (best-effort, as ~Verifier's other frees: it runs in a destructor)
    try {
        for (auto& kv : save_bm_)
            if (kv.second) delete (kv.second);
        for (auto& kv : dcommit_bm_)
            if (kv.second) delete (kv.second);
        save_bm_.clear();
        dcommit_bm_.clear();
        if (dc_arena_) sycl::free(dc_arena_, dpct::get_in_order_queue());
        if (h_commitd_) sycl::free(h_commitd_, dpct::get_in_order_queue());
    } catch (...) {
    }
    dc_arena_ = nullptr;
    h_commitd_ = m_commitd_ = commitd_ = nullptr;
    dc_qkv_ = dc_h_ = dc_gate_ = dc_beta_ = dc_idx_ = dc_hist_ = nullptr;
    dc_rows_ = 0;
}

}  // namespace strata::core
