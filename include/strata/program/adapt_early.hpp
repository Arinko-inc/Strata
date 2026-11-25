// STRATA_ADAPT_EARLY (C1 of IMPL-PLAN-2026-11-26-main-small.md): the adaptive tier's round of one window, started when
// the main process of a remote stage sends its rows to the relay (instead of at the commit, after the head and the
// sampling).  The round picks the same experts from the same usage counts (stage 0's layers have counted this window's
// routing when the rows go out; the head, the sampling, the commit and the draft count nothing), so it changes WHEN the
// host-to-device copies start, not what is copied.  The copies then run while the relay and the far stage compute.
//
//   unset: the same as before (no thread, no log line)
//   =0:    the same as before, and the decode timing line also tells the windows after a round from the others
//   =1:    on (when off_reason() is nullptr; else the startup log says why not and the windows run as before)
//
// Header only; no GPU, no environment reads, no platform headers (tests/program/adapt_early_test.cpp).
#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <utility>

namespace strata::program::adapt_early {

enum class Mode { Unset, Off, On };

/// STRATA_ADAPT_EARLY's value: nullptr or "" -> Unset, a value starting with '1' -> On, anything else -> Off
inline Mode mode(const char* value) {
    if (value == nullptr || value[0] == '\0') return Mode::Unset;
    return value[0] == '1' ? Mode::On : Mode::Off;
}

/// What generate.cpp knows at start (all false = the default)
struct Input {
    Mode mode = Mode::Unset;
    bool remote_main = false;       ///< --remote-stage (the main process of a remote stage)
    bool tier_on = false;           ///< !drive.d.usage.empty()
    bool serial_windows = false;    ///< generate.cpp's serial_windows (!pipe && !piped_groups)
    bool local_split = false;       ///< multi_gpu || !stages.empty()
    bool helpers = false;           ///< peer.valid() || remote_opt || drive.d.remote_count > 0
    bool adapt_async = false;       ///< o.adapt_async
    bool pipeline_windows = false;  ///< o.pipeline_windows > 0
    bool pre_launch_hook = false;   ///< drive.d.fetch_admit > 0 || drive.d.evict_sync || res_checking
    bool pin_blobs = false;         ///< pin_blobs_on()
    bool lagged = false;            ///< adapt_lag() != 1 || adapt_nowait()
    bool resident_ram = false;      ///< src.complement_ready()
};

/// nullptr: the early start can run; else why not (a short English phrase for the startup log).
/// Mode Unset / Off -> "STRATA_ADAPT_EARLY is not 1".  Checked in the order of the fields above.
inline const char* off_reason(const Input& in) {
    if (in.mode != Mode::On) return "STRATA_ADAPT_EARLY is not 1";
    if (!in.remote_main) return "only on the main process of --remote-stage";
    if (!in.tier_on) return "the adaptive tier is off";
    if (!in.serial_windows) return "not with pipelined slot groups";
    if (in.local_split) return "not on a layer split";
    if (in.helpers) return "not with helper GPUs (--peer-device, --remote-expert-opt)";
    if (in.adapt_async) return "not with --adapt-async";
    if (in.pipeline_windows) return "not with --pipeline-windows";
    if (in.pre_launch_hook) return "not with STRATA_ADAPT_FETCH, delayed evictions or STRATA_ADAPT_CHECKRES";
    if (in.pin_blobs) return "not with page-locked swap sources";
    if (in.lagged) return "not with STRATA_ADAPT_LAG / STRATA_ADAPT_NOWAIT";
    if (in.resident_ram) return "not in the resident RAM mode";
    return nullptr;
}

/// One window's adaptive round started on a thread of its own when the rows go to the remote stage.
/// arm / fire / finish are called on ONE thread (the decode loop's; fire() from the remote-run function that loop's
/// Verifier::run calls).  fire() from any other thread is a no-op unless armed (armed_ is atomic).
class Runner {
public:
    using Job = std::function<bool()>;
    struct Result {
        bool armed = false;    ///< a job was armed for this window
        bool fired = false;    ///< it ran (started by fire())
        bool ok = true;        ///< its return value (true when it did not run)
        double host_ms = 0;    ///< the job's own time on its thread
        double wait_ms = 0;    ///< how long finish() waited for it
    };
    Runner() = default;
    ~Runner() { join_(); }                  ///< joins a job still running (never expected)
    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;

    /// empty job: not armed.  Arming while armed replaces the job (not fired)
    void arm(Job job) {
        join_();
        if (!job) {
            armed_.store(false, std::memory_order_release);
            return;
        }
        job_ = std::move(job);
        fired_ = false;
        ok_ = true;
        host_ms_ = 0;
        armed_.store(true, std::memory_order_release);
    }

    /// starts the armed job once (a std::thread); a no-op if not armed or fired
    void fire() {
        if (!armed_.load(std::memory_order_acquire) || fired_) return;
        try {
            th_ = std::thread([this] {
                const auto t0 = Clock::now();
                ok_ = job_();
                host_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            });
            fired_ = true;
        } catch (...) {
            fired_ = false;   // the thread could not start: the caller's own path (the commit's round) takes this window
        }
    }

    /// joins, disarms, returns the window's result; a no-op Result if not armed
    Result finish() {
        Result r;
        if (!armed_.load(std::memory_order_acquire)) return r;
        r.armed = true;
        if (th_.joinable()) {
            const auto t0 = Clock::now();
            th_.join();
            r.wait_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        }
        r.fired = fired_;
        r.ok = ok_;
        r.host_ms = host_ms_;
        armed_.store(false, std::memory_order_release);
        fired_ = false;
        job_ = nullptr;
        return r;
    }

private:
    using Clock = std::chrono::steady_clock;
    void join_() {
        if (th_.joinable()) th_.join();
    }
    std::atomic<bool> armed_{false};
    bool fired_ = false;
    Job job_;
    std::thread th_;
    bool ok_ = true;                        ///< written by the job's thread, read after join
    double host_ms_ = 0;                    ///< likewise
};

/// RAII for one window: arms in the constructor, finishes in the destructor if finish() was not called (every return path)
class Window {
public:
    Window(Runner& r, Runner::Job job) : r_(r) { r_.arm(std::move(job)); }
    ~Window() {
        if (!done_) (void) r_.finish();
    }
    /// once; later calls return a default Result
    Runner::Result finish() {
        if (done_) return {};
        done_ = true;
        return r_.finish();
    }
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

private:
    Runner& r_;
    bool done_ = false;
};

/// The decode timing line's counters (Mode Off and On)
struct Stats {
    long long early = 0;        ///< rounds started at the remote send
    long long late = 0;         ///< rounds due while on but started at the commit (the window never reached the send)
    double host_ms = 0, wait_ms = 0;
    double after_ms = 0, other_ms = 0;   ///< window time (round0 -> after the adapt join): the window after a round / the rest
    long long after_n = 0, other_n = 0;
    void window(bool after_round, double ms) {
        if (after_round) {
            after_ms += ms;
            ++after_n;
        } else {
            other_ms += ms;
            ++other_n;
        }
    }
};

}  // namespace strata::program::adapt_early
