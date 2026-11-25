// STRATA_ADAPT_EARLY (C1): the flag's reading, the "can it run" rule and the one-window runner.  No GPU, no model.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

#include "strata/program/adapt_early.hpp"

using namespace strata::program::adapt_early;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

static Input base_on() {
    Input in;
    in.mode = Mode::On;
    in.remote_main = true;
    in.tier_on = true;
    in.serial_windows = true;
    return in;
}

int main() {
    // 1. the flag's reading
    CHECK(mode(nullptr) == Mode::Unset);
    CHECK(mode("") == Mode::Unset);
    CHECK(mode("0") == Mode::Off);
    CHECK(mode("x") == Mode::Off);
    CHECK(mode("1") == Mode::On);
    CHECK(mode("1abc") == Mode::On);

    // 2. off_reason
    {
        Input in;
        CHECK(off_reason(in) != nullptr);   // Unset
        in = base_on();
        in.mode = Mode::Off;
        CHECK(off_reason(in) != nullptr);
        in = base_on();
        CHECK(off_reason(in) == nullptr);
        bool Input::*const blockers[] = {&Input::local_split, &Input::helpers, &Input::adapt_async, &Input::pipeline_windows,
                                         &Input::pre_launch_hook, &Input::pin_blobs, &Input::lagged, &Input::resident_ram};
        for (auto m : blockers) {
            Input b = base_on();
            b.*m = true;
            CHECK(off_reason(b) != nullptr);
        }
        bool Input::*const needs[] = {&Input::remote_main, &Input::tier_on, &Input::serial_windows};
        for (auto m : needs) {
            Input b = base_on();
            b.*m = false;
            CHECK(off_reason(b) != nullptr);
        }
    }

    // 3. arm -> fire -> finish: once, on another thread
    {
        Runner r;
        int runs = 0;
        std::thread::id tid;
        r.arm([&] { ++runs; tid = std::this_thread::get_id(); return true; });
        r.fire();
        const auto res = r.finish();
        CHECK(runs == 1);
        CHECK(tid != std::this_thread::get_id());
        CHECK(res.armed && res.fired && res.ok);
        CHECK(res.host_ms >= 0);
    }
    // 4. armed, never fired
    {
        Runner r;
        int runs = 0;
        r.arm([&] { ++runs; return true; });
        const auto res = r.finish();
        CHECK(runs == 0);
        CHECK(res.armed && !res.fired && res.ok);
    }
    // 5. fire without an arm
    {
        Runner r;
        r.fire();
        const auto res = r.finish();
        CHECK(!res.armed && !res.fired && res.ok);
        r.arm(Runner::Job());   // an empty job is not armed either
        r.fire();
        CHECK(!r.finish().armed);
    }
    // 6. fire twice: one run
    {
        Runner r;
        std::atomic<int> runs{0};
        r.arm([&] { ++runs; return true; });
        r.fire();
        r.fire();
        r.finish();
        CHECK(runs == 1);
    }
    // 7. a job that fails
    {
        Runner r;
        r.arm([] { return false; });
        r.fire();
        const auto res = r.finish();
        CHECK(res.fired && !res.ok);
    }
    // 8. the Window's destructor waits for the job
    {
        Runner r;
        std::atomic<bool> done{false};
        {
            Window w(r, [&] {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                done = true;
                return true;
            });
            r.fire();
        }
        CHECK(done.load());
    }
    // 8b. Window::finish() once, then a default Result
    {
        Runner r;
        Window w(r, [] { return true; });
        r.fire();
        const auto a = w.finish();
        const auto b = w.finish();
        CHECK(a.fired && a.ok);
        CHECK(!b.armed && !b.fired);
    }
    // 9. 1,000 windows: every job ran once, no thread left over
    {
        Runner r;
        int runs = 0;
        for (int i = 0; i < 1000; ++i) {
            Window w(r, [&] { ++runs; return true; });
            r.fire();
            CHECK(w.finish().fired);
        }
        CHECK(runs == 1000);
    }
    // 10. fire() from another thread while nothing is armed (the --pipeline-windows shape): nothing happens
    {
        Runner r;
        std::thread t([&] { for (int i = 0; i < 10000; ++i) r.fire(); });
        for (int i = 0; i < 10000; ++i) r.fire();
        t.join();
        CHECK(!r.finish().armed);
    }
    // 11. Stats::window
    {
        Stats s;
        s.window(true, 30.0);
        s.window(true, 28.0);
        s.window(false, 27.0);
        s.window(false, 26.0);
        s.window(false, 25.0);
        CHECK(s.after_n == 2 && s.other_n == 3);
        CHECK(s.after_ms / (double) s.after_n == 29.0);
        CHECK(s.other_ms / (double) s.other_n == 26.0);
    }

    if (failed) {
        std::fprintf(stderr, "adapt_early_test: %d failed\n", failed);
        return 1;
    }
    std::printf("adapt_early_test: ok\n");
    return 0;
}
