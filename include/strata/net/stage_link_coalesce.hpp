#pragma once
// STRATA_LINK_COALESCE=1 (Arinko-inc patch, 2026-11-26): a stage link message goes out in one sendmsg (its header, its
// ids and its rows' first slice), and a worker reads a window's body (ids and rows) in one recvmsg.  The bytes on the
// wire are the ones send_msg + send_rows write (an fp8 / bf16 row is encoded on its own: slicing does not change it),
// so either side may run with or without it.  Off (unset, or anything but "1"): stage_link.cpp calls what it called.
#include "strata/net/stage_link.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <cerrno>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#endif

namespace strata::net::coalesce {

struct Part { const void* p; size_t n; };   ///< bytes to send; n == 0 is skipped
struct Sink { void* p; size_t n; };         ///< bytes to read into; n == 0 is skipped

namespace detail {
inline std::atomic<int>& override_flag() {
    static std::atomic<int> v{-1};
    return v;
}
inline bool env_on() {
    static const bool v = [] {
        const char* s = std::getenv("STRATA_LINK_COALESCE");
        if (!s) return false;
        std::string t(s);
        const size_t a = t.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) return false;
        const size_t b = t.find_last_not_of(" \t\r\n");
        return t.substr(a, b - a + 1) == "1";
    }();
    return v;
}
/// one call's most bytes: send_all / recv_all count in 1 GiB too
constexpr size_t kMaxCall = (size_t) 1 << 30;
}  // namespace detail

/// STRATA_LINK_COALESCE: exactly "1" (spaces around it allowed) turns it on, read once per process.  Always false on
/// Windows (no sendmsg there; the old calls stay).
inline bool on() {
#if defined(_WIN32)
    return false;
#else
    const int o = detail::override_flag().load(std::memory_order_relaxed);
    if (o >= 0) return o == 1;
    return detail::env_on();
#endif
}

/// Tests only: -1 = the environment (the default), 0 / 1 = off / on for this process from now on.
inline void force_for_test(int v) { detail::override_flag().store(v < 0 ? -1 : (v ? 1 : 0)); }

#if !defined(_WIN32)
namespace detail {
inline std::string errno_text(int e) { return std::strerror(e); }
}  // namespace detail

/// Every byte of parts[0..n) (n <= 8) in order through ::sendmsg(fd, ..., flags): a short write goes on from where
/// it stopped, EINTR is retried.  err: "remote stage: send failed: " + strerror(errno), as send_all writes it.
inline bool send_parts(int fd, int flags, const Part* parts, int n, std::string& err) {
    if (n < 0 || n > 8) { err = "remote stage: send failed: too many parts"; return false; }
    const char* ptr[8];
    size_t left[8];
    for (int i = 0; i < n; ++i) { ptr[i] = (const char*) parts[i].p; left[i] = parts[i].n; }
    int first = 0;
    for (;;) {
        while (first < n && left[first] == 0) ++first;
        if (first >= n) return true;
        iovec iov[8];
        int k = 0;
        size_t room = detail::kMaxCall;
        for (int i = first; i < n && room > 0; ++i) {
            if (left[i] == 0) continue;
            const size_t t = std::min(left[i], room);
            iov[k].iov_base = const_cast<char*>(ptr[i]);
            iov[k].iov_len = t;
            room -= t;
            ++k;
        }
        msghdr m{};
        m.msg_iov = iov;
        m.msg_iovlen = (decltype(m.msg_iovlen)) k;
        const ssize_t w = ::sendmsg(fd, &m, flags);
        if (w < 0) {
            const int e = errno;
            if (e == EINTR) continue;
            err = std::string("remote stage: send failed: ") + detail::errno_text(e);
            return false;
        }
        size_t d = (size_t) w;
        for (int i = first; i < n && d > 0; ++i) {
            const size_t t = std::min(left[i], d);
            ptr[i] += t;
            left[i] -= t;
            d -= t;
        }
    }
}

/// Every byte of sinks[0..n) (n <= 8) in order through ::recvmsg (a short read goes on, EINTR retried; no MSG_WAITALL,
/// so the socket's receive timeout counts as recv_all's does).
/// err as recv_all writes it: "remote stage: the peer closed the connection" (0 bytes) or
/// "remote stage: receive failed: " + ("timed out" for EAGAIN / EWOULDBLOCK, else strerror(errno)).
inline bool recv_parts(int fd, const Sink* sinks, int n, std::string& err) {
    if (n < 0 || n > 8) { err = "remote stage: receive failed: too many parts"; return false; }
    char* ptr[8];
    size_t left[8];
    for (int i = 0; i < n; ++i) { ptr[i] = (char*) sinks[i].p; left[i] = sinks[i].n; }
    int first = 0;
    for (;;) {
        while (first < n && left[first] == 0) ++first;
        if (first >= n) return true;
        iovec iov[8];
        int k = 0;
        size_t room = detail::kMaxCall;
        for (int i = first; i < n && room > 0; ++i) {
            if (left[i] == 0) continue;
            const size_t t = std::min(left[i], room);
            iov[k].iov_base = ptr[i];
            iov[k].iov_len = t;
            room -= t;
            ++k;
        }
        msghdr m{};
        m.msg_iov = iov;
        m.msg_iovlen = (decltype(m.msg_iovlen)) k;
        const ssize_t r = ::recvmsg(fd, &m, 0);
        if (r == 0) { err = "remote stage: the peer closed the connection"; return false; }
        if (r < 0) {
            const int e = errno;
            if (e == EINTR) continue;
            err = std::string("remote stage: receive failed: ") +
                  ((e == EAGAIN || e == EWOULDBLOCK) ? std::string("timed out") : detail::errno_text(e));
            return false;
        }
        size_t d = (size_t) r;
        for (int i = first; i < n && d > 0; ++i) {
            const size_t t = std::min(left[i], d);
            ptr[i] += t;
            left[i] -= t;
            d -= t;
        }
    }
}
#else
inline bool send_parts(int, int, const Part*, int, std::string& err) {
    err = "remote stage: coalesced sends are not built on Windows";
    return false;
}
inline bool recv_parts(int, const Sink*, int, std::string& err) {
    err = "remote stage: coalesced receives are not built on Windows";
    return false;
}
#endif

/// Header h, then p1 (n1 bytes), then `rows` rows of `width` floats in format f - the bytes send_msg(h, p1) and
/// send_rows(f, ...) write.  The first min(rows, step) rows are encoded into buf first and go out with h and p1 in
/// one send_parts; the rest go a slice of `step` rows at a time, as send_rows sends them.  fp32 rows go as they are
/// (one part).  conv_ms adds the encoding's time (steady_clock), as send_rows does.
inline bool send_msg_rows(int fd, int flags, const StageHeader& h, const void* p1, size_t n1, StageRows f,
                          const float* src, size_t rows, size_t width, size_t step, std::vector<unsigned char>& buf,
                          double& conv_ms, std::string& err) {
    if (f == StageRows::F32) {
        const Part ps[3] = {{&h, sizeof h}, {p1, n1}, {src, rows * width * sizeof(float)}};
        return send_parts(fd, flags, ps, 3, err);
    }
    step = std::max<size_t>(step, 1);
    const size_t first = std::min(rows, step);
    buf.resize(stage_rows_bytes(f, first, width));
    using Clk = std::chrono::steady_clock;
    size_t r = 0;
    for (bool head = true; head || r < rows; head = false) {
        const size_t k = std::min(step, rows - r);
        size_t bytes = 0;
        if (k > 0) {
            const Clk::time_point t = Clk::now();
            stage_rows_encode(f, src + r * width, k, width, buf.data());
            conv_ms += std::chrono::duration<double, std::milli>(Clk::now() - t).count();
            bytes = stage_rows_bytes(f, k, width);
        }
        bool ok;
        if (head) {
            const Part ps[3] = {{&h, sizeof h}, {p1, n1}, {buf.data(), bytes}};
            ok = send_parts(fd, flags, ps, 3, err);
        } else {
            const Part ps[1] = {{buf.data(), bytes}};
            ok = send_parts(fd, flags, ps, 1, err);
        }
        if (!ok) return false;
        r += k;
    }
    return true;
}

}  // namespace strata::net::coalesce
