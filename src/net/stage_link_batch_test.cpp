// src/net/stage_link_batch_test.cpp - protocol 8's batch slots over the remote stage's link, on loopback, no GPU
// (strata-mesh IMPL-PLAN-remote-batch.md section 3; the remote stage's link itself: stage_link_test.cpp).
//
// SKELETON (担当A writes the rest, section 4.2 A-5): for now it checks what the skeleton fixed - the constants, the
// hello's size and layout (192 bytes, the token where every protocol has it), the message numbers - and, over
// loopback, that a protocol 7 peer is refused for its protocol with the reason, both ways.  To come:
//   1. a protocol 7 hello refused both ways, the reason naming both protocols (here now);
//   2. the hello's batch_slots and batch_flags arrive;
//   3. SlotRun round trips: S = 1..8, fp32 and bf16 rows, slots / tokens / positions / flags / tag arrive, the rows
//      come back (the handler changes them by a fixed rule);
//   4. three SlotRuns sent back to back (slot_run_send x 3): the replies come back in the order sent, with their tags;
//   5. SlotCommit carries keep[] whole; a handler's failure is the next SlotRun's error, also across a Reset;
//   6. SlotSave needs the session at L; after a SlotLoad the session is at L (a CkptSave at L is taken);
//   7. a SlotLoad of a checkpoint id the worker does not hold: `missing`;
//   8. malformed slot messages (S = 0, S past the window, bytes that do not add up, n past kStageSlotMax, an unknown
//      flag bit) are answered with an error and the link goes on;
//   9. a worker without slot handlers names --batch in its error;
//  10. stage_link_lost does not count the new errors as a lost link.
#include "strata/net/stage_link.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <thread>

using namespace strata::net;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
bool contains(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

constexpr int64_t kHbf = 4, kD = 4, kChunk = 8, kMaxT = 8;
StageHello hello_of() {
    StageHello h;
    h.n_embd = 4;
    h.hc = 1;
    h.n_layers = 4;
    h.n_expert = 2;
    h.layer_begin = 2;
    h.max_t = (int32_t) kMaxT;
    h.max_context = 64;
    h.chunk = kChunk;
    h.handoff_floats = kHbf;
    h.pack_hash = 0x5354524d;
    return h;
}
}  // namespace

int main() {
    std::printf("stage_link_batch_test (protocol %u)\n", kStageProtocol);

    // ---- what the skeleton fixed
    check(kStageProtocol == 8, "the protocol is 8");
    check(sizeof(StageHello) == 192, "the hello is 192 bytes, as in every protocol");
    check(offsetof(StageHello, batch_slots) == 120 && offsetof(StageHello, batch_flags) == 124 &&
              offsetof(StageHello, token) == 128,
          "batch_slots / batch_flags at 120 / 124, the token at 128");
    check((uint32_t) StageMsg::SlotRun == 16 && (uint32_t) StageMsg::SlotRunOk == 17 &&
              (uint32_t) StageMsg::SlotCommit == 18 && (uint32_t) StageMsg::SlotSave == 19 &&
              (uint32_t) StageMsg::SlotSaveOk == 20 && (uint32_t) StageMsg::SlotLoad == 21 &&
              (uint32_t) StageMsg::SlotLoadOk == 22,
          "the batch slot messages are 16..22");
    check(kStageSlotKeepAll == 1 && kStageBatchKeepAll == 1u && kStageBatchOverlap == 2u && kStageSlotMax == 64,
          "the flags and the slot bound");

    // ---- 1. a protocol 7 peer is refused for its protocol, with the reason, both ways
    float run_in[kMaxT * kHbf] = {}, run_out[kMaxT * kHbf] = {};
    float pf_in0[kChunk * kD] = {}, pf_in1[kChunk * kD] = {}, pf_out0[kChunk * kD] = {}, pf_out1[kChunk * kD] = {};
    StageBuffers buf;
    buf.run_in = run_in;
    buf.run_out = run_out;
    buf.run_floats = (size_t) (kMaxT * kHbf);
    buf.pf_in[0] = pf_in0;
    buf.pf_in[1] = pf_in1;
    buf.pf_out[0] = pf_out0;
    buf.pf_out[1] = pf_out1;
    buf.pf_floats = (size_t) (kChunk * kD);
    buf.handoff_floats = kHbf;
    buf.pf_row_floats = kD;
    StageHandlers h;
    std::atomic<int32_t> worker_protocol{(int32_t) kStageProtocol};
    h.hello = [&](const StageHello&, StageHello& mine, std::string&) {
        mine = hello_of();
        mine.protocol = (uint32_t) worker_protocol.load();
        return true;
    };
    h.run = [](int, const int32_t*, int64_t, std::string&) { return true; };
    h.commit = [](int, std::string&) { return true; };
    h.prefill = [](const int64_t*, int64_t, int64_t, int64_t, int64_t, const float*, float*, std::string&) { return true; };
    h.reset = [](const std::vector<int64_t>&, std::string&) { return true; };

    std::atomic<bool> stop{false};
    std::atomic<int> served{-1};
    StageStats st;
    std::thread t;
    std::string where;
    std::mt19937 rng(std::random_device{}());
    for (int attempt = 0; attempt < 8 && where.empty(); ++attempt) {
        const int port = 20000 + (int) (rng() % 40000);
        served = -1;
        t = std::thread([&, port] { served = serve_stage("127.0.0.1", port, "", h, buf, st, &stop); });
        const std::string at = "127.0.0.1:" + std::to_string(port);
        for (int i = 0; i < 100 && served.load() < 0; ++i) {
            StageHello bad = hello_of();
            bad.protocol = 4;
            StageHello peer;
            std::string e;
            StageClient c;
            if (c.connect(at, bad, peer, e) || contains(e, "the worker refused:")) { where = at; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (where.empty()) {
            stop = true;
            t.join();
            stop = false;
        }
    }
    check(!where.empty(), "a worker listens on loopback");
    if (!where.empty()) {
        StageClient c;
        StageHello peer;
        std::string err;
        StageHello seven = hello_of();
        seven.protocol = 7;
        check(!c.connect(where, seven, peer, err) && contains(err, "the worker refused:") &&
                  contains(err, "protocol 7/8") && contains(err, "same Strata build"),
              "a protocol 7 main process is refused by a protocol 8 worker, with the reason");
        c.close();
        worker_protocol = 7;   // (a worker that answers with a protocol 7 hello)
        err.clear();
        check(!c.connect(where, hello_of(), peer, err) && contains(err, "protocol 8/7") &&
                  !stage_link_lost(err),
              "a protocol 8 main process refuses a protocol 7 worker's hello, not as a lost link");
        c.close();
        worker_protocol = (int32_t) kStageProtocol;
        err.clear();
        StageHello mine = hello_of();
        mine.batch_slots = 4;
        mine.batch_flags = kStageBatchKeepAll;
        check(c.connect(where, mine, peer, err), "a protocol 8 main process is served");
        // TODO(remote-batch A): sections 2-10 above
        c.close();
        stop = true;
        t.join();
    }

    std::printf("%s (%d failed)\n", g_fail == 0 ? "all passed" : "FAILED", g_fail);
    return g_fail == 0 ? 0 : 1;
}
