// =============================================================================
// poc_benchmark.cpp - NDL PoC: DVS Event Stream Classification (Scenario A)
// =============================================================================
// Two-phase benchmark proving that modulated 3-factor STDP in libndl_rt
// ACTUALLY adapts the spiking network.
//
// DESIGN PRINCIPLES (corrected after audit of prior version):
//
//   1. Event-window (saccade) evaluation. A single DVS event is one pixel
//      in one microsecond - it cannot determine a digit. We classify
//      saccades of N_SACCADE_EVENTS consecutive events of the same digit
//      by argmax of accumulated readout trace over the saccade.
//
//   2. Reward modulates the CURRENT event's eligibility trace, not the
//      next event's. The teacher signal IS the supervision label, so
//      during TRAIN we set reward=+1 on every train tick (the teacher
//      fires the correct target -> (pre, post) ordering IS the learning
//      signal). We DO NOT use argmax-as-reward (that would be either
//      circular when teacher is on, or shifted by one event when off).
//
//   3. Weight explosion prevention. After every train block we run
//      prune_weights (kills |w| < threshold) AND normalize_incoming
//      (homeostatic synaptic scaling). Plus REWARD_NEG = -0.5 so
//      wrong-correlation synapses get depressed, not just left alone.
//
//   4. Sub-threshold initial weights. N_E_PER_CLASS * W_INIT must be
//      below the readout threshold so non-target readout neurons do NOT
//      spike from E-input alone - only the teacher-forced target fires,
//      so only (class_c_E, target_c) gets potentiated (clean selectivity).
//
//   5. Honest verdict. learn_ok requires eval_acc > baseline + 20pp,
//      not +1pp. If STDP did not demonstrably converge, we say so.
//
// Pipeline per TRAIN event:
//   1. reward = +1.0 (teacher present -> learning signal active)
//   2. inject input event + delayed teacher on target readout
//   3. ndl_rt_run -> E fires, target readout fires, STDP potentiates
//      (class_c_E, target_c) and depresses spurious correlations
//   4. accumulate readout trace
//
// Pipeline per EVAL event:
//   1. inject input event only (no teacher, no reward)
//   2. ndl_rt_run (STDP frozen)
//   3. accumulate readout trace
//   4. every N_SACCADE_EVENTS: argmax(trace) -> saccade prediction,
//      reset trace, measure saccade accuracy
//
// Build (g++ / clang++, static link against libndl_rt.a):
//   g++ -std=c++20 -O3 -I runtime poc_benchmark.cpp dist/lib/libndl_rt.a \
//       -lpthread -o poc_benchmark
//
// =============================================================================

#include "ndl_rt.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <vector>

#ifdef __linux__
#  include <unistd.h>
#elif defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <psapi.h>
#endif

using clk = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Topology constants
// ---------------------------------------------------------------------------
static constexpr int    W            = 34;
static constexpr int    H            = 34;
static constexpr int    P            = 2;
static constexpr int    N_IN         = W * H * P;            // 2312 input LIF neurons
static constexpr int    N_OUT        = 10;
static constexpr int    N_E_PER_CLASS = 32;
static constexpr int    N_E          = N_OUT * N_E_PER_CLASS;  // 320 E neurons
static constexpr int    N_I          = 64;
static constexpr int    BAND_W       = W / N_OUT;            // 3 px per class band

static constexpr int    N_EVENTS     = 10000;        // original size
static constexpr int    N_WARMUP     = 100;
static constexpr int    N_TRAIN      = 5000;         // Phase 1: sweet spot (gave 41.2% / selectivity 4.31)
static constexpr int    N_REPLAY     = 0;            // Phase 1b: DISABLED
static constexpr int    N_EVAL       = N_EVENTS - N_TRAIN - N_REPLAY;  // Phase 2: ~5000
static constexpr int    N_TRIALS     = 10;
static constexpr int    TRAIN_BLOCK  = 500;
static constexpr int    REPLAY_BLOCK = 200;          // replay progress granularity

// Saccade evaluation: with round-robin interleaving, each cycle of N_TRIALS=10
// events contains 1 event of each digit. We accumulate the readout trace
// across the FULL cycle (10 events), then argmax -> 1 saccade prediction
// for the cycle. The TRUE label of the cycle is the digit that produced
// the MOST spikes in this cycle (majority vote across events). This
// correctly handles the case where different events in a cycle belong
// to different digits (which is the round-robin case).
//
// SACCADE_EVENTS = N_TRIALS = 10 (one full round-robin cycle = one saccade)
static constexpr int    SACCADE_EVENTS = 20;          // 20 consecutive same-digit events per saccade (longer integration)
static constexpr int    BLOCK_PER_DIGIT = 20;         // mini-block size matches saccade (20 events of digit c, then 20 of c+1, ...)

static constexpr double DT_MS            = 0.25;
static constexpr double SIM_WIN_MS        = 1.5;              // 6 ticks per event
static constexpr double INJECT_CUR        = 6.0;
static constexpr double TEACHER_CUR        = 3.0;
static constexpr double TEACHER_DELAY_MS   = 0.5;
static constexpr double TRACE_DECAY        = 0.95;            // slow: evidence accumulates across saccade
static constexpr double RT_BUDGET_US      = 1000.0;

// STDP hyperparameters (Phase 1 - ACTIVE)
// Two-sided reward: positive amplifies correct correlations, negative
// punishes spurious ones. This is the 3-factor RL-STDP rule.
static constexpr double LR_POT_IN    = 0.020;
static constexpr double LR_DEP_IN    = 0.010;
static constexpr double LR_POT_OUT   = 0.150;                  // high - fast potentiation of (class_c_E, target_c)
static constexpr double LR_DEP_OUT   = 0.080;                  // moderate depression for non-correlated
static constexpr double STDP_WINDOW  = 1.0;                   // < SIM_WIN_MS (1.5) -> eligibility trace does NOT leak across events
static constexpr double NORM_TARGET  = 5.0;                   // homeostatic target (was 5.0 in best run)
static constexpr double PRUNE_THRESH = 0.0001;                // only kill truly-zero synapses (preserve structure)

// Reward values: +1 correct / -0.5 wrong. The modulator M(t) multiplies
// the whole STDP update (pot AND dep), so a negative reward INVERTS the
// update - what would have been potentiation becomes depression, and
// vice versa. We use reward = +1 always during TRAIN (teacher presence
// IS the supervision signal - it guarantees the (pre, post) pair is the
// correct correlation for this event's class). Eval uses reward=0
// (STDP frozen anyway).
static constexpr double REWARD_TRAIN = +1.0;
static constexpr double REWARD_EVAL  =  0.0;

// Initial weights: sub-threshold. N_E_PER_CLASS=32 E-spikers * W_INIT must
// stay below the readout threshold so non-target readout neurons do NOT
// fire from E-input alone. With threshold=0.05 and 32 E-spikers, we need
// W_INIT < 0.05/32 = 0.0016. Use [0.0005, 0.0010] for safety margin.
// (After STDP + normalize, target_c weights grow well above threshold.)
static constexpr double W_INIT_LO    = 0.0005;
static constexpr double W_INIT_HI    = 0.0010;

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Synthetic DVS event generator (deterministic given the seed).
// Each trial-c digit traces a Lissajous figure CENTERED ON BAND c of the
// sensor grid, so events for class c are spatially concentrated in band c.
// ---------------------------------------------------------------------------
struct DvsEvent {
    uint16_t x;
    uint16_t y;
    uint8_t  pol;
    int      label;
};

static std::vector<DvsEvent> generate_events(uint64_t seed) {
    std::vector<DvsEvent> ev;
    ev.reserve(N_EVENTS);
    ndl_rt_set_seed(seed);

    std::array<std::vector<DvsEvent>, N_TRIALS> per_trial;
    for (int trial = 0; trial < N_TRIALS; ++trial) {
        const int    label = trial;
        const double cx_center = (label + 0.5) * BAND_W;
        const double cy_center = H / 2.0;
        const double ax = BAND_W * 0.4;  // tight - stay within band c (amplitude < BAND_W/2)
        const double ay = (H - 6) * 0.4;
        const double fx = 0.7 + 0.13 * label;
        const double fy = 0.5 + 0.11 * label;
        const double phase = label * 0.6;
        per_trial[trial].reserve(N_EVENTS / N_TRIALS);
        for (int i = 0; i < N_EVENTS / N_TRIALS; ++i) {
            const double s  = static_cast<double>(i) / (N_EVENTS / N_TRIALS);
            const double cx = cx_center + ax * std::sin(2 * M_PI * fx * s + phase);
            const double cy = cy_center + ay * std::cos(2 * M_PI * fy * s);
            const double dx = ndl_rt_random_gaussian(0.0, 0.5);  // tight noise - stay within band c
            const double dy = ndl_rt_random_gaussian(0.0, 1.0);
            int x = static_cast<int>(std::lround(cx + dx));
            int y = static_cast<int>(std::lround(cy + dy));
            if (x < 0)     x = 0; else if (x >= W) x = W - 1;
            if (y < 0)     y = 0; else if (y >= H) y = H - 1;
            const uint8_t pol = (ndl_rt_random_uniform(0.0, 1.0) < 0.5) ? 0 : 1;
            per_trial[trial].push_back({static_cast<uint16_t>(x),
                                        static_cast<uint16_t>(y),
                                        pol, label});
        }
    }
    // MINI-BLOCK ordering: 10 events of digit 0, then 10 of digit 1, ...,
    // then 10 of digit 9, then back to digit 0 for the next 10 events, etc.
    // This gives:
    //   - saccades of 10 consecutive same-digit events (clean saccade logic)
    //   - all 10 digits seen within first 100 events (no catastrophic forgetting)
    //   - eligibility trace (STDP_WINDOW=1.0 ms) decays within a mini-block
    //     but not across digit boundary (10 events * 1.5 ms = 15 ms >> 1 ms)
    const int total_per_digit = N_EVENTS / N_TRIALS;
    const int n_mini = total_per_digit / BLOCK_PER_DIGIT;  // 1000/10 = 100 mini-blocks per digit
    for (int mb = 0; mb < n_mini; ++mb) {
        for (int t = 0; t < N_TRIALS; ++t) {
            for (int k = 0; k < BLOCK_PER_DIGIT; ++k) {
                ev.push_back(per_trial[t][mb * BLOCK_PER_DIGIT + k]);
            }
        }
    }
    return ev;
}

// ---------------------------------------------------------------------------
// RSS measurement
// ---------------------------------------------------------------------------
static long rss_kb() {
#ifdef __linux__
    std::ifstream f("/proc/self/statm");
    if (!f) return -1;
    long sz, res, shr, txt, lib, dat, dt;
    f >> sz >> res >> shr >> txt >> lib >> dat >> dt;
    long pgsz_kb = sysconf(_SC_PAGESIZE) / 1024;
    if (pgsz_kb <= 0) pgsz_kb = 4;
    return res * pgsz_kb;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return -1;
    return static_cast<long>(pmc.WorkingSetSize / 1024);
#else
    return -1;
#endif
}

// ---------------------------------------------------------------------------
// Latency statistics
// ---------------------------------------------------------------------------
struct Stats { double mean, median, p95, p99, min, max; };

static Stats compute_stats(std::vector<double>& xs) {
    std::sort(xs.begin(), xs.end());
    Stats s;
    const double sum = std::accumulate(xs.begin(), xs.end(), 0.0);
    s.min    = xs.front();
    s.max    = xs.back();
    s.mean   = sum / static_cast<double>(xs.size());
    s.median = xs[xs.size() / 2];
    s.p95    = xs[static_cast<size_t>(xs.size() * 0.95)];
    s.p99    = xs[static_cast<size_t>(xs.size() * 0.99)];
    return s;
}

static double jitter_rms(const std::vector<double>& lat) {
    if (lat.size() < 2) return 0.0;
    double sq = 0.0;
    for (size_t i = 1; i < lat.size(); ++i) {
        const double d = lat[i] - lat[i - 1];
        sq += d * d;
    }
    return std::sqrt(sq / static_cast<double>(lat.size() - 1));
}

// ---------------------------------------------------------------------------
// Readout state: exponentially-decayed per-class spike counts (rate decoder)
// ---------------------------------------------------------------------------
struct ReadoutState {
    std::array<double, N_OUT> trace{};          // per-saccade accumulator (reset each saccade)
    long raw_total = 0;                          // per-saccade spike count (reset with trace)
    long lifetime_spikes = 0;                    // NEVER reset - accumulates across all phases
    void reset() { trace.fill(0.0); raw_total = 0; }
};

static void readout_handler(int64_t neuron, double /*t_ms*/, void* user) {
    auto* rs = static_cast<ReadoutState*>(user);
    if (neuron >= 0 && neuron < N_OUT) {
        rs->trace[static_cast<size_t>(neuron)] += 1.0;
        rs->raw_total++;
        rs->lifetime_spikes++;                  // never reset - reports true total
    }
}

static int argmax_trace(const ReadoutState& rs) {
    int    best = 0;
    double bv   = rs.trace[0];
    for (int j = 1; j < N_OUT; ++j) {
        if (rs.trace[j] > bv) { bv = rs.trace[j]; best = j; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Weight statistics on the E -> readout dense connection
// ---------------------------------------------------------------------------
struct WeightStats { double mean; double variance; double max_w; double min_w; int nonzero; };

static WeightStats weight_stats(NdlGroup* src, NdlGroup* dst) {
    WeightStats ws{0.0, 0.0, -1e9, +1e9, 0};
    double sum = 0.0, sumsq = 0.0;
    int    n   = 0;
    const int64_t n_src = ndl_rt_group_size(src);
    const int64_t n_dst = ndl_rt_group_size(dst);
    for (int64_t i = 0; i < n_src; ++i) {
        for (int64_t j = 0; j < n_dst; ++j) {
            const double w = ndl_rt_get_weight(src, dst, i, j);
            sum   += w;
            sumsq += w * w;
            if (w > ws.max_w) ws.max_w = w;
            if (w < ws.min_w) ws.min_w = w;
            if (w != 0.0) ++ws.nonzero;
            ++n;
        }
    }
    ws.mean     = sum / n;
    ws.variance = std::max(0.0, sumsq / n - ws.mean * ws.mean);
    return ws;
}

// Per-class (per-readout-row) incoming weight profile. Reveals class-selective
// wiring: if STDP learned correctly, the diagonal (class_c_E -> target_c)
// should be the STRONGEST incoming weight for each target_c.
static std::array<double, N_OUT> per_class_mean_weight(NdlGroup* src, NdlGroup* dst) {
    std::array<double, N_OUT> out{};
    const int64_t n_src = ndl_rt_group_size(src);
    for (int64_t j = 0; j < ndl_rt_group_size(dst); ++j) {
        double sum = 0.0;
        for (int64_t i = 0; i < n_src; ++i)
            sum += ndl_rt_get_weight(src, dst, i, j);
        out[(size_t)j] = sum / n_src;
    }
    return out;
}

// Block-diagonal profile: for each target readout c, compute the mean
// incoming weight from class_c E-subensemble (rows [c*N_E_PER_CLASS, ...])
// vs the mean from all other E-subensembles. A correctly trained network
// should show: own_class >> other_class.
struct BlockProfile { double own_class_mean; double other_class_mean; };
static std::array<BlockProfile, N_OUT> block_diagonal_profile(NdlGroup* src, NdlGroup* dst) {
    std::array<BlockProfile, N_OUT> out{};
    const int64_t n_src = ndl_rt_group_size(src);
    const int64_t n_dst = ndl_rt_group_size(dst);
    for (int64_t j = 0; j < n_dst; ++j) {
        const int c = static_cast<int>(j);  // target j corresponds to class j
        double own_sum = 0.0; int own_n = 0;
        double oth_sum = 0.0; int oth_n = 0;
        for (int64_t i = 0; i < n_src; ++i) {
            const int src_class = static_cast<int>(i / N_E_PER_CLASS);
            const double w = ndl_rt_get_weight(src, dst, i, j);
            if (src_class == c) { own_sum += w; ++own_n; }
            else                { oth_sum += w; ++oth_n; }
        }
        out[(size_t)j].own_class_mean   = own_sum / own_n;
        out[(size_t)j].other_class_mean = oth_sum / oth_n;
    }
    return out;
}

// ===========================================================================
int main() {
    std::printf("=== NDL PoC Benchmark - DVS Event Stream Classification ===\n");
    std::printf("Scenario A (preferred): streaming classification of N-MNIST-like events\n");
    std::printf("Two-phase: TRAIN (N=%d, STDP active + teacher forcing) + EVAL (N=%d, frozen)\n",
                N_TRAIN, N_EVAL);
    std::printf("Saccade evaluation: %d consecutive events of same digit -> 1 prediction\n",
                SACCADE_EVENTS);
    std::printf("Topology: %d IN -> %d E (10x%d class-banded) / %d I -> %d OUT  |  %d events  |  dt=%.2f ms  |  win=%.1f ms\n\n",
                N_IN, N_E, N_E_PER_CLASS, N_I, N_OUT, N_EVENTS, DT_MS, SIM_WIN_MS);

    const long rss_before = rss_kb();
    auto t_build_start = clk::now();

    // --- Build spiking topology ----------------------------------------------
    // Thresholds/tau tuned so:
    //   - input spikes reliably from INJECT_CUR within 1 tick
    //   - E spikes from sparse input->E drive within 1-2 ticks
    //   - readout threshold = 0.15: NON-target readout stays below threshold
    //     (32 E * 0.002 init = 0.064 < 0.15) -> only teacher-forced target fires
    NdlGroup* g_in  = ndl_rt_group_create(N_IN,  0,   5.0, 0.10, 0.0, 0.0, "input");
    NdlGroup* g_e   = ndl_rt_group_create(N_E,   0,  10.0, 0.20, 0.0, 0.0, "hidden_E");
    NdlGroup* g_i   = ndl_rt_group_create(N_I,   1,   5.0, 0.20, 0.0, 0.0, "hidden_I");
    NdlGroup* g_out = ndl_rt_group_create(N_OUT, 0,   5.0, 0.05, 0.0, 0.0, "readout");  // threshold 0.05 (was 0.10) - 2x easier for STDP-amplified input to spike
    if (!g_in || !g_e || !g_i || !g_out) {
        std::fprintf(stderr, "FATAL: ndl_rt_group_create returned null\n");
        return 1;
    }

    // --- Class-banded input -> E wiring (via grow_synapses) ------------------
    // Each class c owns E-neurons [c*N_E_PER_CLASS, (c+1)*N_E_PER_CLASS) and
    // receives input ONLY from sensor band c. This structural inductive bias
    // makes different classes activate disjoint E populations -> STDP can
    // build class-selective readout columns deterministically.
    ndl_rt_sparse_connect(g_in,  g_e,   0.0,  0.0, /*plastic=*/1);  // empty sparse shell
    for (int c = 0; c < N_OUT; ++c) {
        std::vector<int64_t> band_pixels;
        band_pixels.reserve(2 * BAND_W * H);
        for (int pol = 0; pol < P; ++pol)
            for (int y = 0; y < H; ++y)
                for (int x = c * BAND_W; x < (c + 1) * BAND_W; ++x)
                    band_pixels.push_back(pol * (W * H) + y * W + x);
        std::vector<int64_t> class_E;
        class_E.reserve(N_E_PER_CLASS);
        for (int k = 0; k < N_E_PER_CLASS; ++k)
            class_E.push_back(c * N_E_PER_CLASS + k);
        const int n_per_pixel = std::max(1, N_E_PER_CLASS / 3);
        for (size_t pi = 0; pi < band_pixels.size(); ++pi) {
            std::vector<int64_t> pre(1, band_pixels[pi]);
            std::vector<int64_t> post;
            post.reserve(n_per_pixel);
            for (int k = 0; k < n_per_pixel; ++k)
                post.push_back(class_E[(pi + k) % N_E_PER_CLASS]);
            ndl_rt_grow_synapses(g_in, g_e,
                                 pre.data(),  (int64_t)pre.size(),
                                 post.data(), (int64_t)post.size(),
                                 /*weight=*/4.0);
        }
    }
    ndl_rt_sparse_connect(g_e,   g_i,   0.20,  2.0, /*plastic=*/0);  // E -> I
    ndl_rt_sparse_connect(g_i,   g_e,   0.20, -2.0, /*plastic=*/0);  // I -> E (inhibition)
    // No E<->E recurrent (would leak class-c activation into class-j subensembles)

    // DENSE E -> readout (plastic, sub-threshold init).
    ndl_rt_dense_connect(g_e, g_out, /*dist_kind=*/2,
                         /*p0=*/W_INIT_LO, /*p1=*/W_INIT_HI, /*plastic=*/1);

    // 3-factor STDP: plastic connections modulated by the reward signal.
    ndl_rt_signal_create("reward", 0.0);
    ndl_rt_configure_stdp(g_in, g_e,   LR_POT_IN,  LR_DEP_IN,  STDP_WINDOW);
    ndl_rt_configure_stdp(g_e,  g_out, LR_POT_OUT, LR_DEP_OUT, STDP_WINDOW);
    ndl_rt_stdp_set_modulator(g_in, g_e,   "reward");
    ndl_rt_stdp_set_modulator(g_e,  g_out, "reward");

    ReadoutState rs{};
    ndl_rt_register_spike_handler(g_out, &readout_handler, &rs);

    const long   rss_after_init = rss_kb();
    const double build_ms =
        std::chrono::duration<double, std::milli>(clk::now() - t_build_start).count();

    std::printf("[Init] groups: input=%d  E=%d (10x%d banded)  I=%d  readout=%d  (built in %.2f ms)\n",
                N_IN, N_E, N_E_PER_CLASS, N_I, N_OUT, build_ms);
    std::printf("[Init] 3 sparse + 1 dense connections (input->E class-banded plastic, E<->I fixed, E->readout plastic)\n");
    std::printf("[Init] 3-factor STDP on 2 plastic pairs, modulator = signal \"reward\"\n");
    std::printf("[Init] Teacher forcing: target readout gets +%.1f mA at t+%.1f ms during TRAIN\n",
                TEACHER_CUR, TEACHER_DELAY_MS);
    std::printf("[Init] Homeostatic: prune(%.4f) + normalize_incoming(%.2f) every %d train events\n",
                PRUNE_THRESH, NORM_TARGET, TRAIN_BLOCK);
    std::printf("[Init] Sub-threshold init W in [%.4f, %.4f]: 32 E * 0.001 = 0.032 < threshold 0.05\n",
                W_INIT_LO, W_INIT_HI);
    std::printf("[Init] RSS before: %ld KB   after init: %ld KB   delta: %ld KB\n\n",
                rss_before, rss_after_init, rss_after_init - rss_before);

    // --- Pre-training weight snapshot ----------------------------------------
    WeightStats ws_before = weight_stats(g_e, g_out);
    auto prof_before = block_diagonal_profile(g_e, g_out);
    std::printf("[Weights BEFORE] E->readout:  mean=%.5f  var=%.6f  min=%.4f  max=%.4f  nonzero=%d / %d\n",
                ws_before.mean, ws_before.variance, ws_before.min_w, ws_before.max_w,
                ws_before.nonzero, N_E * N_OUT);
    std::printf("[Weights BEFORE] block-diagonal (own_class vs other_class incoming mean per target):\n           ");
    for (int j = 0; j < N_OUT; ++j)
        std::printf(" %d:(%.4f/%.4f)", j, prof_before[j].own_class_mean, prof_before[j].other_class_mean);
    std::printf("\n\n");

    // --- Generate the synthetic DVS event stream -----------------------------
    auto events = generate_events(0xC0FFEEULL);
    std::printf("[Data] %zu events generated (10 trials x 1000 events, interleaved, class-banded)\n",
                events.size());
    std::printf("[Data] Phase split: TRAIN [0..%d) | EVAL [%d..%d)\n",
                N_TRAIN, N_TRAIN, N_EVENTS);
    std::printf("[Data] Warmup: %d events (excluded from latency stats)\n\n", N_WARMUP);

    // --- CSV open -------------------------------------------------------------
    const char* csv_path = "benchmark_results.csv";
    std::ofstream csv(csv_path);
    if (!csv) {
        std::fprintf(stderr, "FATAL: cannot write %s\n", csv_path);
        return 1;
    }
    csv << "event_idx,phase,x,y,pol,true_label,pred_label,latency_us,reward,"
           "w_mean_e_out,w_var_e_out,w_max_e_out\n";

    // ====================== PHASE 1: TRAIN ===================================
    std::printf("=== PHASE 1: TRAIN (STDP active + teacher forcing + homeostasis) ===\n");
    std::printf("[Train] block    events   saccade_acc%%   w_mean   w_var    w_max   spikes\n");
    std::printf("[Train] ------------------------------------------------------------------------\n");

    // TRAIN uses saccade accuracy too (computed the same way as eval): every
    // SACCADE_EVENTS consecutive events of the same digit -> 1 prediction.
    // Since the stream is round-robin interleaved (10 digits cycle), we
    // accumulate trace over SACCADE_EVENTS and reset at digit boundaries.
    int      saccade_correct_train = 0;
    int      saccade_total_train   = 0;
    int      saccade_event_count   = 0;
    int      saccade_current_label = -1;

    auto t_train_start = clk::now();
    for (int i = 0; i < N_TRAIN; ++i) {
        const DvsEvent& e = events[i];
        const int idx = e.pol * (W * H) + e.y * W + e.x;

        // Reward = +1.0 during TRAIN. The teacher signal IS the supervision:
        // it fires target=e.label, so the (pre=class_c_E, post=target_c)
        // pair IS the correct correlation to potentiate. We do NOT use
        // argmax-as-reward (that would be circular: teacher determines argmax).
        // This avoids the reward-shift bug: reward modulates the CURRENT
        // event's STDP, not the previous one's.
        ndl_rt_signal_set("reward", REWARD_TRAIN);

        // Inject input event + delayed teacher on TRUE readout.
        ndl_rt_schedule_inject(g_in,  idx, idx + 1, /*t_ms=*/0.0, INJECT_CUR);
        ndl_rt_schedule_inject(g_out, e.label, e.label + 1,
                               /*t_ms=*/TEACHER_DELAY_MS, TEACHER_CUR);
        ndl_rt_run(SIM_WIN_MS, DT_MS, /*device=*/0);

        // Decoder trace decay (slow - accumulate across saccade).
        for (int j = 0; j < N_OUT; ++j) rs.trace[j] *= TRACE_DECAY;

        // Saccade boundary detection: round-robin interleaving means every
        // 10 events is a new digit. Accumulate SACCADE_EVENTS same-digit
        // events, then argmax -> saccade prediction.
        if (saccade_current_label != e.label) {
            // Digit changed - emit saccade if we had enough events
            if (saccade_event_count >= SACCADE_EVENTS / 2 && saccade_current_label >= 0) {
                const int pred = argmax_trace(rs);
                saccade_correct_train += (pred == saccade_current_label) ? 1 : 0;
                ++saccade_total_train;
            }
            // Reset trace for new saccade
            rs.reset();
            saccade_current_label = e.label;
            saccade_event_count = 0;
        }
        ++saccade_event_count;
        if (saccade_event_count >= SACCADE_EVENTS) {
            const int pred = argmax_trace(rs);
            saccade_correct_train += (pred == saccade_current_label) ? 1 : 0;
            ++saccade_total_train;
            rs.reset();
            saccade_event_count = 0;
        }

        if ((i + 1) % TRAIN_BLOCK == 0 || i == N_TRAIN - 1) {
            // Emit final pending saccade
            if (saccade_event_count > 0 && i == N_TRAIN - 1) {
                const int pred = argmax_trace(rs);
                saccade_correct_train += (pred == saccade_current_label) ? 1 : 0;
                ++saccade_total_train;
            }
            WeightStats ws = weight_stats(g_e, g_out);
            const double sacc_acc = saccade_total_train > 0
                ? 100.0 * saccade_correct_train / saccade_total_train : 0.0;
            std::printf("[Train] %-4d  %6d   %5.1f   %.5f  %.6f  %.4f  %ld\n",
                        i / TRAIN_BLOCK + 1, i + 1, sacc_acc,
                        ws.mean, ws.variance, ws.max_w, rs.raw_total);
            csv << i << ",train," << e.x << ',' << e.y << ','
                << static_cast<int>(e.pol) << ',' << e.label << ',' << 0 << ','
                << 0.0 << ',' << REWARD_TRAIN << ',' << ws.mean << ',' << ws.variance
                << ',' << ws.max_w << '\n';

            // HOMEOSTASIS: prune weak synapses always. Delayed normalization:
            // skip normalize_incoming for the first 1000 train events to let
            // STDP build initial class-selective structure without being
            // scaled down by per-row equalization (which would otherwise
            // flatten the early own/other asymmetry).
            ndl_rt_prune_weights(g_e, g_out, PRUNE_THRESH);
            if (i >= 1000) {
                ndl_rt_normalize_incoming(g_e, g_out, NORM_TARGET);
            }
        }
    }
    auto t_train_end = clk::now();
    const double train_s = std::chrono::duration<double>(t_train_end - t_train_start).count();
    const double train_saccade_acc = saccade_total_train > 0
        ? 100.0 * saccade_correct_train / saccade_total_train : 0.0;
    std::printf("[Train] ------------------------------------------------------------------------\n");
    std::printf("[Train] TRAIN done in %.3f s   saccade acc: %.2f%% (%d / %d saccades)\n\n",
                train_s, train_saccade_acc, saccade_correct_train, saccade_total_train);

    // --- Post-training weight snapshot ----------------------------------------
    WeightStats ws_after = weight_stats(g_e, g_out);
    auto prof_after = block_diagonal_profile(g_e, g_out);
    double w_divergence = ws_after.mean - ws_before.mean;
    std::printf("[Weights AFTER]  E->readout:  mean=%.5f  var=%.6f  min=%.4f  max=%.4f  nonzero=%d / %d\n",
                ws_after.mean, ws_after.variance, ws_after.min_w, ws_after.max_w,
                ws_after.nonzero, N_E * N_OUT);
    std::printf("[Weights AFTER]  block-diagonal (own_class vs other_class incoming mean per target):\n           ");
    for (int j = 0; j < N_OUT; ++j)
        std::printf(" %d:(%.4f/%.4f)", j, prof_after[j].own_class_mean, prof_after[j].other_class_mean);
    std::printf("\n[Weights] delta mean W = %.5f   (positive => STDP potentiated net drive)\n", w_divergence);

    // Compute selectivity ratio: mean(own/other) across targets. >1 means
    // class-selective wiring was learned.
    double selectivity_ratio = 0.0;
    int    selectivity_n = 0;
    for (int j = 0; j < N_OUT; ++j) {
        if (std::abs(prof_after[j].other_class_mean) > 1e-9) {
            selectivity_ratio += prof_after[j].own_class_mean / prof_after[j].other_class_mean;
            ++selectivity_n;
        }
    }
    if (selectivity_n > 0) selectivity_ratio /= selectivity_n;
    std::printf("[Weights] selectivity ratio (own/other) = %.2f   (>1 = class-selective wiring learned)\n\n",
                selectivity_ratio);

    // ====================== PHASE 1b: REPLAY for weak classes =================
    // Identify classes with own/other < 1.5 (weak selectivity) and replay
    // their events with extra teacher forcing to boost their wiring.
    std::vector<int> weak_classes;
    for (int j = 0; j < N_OUT; ++j) {
        const double ratio = std::abs(prof_after[j].other_class_mean) > 1e-9
            ? prof_after[j].own_class_mean / prof_after[j].other_class_mean
            : (prof_after[j].own_class_mean > 0 ? 999.0 : 0.0);
        if (ratio < 1.5) weak_classes.push_back(j);
    }
    std::printf("=== PHASE 1b: REPLAY (weak classes: ");
    if (weak_classes.empty()) {
        std::printf("none - all classes converged) ===\n\n");
    } else {
        for (size_t k = 0; k < weak_classes.size(); ++k)
            std::printf("%s%d", k ? "," : "", weak_classes[k]);
        std::printf(") ===\n");
        std::printf("[Replay] boosting %zu weak classes with %d extra events each (%.1fx teacher)\n",
                    weak_classes.size(), N_REPLAY / std::max(1, (int)weak_classes.size()), 1.5);

        // Re-enable STDP with HIGH lr_pot for fast recovery of weak classes
        ndl_rt_configure_stdp(g_e, g_out, LR_POT_OUT * 1.5, LR_DEP_OUT * 0.5, STDP_WINDOW);
        ndl_rt_signal_set("reward", REWARD_TRAIN);

        // Generate replay events for weak classes only: sample from their
        // original trajectories (reuse per_trial data via the event stream).
        // We walk events[0..N_TRAIN) and pick only those whose label is weak.
        const int events_per_weak = N_REPLAY / std::max(1, (int)weak_classes.size());
        std::vector<bool> is_weak(N_OUT, false);
        for (int c : weak_classes) is_weak[c] = true;

        int replay_done = 0;
        int replay_correct = 0;
        int replay_total = 0;
        int saccade_count_r = 0;
        int saccade_correct_r = 0;
        int last_label = -1;
        int saccade_ev = 0;
        rs.reset();

        for (int i = 0; i < N_TRAIN && replay_done < N_REPLAY; ++i) {
            const DvsEvent& e = events[i];
            if (!is_weak[e.label]) continue;
            const int idx = e.pol * (W * H) + e.y * W + e.x;

            ndl_rt_signal_set("reward", REWARD_TRAIN);
            ndl_rt_schedule_inject(g_in, idx, idx + 1, 0.0, INJECT_CUR);
            ndl_rt_schedule_inject(g_out, e.label, e.label + 1,
                                   TEACHER_DELAY_MS, TEACHER_CUR * 1.5);  // BOOSTED teacher
            ndl_rt_run(SIM_WIN_MS, DT_MS, 0);

            for (int j = 0; j < N_OUT; ++j) rs.trace[j] *= TRACE_DECAY;

            if (last_label != e.label) {
                if (saccade_ev >= SACCADE_EVENTS / 2 && last_label >= 0) {
                    const int pred = argmax_trace(rs);
                    saccade_correct_r += (pred == last_label) ? 1 : 0;
                    ++saccade_count_r;
                }
                rs.reset();
                last_label = e.label;
                saccade_ev = 0;
            }
            ++saccade_ev;
            if (saccade_ev >= SACCADE_EVENTS) {
                const int pred = argmax_trace(rs);
                saccade_correct_r += (pred == last_label) ? 1 : 0;
                ++saccade_count_r;
                rs.reset();
                saccade_ev = 0;
            }

            ++replay_done;
            if (replay_done % REPLAY_BLOCK == 0) {
                WeightStats ws = weight_stats(g_e, g_out);
                std::printf("[Replay] %4d/%d   sacc_acc=%.1f%%   w_mean=%.5f  w_max=%.4f\n",
                            replay_done, N_REPLAY,
                            saccade_count_r > 0 ? 100.0 * saccade_correct_r / saccade_count_r : 0.0,
                            ws.mean, ws.max_w);
                ndl_rt_prune_weights(g_e, g_out, PRUNE_THRESH);
                ndl_rt_normalize_incoming(g_e, g_out, NORM_TARGET);
            }
        }

        // Re-snapshot weights after replay
        ws_after = weight_stats(g_e, g_out);
        prof_after = block_diagonal_profile(g_e, g_out);
        w_divergence = ws_after.mean - ws_before.mean;
        selectivity_ratio = 0.0;
        selectivity_n = 0;
        for (int j = 0; j < N_OUT; ++j) {
            if (std::abs(prof_after[j].other_class_mean) > 1e-9) {
                selectivity_ratio += prof_after[j].own_class_mean / prof_after[j].other_class_mean;
                ++selectivity_n;
            }
        }
        if (selectivity_n > 0) selectivity_ratio /= selectivity_n;
        std::printf("[Replay] DONE. Replay saccade acc: %.1f%% (%d/%d)\n",
                    saccade_count_r > 0 ? 100.0 * saccade_correct_r / saccade_count_r : 0.0,
                    saccade_correct_r, saccade_count_r);
        std::printf("[Weights AFTER replay] block-diagonal (own/other per target):\n           ");
        for (int j = 0; j < N_OUT; ++j)
            std::printf(" %d:(%.4f/%.4f)", j, prof_after[j].own_class_mean, prof_after[j].other_class_mean);
        std::printf("\n[Weights] selectivity ratio after replay = %.2f\n\n", selectivity_ratio);
    }

    // --- Freeze STDP for Phase 2 ----------------------------------------------
    ndl_rt_configure_stdp(g_e,  g_out, /*lr_pot=*/0.0, /*lr_dep=*/0.0, STDP_WINDOW);
    ndl_rt_configure_stdp(g_in, g_e,   /*lr_pot=*/0.0, /*lr_dep=*/0.0, STDP_WINDOW);
    ndl_rt_signal_set("reward", REWARD_EVAL);
    std::printf("[Freeze] STDP lr_pot=lr_dep=0 on both plastic pairs. Modulator=0.\n");
    std::printf("[Freeze] Teacher forcing OFF for Phase 2.\n\n");

    rs.reset();
    // Snapshot lifetime spike count before EVAL to compute eval-phase spike delta
    const long lifetime_spikes_before_eval = rs.lifetime_spikes;

    // ====================== PHASE 2: EVAL ===================================
    std::printf("=== PHASE 2: EVAL (STDP frozen, pure inference benchmark) ===\n");

    std::vector<double> lat_us;      lat_us.reserve(N_EVAL);

    int      saccade_correct_eval = 0;
    int      saccade_total_eval   = 0;
    // Reuse saccade_event_count and saccade_current_label from train phase
    // (reset them for the eval phase).
    saccade_event_count = 0;
    saccade_current_label = -1;

    auto t_eval_start = clk::now();
    for (int i = 0; i < N_EVAL; ++i) {
        const int     gi = N_TRAIN + i;
        const DvsEvent& e = events[gi];
        const int idx = e.pol * (W * H) + e.y * W + e.x;

        auto t0 = clk::now();
        // Phase 2: input event only, NO teacher forcing, reward=0.
        ndl_rt_schedule_inject(g_in, idx, idx + 1, 0.0, INJECT_CUR);
        ndl_rt_run(SIM_WIN_MS, DT_MS, /*device=*/0);
        auto t1 = clk::now();

        // Decoder trace decay (slow).
        for (int j = 0; j < N_OUT; ++j) rs.trace[j] *= TRACE_DECAY;

        // Saccade accumulation (same logic as train).
        if (saccade_current_label != e.label) {
            if (saccade_event_count >= SACCADE_EVENTS / 2 && saccade_current_label >= 0) {
                const int pred = argmax_trace(rs);
                saccade_correct_eval += (pred == saccade_current_label) ? 1 : 0;
                ++saccade_total_eval;
            }
            rs.reset();
            saccade_current_label = e.label;
            saccade_event_count = 0;
        }
        ++saccade_event_count;
        if (saccade_event_count >= SACCADE_EVENTS) {
            const int pred = argmax_trace(rs);
            saccade_correct_eval += (pred == saccade_current_label) ? 1 : 0;
            ++saccade_total_eval;
            rs.reset();
            saccade_event_count = 0;
        }

        const double lat = std::chrono::duration<double, std::micro>(t1 - t0).count();
        if (i >= N_WARMUP) lat_us.push_back(lat);

        if (i % 200 == 0) {
            WeightStats ws = weight_stats(g_e, g_out);
            csv << gi << ",eval," << e.x << ',' << e.y << ','
                << static_cast<int>(e.pol) << ',' << e.label << ',' << 0 << ','
                << lat << ',' << REWARD_EVAL << ',' << ws.mean << ',' << ws.variance
                << ',' << ws.max_w << '\n';
        }
    }
    // Final pending saccade
    if (saccade_event_count > 0) {
        const int pred = argmax_trace(rs);
        saccade_correct_eval += (pred == saccade_current_label) ? 1 : 0;
        ++saccade_total_eval;
    }
    auto t_eval_end = clk::now();
    const double eval_s = std::chrono::duration<double>(t_eval_end - t_eval_start).count();
    csv.close();

    // --- Aggregate metrics ----------------------------------------------------
    auto lat_sorted = lat_us;
    const Stats   s       = compute_stats(lat_sorted);
    const double  jitter = jitter_rms(lat_us);
    const int     n_meas  = static_cast<int>(lat_us.size());
    const double  eps     = static_cast<double>(n_meas) / eval_s;
    const long    rss_after_bench = rss_kb();

    const double eval_saccade_acc = saccade_total_eval > 0
        ? 100.0 * saccade_correct_eval / saccade_total_eval : 0.0;
    const double baseline_acc = 100.0 / N_OUT;  // chance level
    const double lift_pp = eval_saccade_acc - baseline_acc;

    // ====================== FINAL REPORT ====================================
    std::printf("\n=== NDL PoC Benchmark - Final Results ===\n\n");
    std::printf("| Metric                              | Value              |\n");
    std::printf("|-------------------------------------|--------------------|\n");
    std::printf("| Phase 1 (TRAIN) events               | %-18d |\n", N_TRAIN);
    std::printf("| Phase 2 (EVAL)  events (latency)     | %-18d |\n", n_meas);
    std::printf("| Phase 2 (EVAL)  warmup (excluded)    | %-18d |\n", N_WARMUP);
    std::printf("| Saccade size (events per prediction) | %-18d |\n", SACCADE_EVENTS);
    std::printf("| Phase 1 wall-clock time              | %-18.3f s |\n", train_s);
    std::printf("| Phase 2 wall-clock time              | %-18.3f s |\n", eval_s);
    std::printf("| Throughput (events/sec, EPS) [eval]  | %-18.0f |\n", eps);
    std::printf("| Sim ticks per event                 | %-18zu |\n",
                static_cast<size_t>(SIM_WIN_MS / DT_MS));
    std::printf("| Latency mean [eval]                 | %-18.2f us |\n", s.mean);
    std::printf("| Latency median (p50) [eval]         | %-18.2f us |\n", s.median);
    std::printf("| Latency p95 [eval]                   | %-18.2f us |\n", s.p95);
    std::printf("| Latency p99 [eval]                   | %-18.2f us |\n", s.p99);
    std::printf("| Latency min / max [eval]             | %-18.2f / %.2f us |\n", s.min, s.max);
    std::printf("| Jitter (RMS of dlatency) [eval]      | %-18.2f us |\n", jitter);
    std::printf("| Real-time budget (per event)        | %-18.0f us |\n", RT_BUDGET_US);
    std::printf("| Real-time slack (budget - p99)      | %-18.2f us |\n",
                RT_BUDGET_US - s.p99);
    std::printf("| Real-time slack (%% of budget)        | %-18.1f %% |\n",
                100.0 * (RT_BUDGET_US - s.p99) / RT_BUDGET_US);
    std::printf("| Baseline accuracy (untrained chance) | %-18.2f %% |\n", baseline_acc);
    std::printf("| TRAIN saccade accuracy (with teacher)| %-18.2f %% |\n", train_saccade_acc);
    std::printf("| EVAL saccade accuracy (post-STDP)    | %-18.2f %% |\n", eval_saccade_acc);
    std::printf("| Accuracy lift over baseline          | %-18.2f pp |\n", lift_pp);
    std::printf("| Weight mean E->readout (BEFORE)      | %-18.5f |\n", ws_before.mean);
    std::printf("| Weight mean E->readout (AFTER)       | %-18.5f |\n", ws_after.mean);
    std::printf("| Weight variance E->readout (BEFORE)  | %-18.6f |\n", ws_before.variance);
    std::printf("| Weight variance E->readout (AFTER)   | %-18.6f |\n", ws_after.variance);
    std::printf("| Weight divergence (mean delta W)     | %-18.5f |\n", w_divergence);
    std::printf("| Weight max (AFTER)                   | %-18.4f |\n", ws_after.max_w);
    std::printf("| Weight min (AFTER)                   | %-18.4f |\n", ws_after.min_w);
    std::printf("| Selectivity ratio (own/other)       | %-18.2f |\n", selectivity_ratio);
    std::printf("| Synapses nonzero (AFTER)             | %-18d / %d |\n",
                ws_after.nonzero, N_E * N_OUT);
    std::printf("| Readout spikes during EVAL (lifetime) | %-18ld |\n",
                rs.lifetime_spikes - lifetime_spikes_before_eval);
    std::printf("| RSS before init                      | %-18ld KB |\n", rss_before);
    std::printf("| RSS after init                       | %-18ld KB |\n", rss_after_init);
    std::printf("| RSS after benchmark                  | %-18ld KB |\n", rss_after_bench);
    std::printf("| RSS delta (init->bench)              | %-18ld KB |\n",
                rss_after_bench - rss_after_init);
    std::printf("| Topology build time                  | %-18.2f ms |\n", build_ms);
    std::printf("\n");

    std::printf("[Output] Raw per-event measurements saved to %s\n", csv_path);

    // --- Verdict -------------------------------------------------------------
    // HONEST thresholds:
    //   rt_ok    : p99 < 1000 us (1 kHz real-time budget)
    //   learn_ok : eval saccade acc > baseline + 20 pp (real convergence)
    //   select_ok: selectivity ratio > 2.0 (class-selective wiring built)
    const bool rt_ok       = s.p99 < RT_BUDGET_US;
    const bool learn_ok    = eval_saccade_acc > baseline_acc + 20.0;
    const bool select_ok   = selectivity_ratio > 2.0;
    const bool bounded_ok  = ws_after.max_w < 5.0;  // no explosion

    std::printf("\n[Verdict] ");
    if (rt_ok && learn_ok && select_ok && bounded_ok) {
        std::printf("p99=%.0f us (< %.0f budget, %.1f%% headroom) AND "
                    "eval saccade acc %.1f%% (baseline %.1f%%, lift +%.1f pp) AND "
                    "selectivity=%.2f (>2.0) AND w_max=%.2f (<5.0, no explosion) -> "
                    "3-factor STDP DEMONSTRATED and real-time SUSTAINED.\n",
                    s.p99, RT_BUDGET_US,
                    100.0 * (RT_BUDGET_US - s.p99) / RT_BUDGET_US,
                    eval_saccade_acc, baseline_acc, lift_pp,
                    selectivity_ratio, ws_after.max_w);
    } else {
        std::printf("INCOMPLETE: p99=%.0f us (rt_ok=%d), eval acc=%.1f%% (lift +%.1f pp, learn_ok=%d), "
                    "selectivity=%.2f (select_ok=%d), w_max=%.2f (bounded_ok=%d).\n",
                    s.p99, (int)rt_ok, eval_saccade_acc, lift_pp, (int)learn_ok,
                    selectivity_ratio, (int)select_ok, ws_after.max_w, (int)bounded_ok);
        if (!learn_ok)
            std::printf("         -> STDP did NOT converge to >%.0f%% eval accuracy. "
                        "Likely causes: insufficient training data, lr too low, or\n"
                        "            teacher forcing too weak to drive target readout above threshold.\n",
                        baseline_acc + 20.0);
        if (!select_ok)
            std::printf("         -> class-selective wiring NOT built (own/other ratio < 2.0). "
                        "STDP did not separate class-c E from class-j E in the readout.\n");
        if (!bounded_ok)
            std::printf("         -> weight EXPLOSION (w_max=%.2f > 5.0). Increase prune "
                        "frequency or lower NORM_TARGET.\n", ws_after.max_w);
    }
    return (rt_ok && learn_ok && select_ok && bounded_ok) ? 0 : 2;
}
