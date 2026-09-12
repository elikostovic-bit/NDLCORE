// NDL v1.0 — Runtime library C ABI (libndl_rt)
// Frozen contract. Included by native codegen (declarations mirrored in .ll)
// and by the VM. Implementation: runtime/*.cpp
//
// Conventions:
//   NeuronType: 0 = Excitatory, 1 = Inhibitory
//   Device:     0 = CPU, 1 = GPU
//   DistKind (dense_connect): 0 = constant(p0), 1 = gaussian(mu=p0,sigma=p1),
//                             2 = uniform(a=p0,b=p1)
//   Errors: ndl_rt_panic prints "runtime error: ..." and exit(70).
//   print protocol: print_begin, values separated by single space, print_end
//   (newline + flush). f64 printed as %.6g.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NdlGroup NdlGroup;
typedef struct NdlTensor NdlTensor;
typedef void (*NdlSpikeHandler)(int64_t neuron, double t_ms, void* user);

// --- printing ---------------------------------------------------------------
void ndl_rt_print_begin(void);
void ndl_rt_print_str(const char* s);
void ndl_rt_print_i64(int64_t v);
void ndl_rt_print_f64(double v);
void ndl_rt_print_bool(int b);
void ndl_rt_print_end(void);

// --- groups -----------------------------------------------------------------
NdlGroup* ndl_rt_group_create(int64_t size, int32_t ntype, double tau,
                              double threshold, double rest, double reset,
                              const char* name);
int64_t ndl_rt_group_size(NdlGroup* g);
const char* ndl_rt_group_name(NdlGroup* g);

// --- connections ------------------------------------------------------------
void ndl_rt_dense_connect(NdlGroup* src, NdlGroup* dst, int32_t dist_kind,
                          double p0, double p1, int32_t plastic);
void ndl_rt_sparse_connect(NdlGroup* src, NdlGroup* dst, double density,
                           double weight, int32_t plastic);
void ndl_rt_one_to_one_connect(NdlGroup* src, NdlGroup* dst, double weight,
                               int32_t plastic);

// --- plasticity -------------------------------------------------------------
void ndl_rt_configure_stdp(NdlGroup* src, NdlGroup* dst, double lr_pot,
                           double lr_dep, double window_ms);

// --- spike handlers ---------------------------------------------------------
void ndl_rt_register_spike_handler(NdlGroup* g, NdlSpikeHandler fn, void* user);

// --- simulation -------------------------------------------------------------
void ndl_rt_schedule_inject(NdlGroup* g, int64_t lo, int64_t hi, double t_ms,
                            double current);
void ndl_rt_run(double duration_ms, double dt_ms, int32_t device);
// v2.1: the global simulation time base in ms (the same clock spike handlers
// receive as t_ms). Host-side temporal decoders (Ф9.4c) anchor their windows
// with it: pass it at window begin, subtract inside the handler.
double ndl_rt_sim_time(void);

// --- introspection ----------------------------------------------------------
double ndl_rt_get_weight(NdlGroup* src, NdlGroup* dst, int64_t i, int64_t j);

// --- v2.1: connectivity introspection (Ф9.4 signal/noise decomposition) ------
// For the SPARSE connection src->dst, collects for every dst neuron j in
// [0, rows_cap) the (i, w) of each synapse whose presynaptic index i is
// present in idx[0..n).   offs_out receives rows_cap+1 prefix offsets into
// cols_out/vals_out (rows beyond dst->size are empty). Synapses that are not
// wired simply contribute nothing. Dense/one2one/unknown -> INT64_MIN.
// Buffers too small: returns the NEGATED required total (-need; nothing was
// written except offs_out) — allocate and call again. No sparse src->dst
// connection: returns INT64_MIN. No state is mutated.
// The device state is synced down first, so this is safe under GPU ownership.
int64_t ndl_rt_row_hits(NdlGroup* src, NdlGroup* dst,
                        const int64_t* idx, int64_t n, int64_t rows_cap,
                        int64_t* offs_out, int64_t* cols_out,
                        double* vals_out, int64_t cap_out);

// --- IO ---------------------------------------------------------------------
void ndl_rt_save_checkpoint(const char* path);
void ndl_rt_load_checkpoint(const char* path);
void ndl_rt_export_raster(NdlGroup* g, const char* path);

// --- GPU --------------------------------------------------------------------
void ndl_rt_gpu_load_ptx(const char* ptx_source);

// --- math / rng / clocks ----------------------------------------------------
double ndl_rt_sin(double x);
double ndl_rt_cos(double x);
double ndl_rt_exp(double x);
double ndl_rt_log(double x);
double ndl_rt_sqrt(double x);
double ndl_rt_random_uniform(double a, double b);
double ndl_rt_random_gaussian(double mu, double sigma);
void ndl_rt_set_seed(uint64_t seed);
int64_t ndl_rt_now_ms(void);
int64_t ndl_rt_now_ns(void);

// --- strings ----------------------------------------------------------------
// Returns an interned concatenation of a and b (owned by the runtime).
const char* ndl_rt_str_concat(const char* a, const char* b);

// --- tensors (f64, 1..3 dims; experimental v1.0) -----------------------------
NdlTensor* ndl_rt_tensor_new(int64_t rank, const int64_t* dims, double fill);
double ndl_rt_tensor_get1(NdlTensor* t, int64_t i0);
double ndl_rt_tensor_get2(NdlTensor* t, int64_t i0, int64_t i1);
double ndl_rt_tensor_get3(NdlTensor* t, int64_t i0, int64_t i1, int64_t i2);
void ndl_rt_tensor_set1(NdlTensor* t, int64_t i0, double v);
void ndl_rt_tensor_set2(NdlTensor* t, int64_t i0, int64_t i1, double v);
void ndl_rt_tensor_set3(NdlTensor* t, int64_t i0, int64_t i1, int64_t i2, double v);
void ndl_rt_dump_tensor(const char* name, NdlTensor* t); // CSV to stdout

// --- v2.0: signals (neuromodulators, M(t) of 3-factor STDP) ------------------
// A signal is a named scalar modulator readable by the host at any time and
// sampled by the simulation on every tick. `set <Name> = expr;` in NDL maps to
// ndl_rt_signal_set; reading <Name> maps to ndl_rt_signal_get.
void ndl_rt_signal_create(const char* name, double initial);
void ndl_rt_signal_set(const char* name, double value);
double ndl_rt_signal_get(const char* name);

// Binds a signal as the M(t) modulator of the 3-factor STDP rule configured
// for src -> dst (pass NULL to unbind). Sema guarantees the pair has a
// configure_stdp and the signal exists.
void ndl_rt_stdp_set_modulator(NdlGroup* src, NdlGroup* dst, const char* name);

// --- v2.0: oscillators --------------------------------------------------------
// Injects I_osc = amplitude * sin(2*pi*freq*t + phase) into every neuron of
// `target` on every tick; t is the global simulation time (ms).
void ndl_rt_oscillator_add(const char* name, NdlGroup* target, double freq_hz,
                           double amplitude, double phase);

// --- v2.0: external streams (sensory input, e.g. VBM embeddings) -------------
// A stream is a named vector register of fixed `size`. A host thread pushes
// fresh vectors through a lock-free SPSC ring (ndl_rt_stream_push); the
// simulation drains the ring on every tick and Poisson-encodes the latest
// vector into the bound group: neuron i spikes with probability
// data[i] * max_freq * dt per tick, contributing a unit synaptic kick.
// encoding: 0 = Poisson (the only v2.0 encoder).
int64_t ndl_rt_stream_create(const char* name, int64_t size);
int64_t ndl_rt_stream_find(const char* name); // -1 when unknown
void ndl_rt_stream_push(int64_t stream, const double* data, int64_t n);
void ndl_rt_stream_set(int64_t stream, int64_t index, double value); // mailbox write
void ndl_rt_bind_input_stream(int64_t stream, NdlGroup* target, int32_t encoding,
                              double max_freq, double kick);

// --- v2.0: structural plasticity ----------------------------------------------
// set_plasticity toggles learning on a connection at runtime (sleep/wake);
// prune_weights zeroes every synapse with |w| < threshold (consolidation).
// normalize_incoming performs homeostatic synaptic scaling: for every
// postsynaptic neuron whose incoming |w| sum exceeds target_sum the row is
// scaled down to it (learned specificity survives, global drive is bounded).
// While the continuous loop is running all three are applied between ticks.
void ndl_rt_set_plasticity(NdlGroup* src, NdlGroup* dst, int32_t enabled);
void ndl_rt_prune_weights(NdlGroup* src, NdlGroup* dst, double threshold);
void ndl_rt_normalize_incoming(NdlGroup* src, NdlGroup* dst, double target_sum);

// --- v2.1: structural plasticity — grow-on-demand binding (Ф9.4b) -------------
// For the SPARSE connection src->dst, adds every (pre[i] -> post[j]) synapse
// of the cross product that is NOT wired yet, at `weight`. Rows stay sorted;
// duplicates (inside the batch and against existing wiring) are skipped, and
// out-of-range indices are ignored. Memory for the pair's binding becomes
// constant (winners × code rows) instead of eroding ∝1/E with the core size.
// While the continuous loop is running the request is queued and applied
// between ticks (returns -1); otherwise it applies immediately and returns
// the number of synapses added. Prune side: synapses whose weight decays
// below the threshold die structurally through prune_weights as before.
int64_t ndl_rt_grow_synapses(NdlGroup* src, NdlGroup* dst,
                             const int64_t* pre, int64_t n_pre,
                             const int64_t* post, int64_t n_post,
                             double weight);
// Cumulative count of synapses added by grow_synapses on this connection
// (runtime-only diagnostic; not part of the checkpoint). Missing connection
// → 0 (stats must not crash).
int64_t ndl_rt_grown_total(NdlGroup* src, NdlGroup* dst);

// --- v2.0: continuous mode (asynchronous event loop) --------------------------
// Starts a simulation thread ticking the network forever at dt_ms. Returns
// immediately. The loop applies oscillators, streams, modulated STDP and
// executes spike handlers between ticks; structural commands issued while it
// runs are deferred to tick boundaries. ndl_rt_stop_continuous stops the loop
// and joins the thread (called automatically at program end).
void ndl_rt_run_continuous(double dt_ms, int32_t device);
void ndl_rt_stop_continuous(void);
void ndl_rt_wait_continuous(void);  // blocks the caller until the loop stops
int32_t ndl_rt_is_continuous(void);

// --- v2.0: readout (RSOL) ------------------------------------------------------
NdlTensor* ndl_rt_get_membrane_potentials(NdlGroup* g); // f64 tensor [size]
NdlTensor* ndl_rt_predict_linear(NdlTensor* x, NdlTensor* w); // out[j]=sum_i w[j,i]*x[i]
double ndl_rt_vector_l2_norm(NdlTensor* t);

// --- Phase 2: GPU validation & benchmark (built-in simulation) ----------------
// :gputest backend — a deterministic synthetic mini-network (sparse E/I with
// dopamine-modulated STDP, oscillator, injections) run for 10k ticks on the
// CPU and then through the GPU backend (CUDA -> OpenCL); statistics compared.
// Returns 1 pass, 0 fail, -1 no GPU backend. Writes a human-readable report.
int ndl_rt_gputest(char* buf, int cap);
// :bench gpu backend — "neurons x ticks/s" table for the built-in simulation
// (sparse E/I networks of constant in-degree incl. streams and STDP), CPU vs
// GPU. max_neurons caps the largest row (pass 100000 for the full table).
void ndl_rt_bench_gpu(char* buf, int cap, int64_t max_neurons);

// Name of the GPU backend that last executed ticks ("cuda"/"opencl"), or
// nullptr when the CPU owns the simulation (for :stats / diagnostics).
const char* ndl_rt_gpu_active_backend(void);

// --- errors -----------------------------------------------------------------
#ifdef __cplusplus
[[noreturn]]
#endif
void ndl_rt_panic(const char* message); // exits with code 70

#ifdef __cplusplus
} // extern "C"
#endif
