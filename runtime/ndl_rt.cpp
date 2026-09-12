// NDL v1.0 — runtime core: panic, print protocol, clocks, RNG (xoshiro256**),
// math wrappers, string interning, tensors. Network/IO/GPU layers live in the
// sibling translation units (see ndl_rt_internal.h).
#include "ndl_rt_internal.h"

#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <unordered_set>
#include <vector>

// --- errors -----------------------------------------------------------------

void ndl_rt_panic(const char* message) {
  std::fprintf(stderr, "runtime error: %s\n", message ? message : "unknown error");
  std::fflush(stderr);
  std::exit(70);
}

std::string ndl_rt_sfmt(const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (n < 0) return std::string();
  if ((size_t)n >= sizeof(buf)) {
    std::string big((size_t)n + 1, '\0');
    va_start(ap, fmt);
    std::vsnprintf(&big[0], big.size(), fmt, ap);
    va_end(ap);
    big.resize((size_t)n);
    return big;
  }
  return std::string(buf, (size_t)n);
}

// --- print protocol ---------------------------------------------------------
// print_begin resets the separator state; every print_* emits its value, the
// first one without and the rest with a single leading space; print_end emits
// '\n' and flushes.

namespace {
bool g_print_first = true;

void print_sep() {
  if (!g_print_first) std::fputc(' ', stdout);
  g_print_first = false;
}
}  // namespace

void ndl_rt_print_begin(void) { g_print_first = true; }

void ndl_rt_print_str(const char* s) {
  print_sep();
  std::fputs(s ? s : "(null)", stdout);
}

void ndl_rt_print_i64(int64_t v) {
  print_sep();
  std::fprintf(stdout, "%" PRId64, v);
}

void ndl_rt_print_f64(double v) {
  print_sep();
  std::fprintf(stdout, "%.6g", v);
}

void ndl_rt_print_bool(int b) {
  print_sep();
  std::fputs(b ? "true" : "false", stdout);
}

void ndl_rt_print_end(void) {
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

// --- clocks -----------------------------------------------------------------

int64_t ndl_rt_now_ms(void) {
  return (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

int64_t ndl_rt_now_ns(void) {
  return (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// --- RNG: xoshiro256** with splitmix64 seeding and Box-Muller gaussian ------

namespace {
inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

// Non-zero default state (all-zero is forbidden for xoshiro256**).
uint64_t g_rng_state[4] = {0x9E3779B97F4A7C15ull, 0xBB67AE8584CAA73Bull,
                           0x3C6EF372FE94F82Bull, 0xA54FF53A5F1D36F1ull};

uint64_t splitmix64(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Box-Muller spare half of the gaussian pair.
bool g_have_spare = false;
double g_spare = 0.0;

constexpr double kTwoPi = 6.283185307179586476925286766559;
}  // namespace

uint64_t ndl_rt_rng_next_u64(void) {
  const uint64_t result = rotl64(g_rng_state[1] * 5, 7) * 9;
  const uint64_t t = g_rng_state[1] << 17;
  g_rng_state[2] ^= g_rng_state[0];
  g_rng_state[3] ^= g_rng_state[1];
  g_rng_state[1] ^= g_rng_state[2];
  g_rng_state[0] ^= g_rng_state[3];
  g_rng_state[2] ^= t;
  g_rng_state[3] = rotl64(g_rng_state[3], 45);
  return result;
}

double ndl_rt_rng_uniform01(void) {
  // 53 random mantissa bits -> uniform in [0, 1).
  return (double)(ndl_rt_rng_next_u64() >> 11) * (1.0 / 9007199254740992.0);
}

void ndl_rt_set_seed(uint64_t seed) {
  uint64_t sm = seed;
  for (int i = 0; i < 4; ++i) g_rng_state[i] = splitmix64(sm);
  if ((g_rng_state[0] | g_rng_state[1] | g_rng_state[2] | g_rng_state[3]) == 0)
    g_rng_state[0] = 1;  // never allow the all-zero state
  g_have_spare = false;
  g_spare = 0.0;
}

double ndl_rt_random_uniform(double a, double b) {
  return a + (b - a) * ndl_rt_rng_uniform01();
}

double ndl_rt_random_gaussian(double mu, double sigma) {
  if (g_have_spare) {
    g_have_spare = false;
    return mu + sigma * g_spare;
  }
  // u1 in (0, 1] so that log(u1) is finite.
  const double u1 = 1.0 - ndl_rt_rng_uniform01();
  const double u2 = ndl_rt_rng_uniform01();
  const double r = std::sqrt(-2.0 * std::log(u1));
  const double theta = kTwoPi * u2;
  g_spare = r * std::sin(theta);
  g_have_spare = true;
  return mu + sigma * r * std::cos(theta);
}

// --- math wrappers (stable ABI for codegen) ----------------------------------

double ndl_rt_sin(double x) { return ::sin(x); }
double ndl_rt_cos(double x) { return ::cos(x); }
double ndl_rt_exp(double x) { return ::exp(x); }
double ndl_rt_log(double x) { return ::log(x); }
double ndl_rt_sqrt(double x) { return ::sqrt(x); }

// --- strings: interned concatenation ----------------------------------------

const char* ndl_rt_str_concat(const char* a, const char* b) {
  static std::mutex mu;
  static std::unordered_set<std::string> intern;  // node-based: stable c_str()
  std::string tmp;
  const size_t la = a ? std::strlen(a) : 0;
  const size_t lb = b ? std::strlen(b) : 0;
  tmp.reserve(la + lb);
  if (a) tmp.append(a, la);
  if (b) tmp.append(b, lb);
  std::lock_guard<std::mutex> lk(mu);
  return intern.insert(std::move(tmp)).first->c_str();
}

// --- tensors (f64, rank 1..3, row-major) -------------------------------------

struct NdlTensor {
  int64_t rank = 1;
  int64_t dims[3] = {1, 1, 1};
  std::vector<double> data;
};

namespace {

void tensor_check(const NdlTensor* t, int64_t want_rank) {
  if (!t) ndl_rt_panic("tensor: null tensor");
  if (t->rank != want_rank) ndl_rt_panic("tensor: rank mismatch");
}

int64_t tensor_flat(const NdlTensor* t, const int64_t* idx) {
  int64_t off = 0;
  for (int64_t k = 0; k < t->rank; ++k) {
    if (idx[k] < 0 || idx[k] >= t->dims[k]) ndl_rt_panic("tensor: index out of range");
    off = off * t->dims[k] + idx[k];
  }
  return off;
}
}  // namespace

NdlTensor* ndl_rt_tensor_new(int64_t rank, const int64_t* dims, double fill) {
  if (rank < 1 || rank > 3) ndl_rt_panic("tensor_new: rank must be 1, 2 or 3");
  if (!dims) ndl_rt_panic("tensor_new: null dims");
  NdlTensor* t = nullptr;
  try {
    t = new NdlTensor();
    t->rank = rank;
    int64_t total = 1;
    for (int64_t k = 0; k < rank; ++k) {
      if (dims[k] <= 0) ndl_rt_panic("tensor_new: dimensions must be positive");
      if (total > INT64_MAX / dims[k]) ndl_rt_panic("tensor_new: size overflow");
      total *= dims[k];
      t->dims[k] = dims[k];
    }
    t->data.assign((size_t)total, fill);
  } catch (const std::bad_alloc&) {
    delete t;
    ndl_rt_panic("out of memory");
  }
  return t;
}

double ndl_rt_tensor_get1(NdlTensor* t, int64_t i0) {
  tensor_check(t, 1);
  const int64_t idx[1] = {i0};
  return t->data[(size_t)tensor_flat(t, idx)];
}

double ndl_rt_tensor_get2(NdlTensor* t, int64_t i0, int64_t i1) {
  tensor_check(t, 2);
  const int64_t idx[2] = {i0, i1};
  return t->data[(size_t)tensor_flat(t, idx)];
}

double ndl_rt_tensor_get3(NdlTensor* t, int64_t i0, int64_t i1, int64_t i2) {
  tensor_check(t, 3);
  const int64_t idx[3] = {i0, i1, i2};
  return t->data[(size_t)tensor_flat(t, idx)];
}

void ndl_rt_tensor_set1(NdlTensor* t, int64_t i0, double v) {
  tensor_check(t, 1);
  const int64_t idx[1] = {i0};
  t->data[(size_t)tensor_flat(t, idx)] = v;
}

void ndl_rt_tensor_set2(NdlTensor* t, int64_t i0, int64_t i1, double v) {
  tensor_check(t, 2);
  const int64_t idx[2] = {i0, i1};
  t->data[(size_t)tensor_flat(t, idx)] = v;
}

void ndl_rt_tensor_set3(NdlTensor* t, int64_t i0, int64_t i1, int64_t i2, double v) {
  tensor_check(t, 3);
  const int64_t idx[3] = {i0, i1, i2};
  t->data[(size_t)tensor_flat(t, idx)] = v;
}

// CSV dump to stdout:
//   # <name>: dims=d0 x d1 (x d2)
//   i0(,i1)(,i2),<value>
void ndl_rt_dump_tensor(const char* name, NdlTensor* t) {
  if (!t) ndl_rt_panic("dump_tensor: null tensor");
  std::fprintf(stdout, "# %s: dims=", name ? name : "tensor");
  for (int64_t k = 0; k < t->rank; ++k)
    std::fprintf(stdout, "%s%lld", k ? " x " : "", (long long)t->dims[k]);
  std::fputc('\n', stdout);
  if (t->rank == 1) {
    for (int64_t i0 = 0; i0 < t->dims[0]; ++i0)
      std::fprintf(stdout, "%lld,%g\n", (long long)i0, t->data[(size_t)i0]);
  } else if (t->rank == 2) {
    for (int64_t i0 = 0; i0 < t->dims[0]; ++i0)
      for (int64_t i1 = 0; i1 < t->dims[1]; ++i1) {
        const int64_t idx[2] = {i0, i1};
        std::fprintf(stdout, "%lld,%lld,%g\n", (long long)i0, (long long)i1,
                     t->data[(size_t)tensor_flat(t, idx)]);
      }
  } else {
    for (int64_t i0 = 0; i0 < t->dims[0]; ++i0)
      for (int64_t i1 = 0; i1 < t->dims[1]; ++i1)
        for (int64_t i2 = 0; i2 < t->dims[2]; ++i2) {
          const int64_t idx[3] = {i0, i1, i2};
          std::fprintf(stdout, "%lld,%lld,%lld,%g\n", (long long)i0, (long long)i1,
                       (long long)i2, t->data[(size_t)tensor_flat(t, idx)]);
        }
  }
  std::fflush(stdout);
}

// --- v2.0: readout (RSOL) -----------------------------------------------------
// Implemented here because the internal NdlTensor layout is private to this TU.

NdlTensor* ndl_rt_get_membrane_potentials(NdlGroup* g) {
  if (!g) ndl_rt_panic("get_membrane_potentials: null group");
  // If a GPU backend owns the live state, pull this group's potentials down
  // first (readout runs on the host while the device ticks).
  ndl_rt_gpu_sync_group_v(g);
  const int64_t dims[1] = {g->size};
  NdlTensor* t = ndl_rt_tensor_new(1, dims, 0.0);
  for (int64_t i = 0; i < g->size; ++i) t->data[(size_t)i] = g->v[(size_t)i];
  return t;
}

NdlTensor* ndl_rt_predict_linear(NdlTensor* x, NdlTensor* w) {
  if (!x) ndl_rt_panic("predict_linear: null input tensor");
  if (!w) ndl_rt_panic("predict_linear: null weight tensor");
  if (x->rank != 1) ndl_rt_panic("predict_linear: input must be a rank-1 tensor");
  if (w->rank != 2) ndl_rt_panic("predict_linear: weights must be a rank-2 tensor");
  const int64_t n = x->dims[0];     // input length
  const int64_t m = w->dims[0];     // output length
  const int64_t k = w->dims[1];     // weight columns
  if (k != n)
    ndl_rt_panic(ndl_rt_sfmt("predict_linear: input length %lld does not match "
                             "weight columns %lld", (long long)n, (long long)k)
                     .c_str());
  const int64_t dims[1] = {m};
  NdlTensor* out = ndl_rt_tensor_new(1, dims, 0.0);
  for (int64_t j = 0; j < m; ++j) {
    double acc = 0.0;
    const double* row = &w->data[(size_t)(j * k)];
    for (int64_t i = 0; i < n; ++i) acc += row[i] * x->data[(size_t)i];
    out->data[(size_t)j] = acc;
  }
  return out;
}

double ndl_rt_vector_l2_norm(NdlTensor* t) {
  if (!t) ndl_rt_panic("vector_l2_norm: null tensor");
  if (t->rank != 1) ndl_rt_panic("vector_l2_norm: tensor must be rank-1");
  double acc = 0.0;
  for (double v : t->data) acc += v * v;
  return std::sqrt(acc);
}
