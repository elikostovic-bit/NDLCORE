// NDL v1.0 — runtime IO: checkpoint save/load (.ndlbin, little-endian) and
// raster CSV export. Creates missing parent directories (mkdir -p semantics).
//
// Checkpoint binary format (all integers/doubles little-endian, no padding —
// fields are written back to back):
//
//   Header:
//     char[4]  magic = "NDLB"
//     u32      version = 1
//     u32      n_groups
//     u32      n_conns
//     u32      n_stdp
//     u32      reserved = 0
//   Group (n_groups records):
//     u32      name_len; u8 name[name_len]
//     i64      size
//     i32      ntype
//     f64      tau, threshold, rest, reset, window_ms
//     f64      v[size]
//   Connection (n_conns records):
//     i32      kind            (0 = dense, 1 = sparse, 2 = one2one)
//     u8       plastic
//     u32      src_len; u8 src[src_len]
//     u32      dst_len; u8 dst[dst_len]
//     dense:   u64 n; f64 w[n]                     (n = n_dst*n_src, dst-major)
//     sparse:  u64 n_rowptr; u64 rowptr[n_rowptr]; u64 nnz;
//              u32 cols[nnz]; f64 vals[nnz]
//     one2one: u64 n; f64 w[n]
//   Stdp (n_stdp records):
//     u32 src_len; u8 src[src_len]; u32 dst_len; u8 dst[dst_len];
//     f64 lr_pot, lr_dep, window_ms
//
// spike_log is intentionally not persisted; loading clears it on all groups.
#include "ndl_rt_internal.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#if defined(_WIN32)
  #include <direct.h>  // _mkdir
#endif

namespace {

// --- mkdir -p -----------------------------------------------------------------

void mkdirs(const std::string& dir) {
  if (dir.empty()) return;
  std::string d = dir;
#if defined(_WIN32)
  // Accept backslash-separated Windows paths by normalizing to '/'.
  for (char& c : d) {
    if (c == '\\') c = '/';
  }
#endif
  std::string cur;
  size_t i = 0;
  if (d[0] == '/') {
    cur = "/";
    i = 1;
  }
  while (i <= d.size()) {
    size_t j = d.find('/', i);
    if (j == std::string::npos) j = d.size();
    if (j > i) {
      // Prepend '/' only between components (or for absolute roots) — never
      // before the first component of a RELATIVE path.
      if (!cur.empty() && !(cur.size() == 1 && cur[0] == '/')) cur += '/';
      cur.append(d, i, j - i);
#if defined(_WIN32)
      ::_mkdir(cur.c_str());  // EEXIST and failures surface at fopen
#else
      ::mkdir(cur.c_str(), 0777);  // EEXIST and failures surface at fopen
#endif
    }
    if (j == d.size()) break;
    i = j + 1;
  }
}

void mkdir_parent_of(const std::string& filepath) {
  const size_t pos = filepath.find_last_of("/\\");
  if (pos == std::string::npos) return;
  mkdirs(filepath.substr(0, pos));
}

// --- writer --------------------------------------------------------------------

struct Writer {
  FILE* f = nullptr;
  bool ok = true;

  void put(const void* p, size_t n) {
    if (ok && std::fwrite(p, 1, n, f) != n) ok = false;
  }
  void u32(uint32_t v) { put(&v, 4); }
  void i32(int32_t v) { put(&v, 4); }
  void u64(uint64_t v) { put(&v, 8); }
  void i64(int64_t v) { put(&v, 8); }
  void f64(double v) { put(&v, 8); }
  void u8(uint8_t v) { put(&v, 1); }
  void name(const std::string& s) {
    u32((uint32_t)s.size());
    put(s.data(), s.size());
  }
  void f64vec(const std::vector<double>& v) {
    put(v.data(), v.size() * sizeof(double));
  }
};

// --- reader ----------------------------------------------------------------------

[[noreturn]] void truncated() {
  ndl_rt_panic("load_checkpoint: truncated or corrupt file");
  std::abort();  // unreachable (ndl_rt_panic exits); silences -Wreturn-type
}

struct Reader {
  FILE* f = nullptr;

  void need(size_t n, void* dst) {
    if (std::fread(dst, 1, n, f) != n) truncated();
  }
  uint32_t u32() {
    uint32_t v = 0;
    need(4, &v);
    return v;
  }
  int32_t i32() {
    int32_t v = 0;
    need(4, &v);
    return v;
  }
  uint64_t u64() {
    uint64_t v = 0;
    need(8, &v);
    return v;
  }
  int64_t i64() {
    int64_t v = 0;
    need(8, &v);
    return v;
  }
  double f64() {
    double v = 0;
    need(8, &v);
    return v;
  }
  uint8_t u8() {
    uint8_t v = 0;
    need(1, &v);
    return v;
  }
  std::string name() {
    const uint32_t len = u32();
    if (len > (1u << 20)) truncated();  // sanity bound against corrupt files
    std::string s(len, '\0');
    if (len) need(len, &s[0]);
    return s;
  }
};

NdlGroup* find_group_by_name(const std::vector<NdlGroup*>& groups,
                             const std::string& name) {
  for (NdlGroup* g : groups)
    if (g->name == name) return g;
  return nullptr;
}

}  // namespace

// --- public API -------------------------------------------------------------------

void ndl_rt_save_checkpoint(const char* path) {
  if (!path || !*path) ndl_rt_panic("save_checkpoint: empty path");
  // If a GPU backend owns the live state, pull it back to the host mirrors
  // first (weights, v, trace). No-op on the CPU path. Ф9.4f: SHARED on
  // struct_mx — a concurrent grow can no longer swap/free the CSR buffers
  // mid-serialization (was an acknowledged mid-tick race); v/spike_log remain
  // best-effort (values may tear, buffers cannot move under us).
  ndl_rt_gpu_sync_down();
  SimState& S = ndl_rt_sim();
  std::shared_lock<std::shared_mutex> struct_lk(S.struct_mx);

  mkdir_parent_of(path);
  FILE* f = std::fopen(path, "wb");
  if (!f)
    ndl_rt_panic(ndl_rt_sfmt("save_checkpoint: cannot open '%s'", path).c_str());

  Writer w;
  w.f = f;
  w.put("NDLB", 4);
  w.u32(1);                            // version
  w.u32((uint32_t)S.groups.size());
  w.u32((uint32_t)S.conns.size());
  w.u32((uint32_t)S.stdps.size());
  w.u32(0);                            // reserved

  for (const NdlGroup* g : S.groups) {
    w.name(g->name);
    w.i64(g->size);
    w.i32(g->ntype);
    w.f64(g->tau);
    w.f64(g->threshold);
    w.f64(g->rest);
    w.f64(g->reset);
    w.f64(g->window_ms);
    w.f64vec(g->v);
  }

  for (const Connection* c : S.conns) {
    w.i32(c->kind);
    w.u8(c->plastic ? 1 : 0);
    w.name(c->src ? c->src->name : std::string());
    w.name(c->dst ? c->dst->name : std::string());
    if (c->kind == CONN_DENSE) {
      w.u64((uint64_t)c->w.size());
      w.f64vec(c->w);
    } else if (c->kind == CONN_SPARSE) {
      w.u64((uint64_t)c->rowptr.size());
      w.put(c->rowptr.data(), c->rowptr.size() * sizeof(uint64_t));
      w.u64((uint64_t)c->cols.size());
      w.put(c->cols.data(), c->cols.size() * sizeof(uint32_t));
      w.f64vec(c->vals);
    } else {
      w.u64((uint64_t)c->w.size());
      w.f64vec(c->w);
    }
  }

  for (const StdpCfg* p : S.stdps) {
    w.name(p->src ? p->src->name : std::string());
    w.name(p->dst ? p->dst->name : std::string());
    w.f64(p->lr_pot);
    w.f64(p->lr_dep);
    w.f64(p->window_ms);
  }

  const bool ok = w.ok;
  const bool closed = std::fclose(f) == 0;
  if (!ok || !closed)
    ndl_rt_panic(ndl_rt_sfmt("save_checkpoint: write failed for '%s'", path).c_str());
}

void ndl_rt_load_checkpoint(const char* path) {
  if (!path || !*path) ndl_rt_panic("load_checkpoint: empty path");
  // Ф9.4f: UNIQUE — load rebuilds every group/connection (frees and replaces
  // all CSR buffers); must exclude tick bodies and other structural ops.
  std::unique_lock<std::shared_mutex> struct_lk(ndl_rt_sim().struct_mx);
  FILE* f = std::fopen(path, "rb");
  if (!f)
    ndl_rt_panic(ndl_rt_sfmt("load_checkpoint: cannot open '%s'", path).c_str());
  Reader r;
  r.f = f;

  char magic[4] = {0, 0, 0, 0};
  r.need(4, magic);
  if (std::memcmp(magic, "NDLB", 4) != 0)
    ndl_rt_panic("load_checkpoint: bad magic (not an NDL checkpoint)");
  const uint32_t version = r.u32();
  if (version != 1)
    ndl_rt_panic(
        ndl_rt_sfmt("load_checkpoint: unsupported version %u", version).c_str());
  const uint32_t n_groups = r.u32();
  const uint32_t n_conns = r.u32();
  const uint32_t n_stdp = r.u32();
  (void)r.u32();  // reserved

  SimState& S = ndl_rt_sim();

  // --- groups: patch existing (by name) or create ---
  for (uint32_t k = 0; k < n_groups; ++k) {
    const std::string name = r.name();
    const int64_t size = r.i64();
    const int32_t ntype = r.i32();
    const double tau = r.f64();
    const double threshold = r.f64();
    const double rest = r.f64();
    const double reset = r.f64();
    const double window_ms = r.f64();
    if (size <= 0 || !(tau > 0.0))
      ndl_rt_panic(ndl_rt_sfmt("load_checkpoint: invalid group '%s'", name.c_str())
                       .c_str());

    NdlGroup* g = find_group_by_name(S.groups, name);
    try {
      if (g) {
        if (g->size != size) {
          ndl_rt_panic(ndl_rt_sfmt(
                           "load_checkpoint: group '%s' size mismatch (have %lld, file %lld)",
                           name.c_str(), (long long)g->size, (long long)size)
                           .c_str());
        }
      } else {
        g = new NdlGroup();
        g->name = name;
        g->size = size;
        g->v.assign((size_t)size, rest);
        g->i_syn.assign((size_t)size, 0.0);
        g->trace.assign((size_t)size, 0.0);
        g->spikes.assign((size_t)size, 0);
        g->prev_spikes.assign((size_t)size, 0);
        S.groups.push_back(g);
      }
      g->ntype = ntype;
      g->tau = tau;
      g->threshold = threshold;
      g->rest = rest;
      g->reset = reset;
      g->window_ms = window_ms;
      g->v.assign((size_t)size, 0.0);
      r.need((size_t)size * sizeof(double), g->v.data());
    } catch (const std::bad_alloc&) {
      ndl_rt_panic("out of memory");
    }
  }

  // --- connections: replace any existing link for the same pair ---
  for (uint32_t k = 0; k < n_conns; ++k) {
    const int32_t kind = r.i32();
    const uint8_t plastic = r.u8();
    const std::string src_name = r.name();
    const std::string dst_name = r.name();
    NdlGroup* src = find_group_by_name(S.groups, src_name);
    NdlGroup* dst = find_group_by_name(S.groups, dst_name);
    if (!src || !dst)
      ndl_rt_panic(ndl_rt_sfmt("load_checkpoint: unknown group in connection '%s' -> '%s'",
                               src_name.c_str(), dst_name.c_str())
                       .c_str());
    if (kind != CONN_DENSE && kind != CONN_SPARSE && kind != CONN_ONE2ONE)
      truncated();

    Connection* c = nullptr;
    try {
      c = new Connection();
      c->kind = kind;
      c->src = src;
      c->dst = dst;
      c->plastic = plastic != 0;
      if (kind == CONN_DENSE) {
        const uint64_t n = r.u64();
        if (n != (uint64_t)src->size * (uint64_t)dst->size)
          truncated();
        c->w.assign((size_t)n, 0.0);
        r.need((size_t)n * sizeof(double), c->w.data());
      } else if (kind == CONN_SPARSE) {
        const uint64_t n_rowptr = r.u64();
        if (n_rowptr != (uint64_t)dst->size + 1) truncated();
        c->rowptr.assign((size_t)n_rowptr, 0);
        r.need((size_t)n_rowptr * sizeof(uint64_t), c->rowptr.data());
        const uint64_t nnz = r.u64();
        c->cols.resize((size_t)nnz);
        r.need((size_t)nnz * sizeof(uint32_t), c->cols.data());
        c->vals.resize((size_t)nnz);
        r.need((size_t)nnz * sizeof(double), c->vals.data());
        // CSR sanity
        if (c->rowptr[0] != 0 || c->rowptr.back() != nnz) truncated();
        for (size_t j = 0; j + 1 < c->rowptr.size(); ++j)
          if (c->rowptr[j + 1] < c->rowptr[j]) truncated();
      } else {
        const uint64_t n = r.u64();
        if (n != (uint64_t)src->size) truncated();
        c->w.assign((size_t)n, 0.0);
        r.need((size_t)n * sizeof(double), c->w.data());
      }
    } catch (const std::bad_alloc&) {
      delete c;
      ndl_rt_panic("out of memory");
    }
    // Replace existing connection(s) for this pair.
    for (size_t q = S.conns.size(); q-- > 0;) {
      if (S.conns[q]->src == src && S.conns[q]->dst == dst) {
        delete S.conns[q];
        S.conns.erase(S.conns.begin() + (long)q);
      }
    }
    S.conns.push_back(c);
    S.topology_version++;
  }

  // --- STDP configs: replace per pair ---
  for (uint32_t k = 0; k < n_stdp; ++k) {
    const std::string src_name = r.name();
    const std::string dst_name = r.name();
    const double lr_pot = r.f64();
    const double lr_dep = r.f64();
    const double window_ms = r.f64();
    NdlGroup* src = find_group_by_name(S.groups, src_name);
    NdlGroup* dst = find_group_by_name(S.groups, dst_name);
    if (!src || !dst)
      ndl_rt_panic(ndl_rt_sfmt("load_checkpoint: unknown group in STDP '%s' -> '%s'",
                               src_name.c_str(), dst_name.c_str())
                       .c_str());
    StdpCfg* cfg = nullptr;
    for (StdpCfg* p : S.stdps)
      if (p->src == src && p->dst == dst) { cfg = p; break; }
    try {
      if (cfg) {
        cfg->lr_pot = lr_pot;
        cfg->lr_dep = lr_dep;
        cfg->window_ms = window_ms;
      } else {
        cfg = new StdpCfg();
        cfg->src = src;
        cfg->dst = dst;
        cfg->lr_pot = lr_pot;
        cfg->lr_dep = lr_dep;
        cfg->window_ms = window_ms;
        S.stdps.push_back(cfg);
      }
      src->window_ms = window_ms;
      dst->window_ms = window_ms;
    } catch (const std::bad_alloc&) {
      ndl_rt_panic("out of memory");
    }
  }

  std::fclose(f);

  // Noise RNG states are not part of the checkpoint; re-seed where needed.
  for (NdlGroup* g : S.groups)
    if ((int64_t)g->noise_rng.size() != g->size) {
      g->noise_rng.resize((size_t)g->size);
      for (auto& s : g->noise_rng) s = ndl_rt_rng_next_u64();
    }

  // Spike history is not part of the checkpoint.
  for (NdlGroup* g : S.groups) g->spike_log.clear();
  // Device buffers (if any) no longer match the loaded topology/weights.
  ndl_rt_gpu_backend_invalidate();
}

void ndl_rt_export_raster(NdlGroup* g, const char* path) {
  if (!g) ndl_rt_panic("export_raster: null group");
  if (!path || !*path) ndl_rt_panic("export_raster: empty path");

  mkdir_parent_of(path);
  FILE* f = std::fopen(path, "w");
  if (!f)
    ndl_rt_panic(ndl_rt_sfmt("export_raster: cannot open '%s'", path).c_str());

  std::fprintf(f, "t_ms,neuron_id\n");
  for (const auto& sp : g->spike_log)
    std::fprintf(f, "%g,%lld\n", sp.first, (long long)sp.second);

  if (std::fflush(f) != 0 || std::fclose(f) != 0)
    ndl_rt_panic(ndl_rt_sfmt("export_raster: write failed for '%s'", path).c_str());
}
