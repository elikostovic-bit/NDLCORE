// NDL v1.0 — runtime scheduler: fixed thread pool + deterministic parallel_for.
//
// Pool layout: N-1 workers + the calling thread acting as coordinator (N =
// hardware concurrency, at least 1). The pool is spawned lazily on the first
// parallel_for call and joined at process exit.
//
// Determinism: ndl_rt_parallel_for partitions [begin, end) into contiguous
// chunks deterministically (boundaries depend only on begin/end/minGrain and
// worker count). Chunks are disjoint, so execution order cannot affect the
// result as long as body() only mutates data belonging to its own sub-range —
// which is the only way the network layer uses it.
//
// Ф9.4f: ПЕРЕПИСАН. Прежняя схема (атомарные next/done + воровство чанков +
// два предиката на одном cv) давала редкий дедлок барьера и чтение мусора
// под воркерами на 12800 (segfault prop_sparse_body: cols[k] мусор;
// воспроизводимо W2W-teach ×12800, ловится NDL_POOL=0 — без пула чисто).
// Новая схема — учебниковая волна БЕЗ воровства:
//   * слот = один непрерывный чанк (воркер s → чанк s, координатор → чанк W);
//   * публикация волны: ++gen под mutex (ничего не сбрасывается — нет
//     гонки publish-vs-в-полёте);
//   * воркер спит, пока gen не обгонит его last_seen; исполняет свой чанк;
//   * барьер: done_count == nslots (считаются ВСЕ слоты, включая
//     координатор); репартиция lo/hi следующей волны возможна только после
//     полного выхода предыдущей — буферы чанков стабильны в волне.
// Один mutex, один cv, ноль «хитрых» атомарных интерливингов.
#include "ndl_rt_internal.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct Pool {
  std::mutex mu;
  std::condition_variable cv;

  // Wave state (stable while workers execute; rewritten only after retire).
  void (*body)(int64_t, int64_t, void*) = nullptr;
  void* ctx = nullptr;
  std::vector<int64_t> lo, hi;  // one contiguous chunk per slot
  size_t nslots = 0;            // workers.size() + 1 (coordinator)
  uint64_t gen = 0;             // current wave number (published)
  size_t done_count = 0;        // slots finished the current wave
  bool stop = false;
  bool spawned = false;

  std::vector<std::thread> workers;
};

Pool g_pool;

// Worker for slot s (0..W-1). The coordinator executes slot W inline.
void worker_main(unsigned slot) {
  Pool& p = g_pool;
  uint64_t seen = 0;
  for (;;) {
    // Wait for a wave we have not executed yet (or shutdown).
    {
      std::unique_lock<std::mutex> lk(p.mu);
      p.cv.wait(lk, [&] { return p.stop || p.gen > seen; });
      if (p.stop) return;
      seen = p.gen;
    }
    // Execute our chunk. lo/hi are stable: the next wave may only be
    // published after this one fully retires (done_count == nslots).
    const int64_t b = p.lo[slot], e = p.hi[slot];
    if (e > b) p.body(b, e, p.ctx);
    // Retire: count ourselves in; the last slot wakes the coordinator.
    {
      std::lock_guard<std::mutex> lk(p.mu);
      if (++p.done_count == p.nslots) p.cv.notify_all();
    }
  }
}

void ensure_workers() {
  Pool& p = g_pool;
  std::lock_guard<std::mutex> lk(p.mu);
  if (p.spawned) return;
  p.spawned = true;
  unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) hw = 1;
  unsigned n = hw > 1 ? hw - 1 : 0;  // N-1 workers, coordinator is thread N
  if (n > 32) n = 32;                // sanity cap
  p.nslots = (size_t)n + 1;          // + coordinator slot
  p.workers.reserve(n);
  for (unsigned k = 0; k < n; ++k) p.workers.emplace_back(worker_main, k);
}

}  // namespace

void ndl_rt_parallel_for(int64_t begin, int64_t end, int64_t minGrain,
                         void (*body)(int64_t lo, int64_t hi, void* ctx),
                         void* ctx) {
  if (!body || end <= begin) return;
  if (minGrain < 1) minGrain = 1;

  Pool& p = g_pool;
  ensure_workers();

  // Ф9.4f-отладка: NDL_POOL=0 — весь work синхронно на координаторе.
  if (const char* e = std::getenv("NDL_POOL"); e && e[0] == '0') {
    body(begin, end, ctx);
    return;
  }
  if (p.workers.empty()) {  // single-core machine: coordinator only
    body(begin, end, ctx);
    return;
  }

  const int64_t n = end - begin;
  int64_t nch = (n + minGrain - 1) / minGrain;
  if (nch > (int64_t)p.nslots) nch = (int64_t)p.nslots;
  if (nch < 1) nch = 1;
  if (nch == 1) {  // too small to parallelize
    body(begin, end, ctx);
    return;
  }

  // Deterministic partition into nslots contiguous pieces (some may be
  // empty when the range is smaller than the slot count; empty chunks are
  // skipped by the e > b guard in both the workers and the coordinator).
  p.lo.assign(p.nslots, 0);
  p.hi.assign(p.nslots, 0);
  {
    const int64_t base = n / nch;
    const int64_t rem = n % nch;
    int64_t off = begin;
    for (int64_t k = 0; k < nch; ++k) {
      const int64_t len = base + (k < rem ? 1 : 0);
      p.lo[(size_t)k] = off;
      p.hi[(size_t)k] = off + len;
      off += len;
    }
    // Slots >= nch stay empty (lo == hi == 0).
  }

  // Publish the wave. done_count of the PREVIOUS wave is already == nslots
  // (retire barrier), so resetting it here is race-free: no worker can be
  // inside a body at this point.
  {
    std::lock_guard<std::mutex> lk(p.mu);
    p.body = body;
    p.ctx = ctx;
    p.done_count = 0;
    ++p.gen;
  }
  p.cv.notify_all();

  // The coordinator executes the LAST slot inline and counts itself in.
  {
    const size_t s = p.nslots - 1;
    const int64_t b = p.lo[s], e = p.hi[s];
    if (e > b) body(b, e, ctx);
    std::lock_guard<std::mutex> lk(p.mu);
    ++p.done_count;
  }

  // Retire barrier: every slot (workers + coordinator) increments done_count.
  {
    std::unique_lock<std::mutex> lk(p.mu);
    p.cv.wait(lk, [&] { return p.done_count == p.nslots; });
  }
  // Wake the workers for their next wait round (they may be parked waiting
  // for the next gen; nothing needed for correctness of THIS call).
  p.cv.notify_all();
}
