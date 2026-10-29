// Host-side row lookup engine for the PLE `nvmecache` offload backend.
//
// The PLE n-gram table stays in the checkpoint's own safetensors shards and
// the GPU never dereferences those pages. A gather hands this engine a block
// of row ids in pinned memory; the engine serves each row from a DRAM page
// cache or from O_DIRECT preads on the shard fds, writes the raw rows into
// pinned staging, and the (captured) graph copies them to the device. See
// qwen4_exp_ple_nvme.py for the CUDA side and qwen4_exp_ple_table.py for
// segment discovery.
//
// ple_lookup_run is a cudaLaunchHostFunc target: it runs on a driver thread
// with the launching stream blocked, so it may call no CUDA API and must not
// allocate (malloc blocked behind an allocator lock held by a stream-holding
// thread would deadlock the driver). Everything a run needs -- its job
// cells, sort keys and wait primitive -- is preallocated at
// ple_lookup_create; the read fan-out rides the store's worker pool and the
// host function returns once every row of the run is in staging (a blocking
// wait is fine here: only the launching stream is stalled, and the next
// decoder layer's work is already queued ahead of the copy back).
//
// Cache: sets of PLE_WAYS 4 KiB slots tagged by (segment, page). The page is
// both the read unit and the cache unit: rows are 120-320 B, so ~13-34 rows
// share a slot. A slot is CAS-claimed before it is written and published
// with its final tag afterwards; a reader validates the tag before and
// after its copy. The table is immutable, so a tag that survived the copy
// means the bytes did too.

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PLE_PAGE_SHIFT 12
#define PLE_PAGE (1u << PLE_PAGE_SHIFT)
/* A job covers rows whose starting pages span at most this many pages, so
 * one buffer of PLE_JOB_PAGES + 1 pages always contains those rows in full
 * (the +1 carries a row straddling the last page). */
#define PLE_JOB_PAGES 8
#define PLE_WAYS 8
#define PLE_MAX_SEGMENTS 4096
#define PLE_MAX_ROWS (1u << 20)

/* Tag word: 2 state bits | 16 segment | 46 file page (46 bits = 64 TiB per
 * shard file; safetensors shards are single-digit GiB). */
#define PLE_TAG_INVALID 0u
#define PLE_TAG_CLAIMED 1u
#define PLE_TAG_READY 2u
#define PLE_STATE_SHIFT 62
#define PLE_TAG_LOW_MASK (((uint64_t)1 << PLE_STATE_SHIFT) - 1)

/* Sort key: segment (12) | starting page (32) | row slot (20). */
#define PLE_KEY_SEG_SHIFT 52
#define PLE_KEY_PAGE_SHIFT 20
#define PLE_KEY_ROW_MASK ((1u << 20) - 1)

typedef struct {
    int fd;         /* O_RDONLY; O_DIRECT unless the filesystem refused it */
    int direct;     /* whether fd is O_DIRECT */
    char *path;     /* owned; kept to reopen buffered if O_DIRECT reads fail */
    uint64_t begin; /* absolute file offset of row_start */
    uint64_t row_start;
    uint64_t row_end; /* exclusive; global vocabulary rows */
} ple_segment;

typedef struct ple_lookup ple_lookup;

typedef struct {
    ple_lookup *lk;
    uint64_t cover_start; /* page-aligned file offset to pread */
    uint32_t cover_bytes;
    uint32_t seg;
    uint32_t key_lo, key_hi; /* sorted-key range whose rows this read serves */
} ple_job;

typedef struct ple_jobcell {
    ple_job job;
    struct ple_jobcell *next;
} ple_jobcell;

struct ple_lookup {
    struct ple_store *store;
    const int64_t *ids; /* pinned; contents valid when the host fn runs */
    const uint32_t *count; /* pinned row count; a graph's D2H ids copy fills it */
    uint8_t *rows;      /* pinned staging, `capacity` rows of row_bytes */
    uint32_t capacity;
    uint64_t vocab_start, vocab_end; /* global rows served, else zeroed */
    uint64_t *keys;     /* scratch [capacity] */
    ple_jobcell *cells; /* scratch [capacity + 1] */
    _Atomic int32_t pending; /* jobs not yet drained */
    _Atomic uint32_t status; /* 0, else the first errno seen */
};

typedef struct ple_store {
    ple_segment *segs;
    uint32_t nseg;
    uint32_t row_bytes;

    /* page cache */
    _Atomic uint64_t *tags; /* nslots tag words, PLE_WAYS consecutive per set */
    uint8_t *data;          /* nslots * PLE_PAGE */
    uint32_t nslots;
    uint32_t nsets;
    uint32_t *rr;           /* set cursor for eviction */

    /* worker pool + job queue */
    uint32_t qd;
    pthread_t *workers;
    int stop;
    pthread_mutex_t qmtx;
    pthread_cond_t qcond;
    ple_jobcell *qhead, *qtail;

    /* aligned read-buffer pool */
    uint8_t *bufs;
    uint32_t buf_stride;
    uint32_t *free_stack;
    uint32_t free_top;
    pthread_mutex_t bmtx;
    pthread_cond_t bcond;

    _Atomic uint64_t hits, misses, pages_read;
} ple_store;

static inline uint64_t ple_tag(uint32_t state, uint32_t seg, uint64_t page) {
    return ((uint64_t)state << PLE_STATE_SHIFT) | ((uint64_t)seg << 46) | page;
}

static inline uint32_t ple_hash(uint32_t seg, uint64_t page, uint32_t nsets) {
    uint64_t h = (seg + 1) * 0x9E3779B97F4A7C15ull ^
                 (page + 1) * 0xC2B2AE3D27D4EB4Full;
    return (uint32_t)((h >> 24) % nsets);
}

static uint8_t *ple_aligned(size_t n) {
    void *p = NULL;
    if (n == 0) n = PLE_PAGE;
    if (posix_memalign(&p, PLE_PAGE, n) != 0) return NULL;
    return p;
}

ple_store *ple_nvme_open(const char **paths, const uint64_t *begins,
                         const uint64_t *row_starts, const uint64_t *row_ends,
                         uint32_t nseg, uint32_t row_bytes,
                         uint64_t cache_bytes, uint32_t qd, char *err,
                         size_t errlen) {
    err[0] = '\0';
    if (row_bytes == 0 || row_bytes > PLE_PAGE) {
        snprintf(err, errlen, "row_bytes %u not in (0, %u]", row_bytes, PLE_PAGE);
        return NULL;
    }
    if (nseg == 0 || nseg > PLE_MAX_SEGMENTS) {
        snprintf(err, errlen, "nseg %u out of range", nseg);
        return NULL;
    }
    ple_store *s = calloc(1, sizeof(*s));
    s->segs = calloc(nseg, sizeof(ple_segment));
    s->row_bytes = row_bytes;
    s->nseg = nseg;
    for (uint32_t i = 0; i < nseg; i++) {
        if (i && row_starts[i] != s->segs[i - 1].row_end) {
            snprintf(err, errlen, "segments not contiguous at row %llu",
                     (unsigned long long)row_starts[i]);
            return NULL;
        }
        s->segs[i].path = strdup(paths[i]);
        s->segs[i].begin = begins[i];
        s->segs[i].row_start = row_starts[i];
        s->segs[i].row_end = row_ends[i];
        s->segs[i].fd = open(paths[i], O_RDONLY | O_DIRECT);
        s->segs[i].direct = s->segs[i].fd >= 0;
        if (s->segs[i].fd < 0) s->segs[i].fd = open(paths[i], O_RDONLY);
        if (s->segs[i].fd < 0) {
            snprintf(err, errlen, "open %s: %s", paths[i], strerror(errno));
            return NULL;
        }
    }
    s->nslots = (uint32_t)(cache_bytes / PLE_PAGE);
    if (s->nslots < PLE_WAYS * 64) s->nslots = PLE_WAYS * 64; /* 2 MiB floor */
    s->nslots -= s->nslots % PLE_WAYS;
    s->nsets = s->nslots / PLE_WAYS;
    s->tags = calloc(s->nslots, sizeof(_Atomic uint64_t));
    s->rr = calloc(s->nsets, sizeof(uint32_t));
    s->data = ple_aligned((size_t)s->nslots * PLE_PAGE);
    if (!s->tags || !s->rr || !s->data) {
        snprintf(err, errlen, "page cache alloc of %u pages failed", s->nslots);
        return NULL;
    }

    s->qd = qd ? qd : 32;
    s->workers = calloc(s->qd, sizeof(pthread_t));
    s->buf_stride = (PLE_JOB_PAGES + 1) * PLE_PAGE;
    s->bufs = ple_aligned((size_t)s->qd * s->buf_stride);
    s->free_stack = calloc(s->qd, sizeof(uint32_t));
    pthread_mutex_init(&s->qmtx, NULL);
    pthread_cond_init(&s->qcond, NULL);
    pthread_mutex_init(&s->bmtx, NULL);
    pthread_cond_init(&s->bcond, NULL);
    s->free_top = s->qd;
    for (uint32_t i = 0; i < s->qd; i++) s->free_stack[i] = i;
    extern void *ple_worker_main(void *);
    for (uint32_t i = 0; i < s->qd; i++)
        pthread_create(&s->workers[i], NULL, ple_worker_main, s);
    return s;
}

void ple_nvme_close(ple_store *s) {
    if (!s) return;
    pthread_mutex_lock(&s->qmtx);
    s->stop = 1;
    pthread_cond_broadcast(&s->qcond);
    pthread_mutex_unlock(&s->qmtx);
    for (uint32_t i = 0; i < s->qd; i++) pthread_join(s->workers[i], NULL);
    for (uint32_t i = 0; i < s->nseg; i++) {
        close(s->segs[i].fd);
        free(s->segs[i].path);
    }
    free(s->segs); free((void *)s->tags); free(s->data); free(s->rr);
    free(s->bufs); free(s->free_stack); free(s->workers); free(s);
}

ple_lookup *ple_lookup_create(ple_store *s, const int64_t *ids,
                              const uint32_t *count, uint8_t *rows,
                              uint32_t capacity, uint64_t vocab_start,
                              uint64_t vocab_end, char *err, size_t errlen) {
    err[0] = '\0';
    if (capacity == 0 || capacity > PLE_MAX_ROWS) {
        snprintf(err, errlen, "lookup capacity %u not in (0, %u]", capacity,
                 PLE_MAX_ROWS);
        return NULL;
    }
    ple_lookup *lk = calloc(1, sizeof(*lk));
    lk->keys = calloc(capacity, sizeof(uint64_t));
    lk->cells = calloc(capacity + 1, sizeof(ple_jobcell));
    if (!lk->keys || !lk->cells) {
        snprintf(err, errlen, "lookup scratch for %u rows failed", capacity);
        free(lk->keys); free(lk->cells); free(lk);
        return NULL;
    }
    lk->store = s;
    lk->ids = ids;
    lk->count = count;
    lk->rows = rows;
    lk->capacity = capacity;
    lk->vocab_start = vocab_start;
    lk->vocab_end = vocab_end;
    return lk;
}

void ple_lookup_destroy(ple_lookup *lk) {
    if (!lk) return;
    free(lk->keys); free(lk->cells); free(lk);
}

uint32_t ple_lookup_status(ple_lookup *lk) { return atomic_load(&lk->status); }

void ple_store_stats(ple_store *s, uint64_t *hits, uint64_t *misses,
                     uint64_t *pages_read) {
    *hits = atomic_load(&s->hits);
    *misses = atomic_load(&s->misses);
    *pages_read = atomic_load(&s->pages_read);
}

/* last segment whose row_start <= row; caller has range-checked */
static uint32_t ple_seg_of(ple_store *s, uint64_t row) {
    int lo = 0, hi = (int)s->nseg - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (s->segs[mid].row_start <= row) lo = mid; else hi = mid - 1;
    }
    return (uint32_t)lo;
}

static uint64_t ple_row_off(ple_store *s, uint32_t seg, uint64_t row) {
    return s->segs[seg].begin + (row - s->segs[seg].row_start) * s->row_bytes;
}

static int ple_row_from_cache(ple_store *s, uint32_t seg, uint64_t file_off,
                              uint8_t *dst) {
    uint32_t rb = s->row_bytes;
    uint64_t page = file_off >> PLE_PAGE_SHIFT;
    uint32_t off = (uint32_t)(file_off & (PLE_PAGE - 1));
    uint32_t two = (off + rb > PLE_PAGE);
    uint64_t want0 = ple_tag(PLE_TAG_READY, seg, page);
    uint64_t want1 = ple_tag(PLE_TAG_READY, seg, page + 1);
    uint32_t set0 = ple_hash(seg, page, s->nsets), way0 = PLE_WAYS;
    uint32_t set1 = 0, way1 = PLE_WAYS;
    for (uint32_t w = 0; w < PLE_WAYS; w++) {
        uint64_t t = atomic_load_explicit(&s->tags[set0 * PLE_WAYS + w],
                                          memory_order_acquire);
        if (t == want0) { way0 = w; break; }
    }
    if (way0 == PLE_WAYS) return -1;
    if (two) {
        set1 = ple_hash(seg, page + 1, s->nsets);
        for (uint32_t w = 0; w < PLE_WAYS; w++) {
            uint64_t t = atomic_load_explicit(&s->tags[set1 * PLE_WAYS + w],
                                              memory_order_acquire);
            if (t == want1) { way1 = w; break; }
        }
        if (way1 == PLE_WAYS) return -1;
    }
    uint8_t *d0 = s->data + (size_t)(set0 * PLE_WAYS + way0) * PLE_PAGE + off;
    if (!two) {
        memcpy(dst, d0, rb);
    } else {
        uint8_t *d1 = s->data + (size_t)(set1 * PLE_WAYS + way1) * PLE_PAGE;
        memcpy(dst, d0, PLE_PAGE - off);
        memcpy(dst + (PLE_PAGE - off), d1, rb - (PLE_PAGE - off));
    }
    /* Re-check the tags after the copy: an eviction between validation and
     * memcpy could splice another (seg, page) into the row. The slot the
     * row's own page landed in must still carry the same tag. */
    uint64_t t2 = atomic_load_explicit(&s->tags[set0 * PLE_WAYS + way0],
                                       memory_order_acquire);
    if (t2 != want0) return -1;
    if (two) {
        uint64_t t3 = atomic_load_explicit(&s->tags[set1 * PLE_WAYS + way1],
                                           memory_order_acquire);
        if (t3 != want1) return -1;
    }
    return 0;
}

/* Publish a page read into `src` (cover_start + i*PLE_PAGE). Claim first so
 * readers never observe READY while the copy runs; a lost CAS means another
 * publisher is writing this page and its bytes are equally correct. */
static void ple_page_publish(ple_store *s, uint32_t seg, uint64_t page,
                             const uint8_t *src) {
    uint32_t set = ple_hash(seg, page, s->nsets);
    uint64_t want = ple_tag(PLE_TAG_READY, seg, page);
    for (uint32_t w = 0; w < PLE_WAYS; w++) {
        uint32_t i = set * PLE_WAYS + w;
        uint64_t old = atomic_load_explicit(&s->tags[i], memory_order_relaxed);
        if (old == want) return; /* already resident */
        if ((old >> PLE_STATE_SHIFT) == PLE_TAG_INVALID) {
            uint64_t claim = ple_tag(PLE_TAG_CLAIMED, seg, 0);
            if (atomic_compare_exchange_strong_explicit(
                    &s->tags[i], &old, claim, memory_order_relaxed,
                    memory_order_relaxed)) {
                memcpy(s->data + (size_t)i * PLE_PAGE, src, PLE_PAGE);
                atomic_store_explicit(&s->tags[i], want, memory_order_release);
                return;
            }
        }
    }
    /* no invalid way: evict by per-set round robin (a claim in flight just
     * loses; the next job re-publishes if it still needs the page) */
    uint32_t i = set * PLE_WAYS + (s->rr[set]++ % PLE_WAYS);
    uint64_t old = atomic_load_explicit(&s->tags[i], memory_order_relaxed);
    uint64_t claim = ple_tag(PLE_TAG_CLAIMED, seg, 0);
    if (!atomic_compare_exchange_strong_explicit(&s->tags[i], &old, claim,
                                                 memory_order_relaxed,
                                                 memory_order_relaxed))
        return;
    memcpy(s->data + (size_t)i * PLE_PAGE, src, PLE_PAGE);
    atomic_store_explicit(&s->tags[i], want, memory_order_release);
}

static int ple_cmp_key(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static void ple_job_finish(ple_lookup *lk) {
    atomic_fetch_sub(&lk->pending, 1); /* the run spins on this */
}

static void ple_job_run(ple_store *s, ple_job *j) {
    ple_lookup *lk = j->lk;
    ple_segment *sg = &s->segs[j->seg];
    pthread_mutex_lock(&s->bmtx);
    while (s->free_top == 0) pthread_cond_wait(&s->bcond, &s->bmtx);
    uint32_t bidx = s->free_stack[--s->free_top];
    pthread_mutex_unlock(&s->bmtx);
    uint8_t *buf = s->bufs + (size_t)bidx * s->buf_stride;

    ssize_t r = 0, got = 0;
    while ((size_t)got < j->cover_bytes) {
        r = pread(sg->fd, buf + got, j->cover_bytes - got,
                  (off_t)(j->cover_start + got));
        if (r > 0) { got += r; continue; }
        if (r < 0 && sg->direct && errno == EINVAL) {
            /* fs refused O_DIRECT reads (opened anyway via fallback flags):
             * reopen buffered -- the fd itself carries the O_DIRECT flag. */
            int fd = open(sg->path, O_RDONLY);
            if (fd >= 0) {
                close(sg->fd);
                sg->fd = fd;
            }
            sg->direct = 0;
            continue;
        }
        if (r < 0 && errno == EINTR) continue;
        if (r == 0) break; /* EOF: short tail, zero-filled below */
        uint32_t expect = 0;
        atomic_compare_exchange_strong(&lk->status, &expect, (uint32_t)errno);
        break;
    }
    if ((size_t)got < j->cover_bytes)
        memset(buf + got, 0, j->cover_bytes - (size_t)got);
    atomic_fetch_add(&s->pages_read,
                     (j->cover_bytes + PLE_PAGE - 1) >> PLE_PAGE_SHIFT);

    for (uint32_t k = j->key_lo; k < j->key_hi; k++) {
        uint64_t key = lk->keys[k];
        uint32_t row_slot = (uint32_t)(key & PLE_KEY_ROW_MASK);
        uint64_t id = (uint64_t)lk->ids[row_slot];
        uint64_t fo = ple_row_off(s, j->seg, id);
        memcpy(lk->rows + (size_t)row_slot * s->row_bytes,
               buf + (fo - j->cover_start), s->row_bytes);
    }
    uint32_t npages = (j->cover_bytes + PLE_PAGE - 1) >> PLE_PAGE_SHIFT;
    for (uint32_t p = 0; p < npages; p++) {
        uint64_t file_page = (j->cover_start >> PLE_PAGE_SHIFT) + p;
        if ((uint64_t)(p + 1) * PLE_PAGE <= (uint64_t)got)
            ple_page_publish(s, j->seg, file_page, buf + (size_t)p * PLE_PAGE);
    }

    pthread_mutex_lock(&s->bmtx);
    s->free_stack[s->free_top++] = bidx;
    pthread_cond_signal(&s->bcond);
    pthread_mutex_unlock(&s->bmtx);
    ple_job_finish(lk);
}

void *ple_worker_main(void *arg) {
    ple_store *s = arg;
    for (;;) {
        pthread_mutex_lock(&s->qmtx);
        while (!s->stop && !s->qhead) pthread_cond_wait(&s->qcond, &s->qmtx);
        if (s->stop) { pthread_mutex_unlock(&s->qmtx); return NULL; }
        ple_jobcell *cell = s->qhead;
        s->qhead = cell->next;
        if (!s->qhead) s->qtail = NULL;
        pthread_mutex_unlock(&s->qmtx);
        ple_job_run(s, &cell->job);
    }
    return NULL;
}

/* cudaLaunchHostFunc target (cudaHostFn_t: returns cudaError_t). Row errors
 * are not the launch's failure to report -- they land in lk->status for the
 * Python side to raise after the gather. Always return cudaSuccess so the
 * stream survives. */
int ple_lookup_run(void *arg) {
    ple_lookup *lk = arg;
    ple_store *s = lk->store;
    uint32_t rb = s->row_bytes, nkeys = 0;
    /* The count rides in pinned memory: a captured graph copies it with the
     * ids, so replays need no host-side bookkeeping. Cap guards a stale or
     * corrupt count. */
    uint32_t n = *lk->count > lk->capacity ? lk->capacity : *lk->count;
    atomic_store(&lk->status, 0);

    for (uint32_t i = 0; i < n; i++) {
        int64_t gid = lk->ids[i];
        if (gid < (int64_t)lk->vocab_start || gid >= (int64_t)lk->vocab_end) {
            memset(lk->rows + (size_t)i * rb, 0, rb);
            continue;
        }
        uint32_t seg = ple_seg_of(s, (uint64_t)gid);
        uint64_t fo = ple_row_off(s, seg, (uint64_t)gid);
        if (ple_row_from_cache(s, seg, fo, lk->rows + (size_t)i * rb) == 0) {
            atomic_fetch_add(&s->hits, 1);
            continue;
        }
        atomic_fetch_add(&s->misses, 1);
        uint64_t page = fo >> PLE_PAGE_SHIFT;
        lk->keys[nkeys++] = ((uint64_t)seg << PLE_KEY_SEG_SHIFT) |
                            (page << PLE_KEY_PAGE_SHIFT) | i;
    }
    if (nkeys == 0) return 0;
    qsort(lk->keys, nkeys, sizeof(uint64_t), ple_cmp_key);

    /* Group the sorted misses into per-segment reads of at most
     * PLE_JOB_PAGES starting pages each. */
    ple_jobcell *cells = lk->cells;
    uint32_t ncells = 0;
    uint32_t k = 0;
    while (k < nkeys) {
        uint32_t seg = (uint32_t)(lk->keys[k] >> PLE_KEY_SEG_SHIFT);
        uint64_t first_page = (lk->keys[k] >> PLE_KEY_PAGE_SHIFT) & 0xFFFFFFFFull;
        uint32_t j = k + 1;
        while (j < nkeys) {
            uint32_t s2 = (uint32_t)(lk->keys[j] >> PLE_KEY_SEG_SHIFT);
            uint64_t p2 = (lk->keys[j] >> PLE_KEY_PAGE_SHIFT) & 0xFFFFFFFFull;
            if (s2 != seg || p2 > first_page + PLE_JOB_PAGES - 1) break;
            j++;
        }
        uint64_t id_first = (uint64_t)lk->ids[lk->keys[k] & PLE_KEY_ROW_MASK];
        uint64_t fo_first = ple_row_off(s, seg, id_first);
        uint64_t last_page = (lk->keys[j - 1] >> PLE_KEY_PAGE_SHIFT) & 0xFFFFFFFFull;
        ple_jobcell *c = &cells[ncells++];
        c->job.lk = lk;
        c->job.seg = seg;
        c->job.cover_start = fo_first & ~(uint64_t)(PLE_PAGE - 1);
        /* Within one page the keys are sorted by row slot, not row id, so
         * the last starting page is the only safe bound: cover it plus the
         * page a row straddling into it may touch (+1 page total, inside
         * the PLE_JOB_PAGES + 1 buffer). */
        uint64_t cover_end = (last_page + 2) * (uint64_t)PLE_PAGE;
        c->job.cover_bytes = (uint32_t)(cover_end - c->job.cover_start);
        c->job.key_lo = k;
        c->job.key_hi = j;
        c->next = NULL;
        k = j;
    }

    atomic_store(&lk->pending, (int32_t)ncells);
    pthread_mutex_lock(&s->qmtx);
    for (uint32_t i = 0; i < ncells; i++) {
        if (s->qtail) s->qtail->next = &cells[i]; else s->qhead = &cells[i];
        s->qtail = &cells[i];
    }
    pthread_cond_broadcast(&s->qcond);
    pthread_mutex_unlock(&s->qmtx);
    /* Driver host-func threads must not block on a condvar (one services
     * all streams): drain by spinning. Paused first, then yielding, so a
     * cold gather whose reads outlive the spin costs context switches,
     * not an SMT starve. */
    for (int spins = 0; atomic_load(&lk->pending) > 0; spins++) {
        if (spins < 30000) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
        } else {
            sched_yield();
        }
    }
    return 0;
}
