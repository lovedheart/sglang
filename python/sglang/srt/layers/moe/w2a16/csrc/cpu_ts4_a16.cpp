// cpu_ts4_a16.cpp — W2A16 + WI8A MoE FFN, pure CPU (AVX-512 F/BW/DQ/VL + F16C,
// OpenMP). Two entry points sharing the ts4 decode helpers:
//   ts4_a16_ffn  — fp16 h1 (mirrors Vulkan -DA16)
//   ts4_i8a_ffn  — int8 h1 (mirrors Vulkan -DI8A / ig_moe.ts4_cpu_ref numpy ref)
//
// ts4/tsym4 format (vLLM-Moet w2 port, see ig_moe/_ts4.py):
//   w[n,k] = LUT[code] * 2^(scale-127) * gs ; LUT = {-4,-1,+1,+4}
//   plane: u8, 4 codes/byte along k (code k in bits 2*(k%4)..+1)
//   scales: u8 UE8M0, one per 32 k.
// A16 semantics (mirrors the Vulkan -DA16 path): input x is fp16 (stored as
// u16, exact to fp32), the h1 intermediate is rounded to fp16 before the w2
// GEMM; dots accumulate in fp32 with the pow2 block scales folded into the
// decoded weight vectors (exact: levels are +-1/+-4, scale is a power of two).
//
// ABI:
//   void ts4_a16_ffn(const void* x16, const int32_t* ids, const float* tw,
//                    const uint8_t* w13p, const uint8_t* s13, float gs13,
//                    const uint8_t* w2p,  const uint8_t* s2,  float gs2,
//                    int T, int K, int E, int H, int I, float* out32,
//                    int nthreads);
// ids [T*K], tw [T*K], x16 [T*H] IEEE half, out32 [T*H] fp32 (overwritten).
// H%32==0, I%32==0 (contract).
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <vector>

namespace {

const float LUT4[4] = {-4.0f, -1.0f, 1.0f, 4.0f};
const __m128i PAT = _mm_set_epi8(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
const __m512i SHP = _mm512_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6);
const __m512i M3 = _mm512_set1_epi32(3);

// 16 codes (one dword) -> 16 fp32 levels in a zmm.
inline __m512 decode16(uint32_t dword, __m512 lutbcast) {
    __m128i b4 = _mm_cvtsi32_si128((int)dword);
    __m128i exp4 = _mm_shuffle_epi8(b4, PAT);       // lane j -> byte j/4
    __m512i d = _mm512_cvtepu8_epi32(exp4);         // 16 lanes
    __m512i codes = _mm512_and_si512(_mm512_srlv_epi32(d, SHP), M3);
    return _mm512_permutexvar_ps(codes, lutbcast);  // index mod 4 picks LUT
}

} // namespace

extern "C" void ts4_a16_ffn(const void *x16v, const int32_t *ids, const float *tw,
                            const uint8_t *w13p, const uint8_t *s13, float gs13,
                            const uint8_t *w2p, const uint8_t *s2, float gs2,
                            int T, int K, int E, int H, int I, float *out32,
                            int nthreads) {
    (void)E;
    const uint16_t *x16 = (const uint16_t *)x16v;
    const int pairs = T * K;
    if (nthreads > 0) omp_set_num_threads(nthreads);
    float p2[256];
    for (int i = 0; i < 256; i++) p2[i] = exp2f((float)(i - 127));
    const __m512 lut = _mm512_broadcast_f32x4(_mm_loadu_ps(LUT4));

    const size_t W13ROW = (size_t)H / 4, W13EXP = (size_t)2 * I * W13ROW;
    const size_t S13ROW = (size_t)H / 32, S13EXP = (size_t)2 * I * S13ROW;
    const size_t W2ROW = (size_t)I / 4, W2EXP = (size_t)H * W2ROW;
    const size_t S2ROW = (size_t)I / 32, S2EXP = (size_t)H * S2ROW;

    // fp32 mirror of x (exact from fp16)
    std::vector<float> xf((size_t)T * H);
    for (int t = 0; t < T; t++) {
        const uint16_t *xr = x16 + (size_t)t * H;
        float *xd = xf.data() + (size_t)t * H;
        for (int h = 0; h < H; h += 16)
            _mm512_storeu_ps(xd + h, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(xr + h))));
    }

    // ---- Stage A: h16[pair][n] = round_fp16(silu(gate*gs13) * (up*gs13))
    std::vector<uint16_t> h16((size_t)pairs * I);
#pragma omp parallel for schedule(static)
    for (long pairn = 0; pairn < (long)pairs * I; pairn++) {
        int pair = (int)(pairn / I);
        int n = (int)(pairn % I);
        int token = pair / K;
        int e = ids[pair];
        const uint8_t *wg = w13p + (size_t)e * W13EXP + (size_t)n * W13ROW;
        const uint8_t *wu = wg + (size_t)I * W13ROW;
        const uint8_t *sg = s13 + (size_t)e * S13EXP + (size_t)n * S13ROW;
        const uint8_t *su = sg + (size_t)I * S13ROW;
        const float *a = xf.data() + (size_t)token * H;
        __m512 g0 = _mm512_setzero_ps(), g1 = _mm512_setzero_ps();
        __m512 u0 = _mm512_setzero_ps(), u1 = _mm512_setzero_ps();
        for (int k = 0; k < H; k += 32) {
            __m512 a0 = _mm512_loadu_ps(a + k);
            __m512 a1 = _mm512_loadu_ps(a + k + 16);
            __m512 cg = _mm512_set1_ps(p2[sg[k / 32]]);
            __m512 cu = _mm512_set1_ps(p2[su[k / 32]]);
            __m512 w0 = _mm512_mul_ps(decode16(*(const uint32_t *)(wg + k / 4), lut), cg);
            __m512 w1 = _mm512_mul_ps(decode16(*(const uint32_t *)(wg + k / 4 + 4), lut), cg);
            g0 = _mm512_fmadd_ps(a0, w0, g0);
            g1 = _mm512_fmadd_ps(a1, w1, g1);
            w0 = _mm512_mul_ps(decode16(*(const uint32_t *)(wu + k / 4), lut), cu);
            w1 = _mm512_mul_ps(decode16(*(const uint32_t *)(wu + k / 4 + 4), lut), cu);
            u0 = _mm512_fmadd_ps(a0, w0, u0);
            u1 = _mm512_fmadd_ps(a1, w1, u1);
        }
        float gx = (_mm512_reduce_add_ps(g0) + _mm512_reduce_add_ps(g1)) * gs13;
        float ux = (_mm512_reduce_add_ps(u0) + _mm512_reduce_add_ps(u1)) * gs13;
        float xc = gx < -20.f ? -20.f : (gx > 20.f ? 20.f : gx);
        float hv = (gx / (1.f + expf(-xc))) * ux;      // silu*up, GPU-identical
        _mm_storeu_si16((void *)(h16.data() + (size_t)pair * I + n),
                        _mm_cvtps_ph(_mm_set1_ps(hv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

    // fp32 mirror of the fp16-rounded h1 (unpacked once for the Stage-B loads)
    std::vector<float> h32((size_t)pairs * I);
#pragma omp parallel for schedule(static)
    for (int p = 0; p < pairs; p++) {
        const uint16_t *s = h16.data() + (size_t)p * I;
        float *d = h32.data() + (size_t)p * I;
        for (int i = 0; i < I; i += 16)
            _mm512_storeu_ps(d + i, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(s + i))));
    }

    // ---- Stage B: out[t][m] = gs2 * sum_k tw[pair] * (h32[pair] . w2[e,m])
    // 4-row tiling: each activation block is loaded once per 4 weight rows.
#pragma omp parallel for schedule(static)
    for (int m0 = 0; m0 < H; m0 += 4) {
        int nr = m0 + 3 < H ? 4 : H - m0;
        for (int t = 0; t < T; t++) {
            float tot[4] = {0.f, 0.f, 0.f, 0.f};
            for (int k = 0; k < K; k++) {
                int pair = t * K + k;
                int e = ids[pair];
                const uint8_t *wp = w2p + (size_t)e * W2EXP + (size_t)m0 * W2ROW;
                const uint8_t *sp = s2 + (size_t)e * S2EXP + (size_t)m0 * S2ROW;
                const float *a = h32.data() + (size_t)pair * I;
                __m512 acc[4][2];
                for (int r = 0; r < 4; r++) acc[r][0] = acc[r][1] = _mm512_setzero_ps();
                for (int kk = 0; kk < I; kk += 32) {
                    __m512 a0 = _mm512_loadu_ps(a + kk);
                    __m512 a1 = _mm512_loadu_ps(a + kk + 16);
                    for (int r = 0; r < nr; r++) {
                        __m512 c = _mm512_set1_ps(p2[sp[(size_t)r * S2ROW + kk / 32]]);
                        __m512 w0 = _mm512_mul_ps(decode16(*(const uint32_t *)(wp + (size_t)r * W2ROW + kk / 4), lut), c);
                        __m512 w1 = _mm512_mul_ps(decode16(*(const uint32_t *)(wp + (size_t)r * W2ROW + kk / 4 + 4), lut), c);
                        acc[r][0] = _mm512_fmadd_ps(a0, w0, acc[r][0]);
                        acc[r][1] = _mm512_fmadd_ps(a1, w1, acc[r][1]);
                    }
                }
                for (int r = 0; r < nr; r++)
                    tot[r] += (_mm512_reduce_add_ps(acc[r][0]) + _mm512_reduce_add_ps(acc[r][1])) * tw[pair];
            }
            for (int r = 0; r < nr; r++) out32[(size_t)t * H + m0 + r] = tot[r] * gs2;
        }
    }
}

// ---------------------------------------------------------------- i8a path
// h1 stays fp32 (like the Vulkan -DI8A Stage A output), then per 16-block:
//   eb = clamp(exponent(amax)-127-6, +-127)   (amax==0 -> -127), byte eb+127
//   q  = clamp(round_half_even(h * 2^-eb), +-127)
// Stage B dot: exact integer partial sums in fp32 (|q*w|<=508, 16-sum<=8128,
// all products share pow2 scale within a block), result scaled by
// 2^(eb + wsb - 254) and summed in fp32 over blocks — bit-compatible with
// ig_moe/ts4_cpu_ref.ts4_w2_gemm up to h1's fp32 accumulation order.
extern "C" void ts4_i8a_ffn(const void *x16v, const int32_t *ids, const float *tw,
                            const uint8_t *w13p, const uint8_t *s13, float gs13,
                            const uint8_t *w2p, const uint8_t *s2, float gs2,
                            int T, int K, int E, int H, int I, float *out32,
                            int nthreads) {
    (void)E;
    const uint16_t *x16 = (const uint16_t *)x16v;
    const int pairs = T * K;
    if (nthreads > 0) omp_set_num_threads(nthreads);
    float p2[256];
    for (int i = 0; i < 256; i++) p2[i] = exp2f((float)(i - 127));
    const __m512 lut = _mm512_broadcast_f32x4(_mm_loadu_ps(LUT4));

    const size_t W13ROW = (size_t)H / 4, W13EXP = (size_t)2 * I * W13ROW;
    const size_t S13ROW = (size_t)H / 32, S13EXP = (size_t)2 * I * S13ROW;
    const size_t W2ROW = (size_t)I / 4, W2EXP = (size_t)H * W2ROW;
    const size_t S2ROW = (size_t)I / 32, S2EXP = (size_t)H * S2ROW;

    std::vector<float> xf((size_t)T * H);
    for (int t = 0; t < T; t++) {
        const uint16_t *xr = x16 + (size_t)t * H;
        float *xd = xf.data() + (size_t)t * H;
        for (int h = 0; h < H; h += 16)
            _mm512_storeu_ps(xd + h, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(xr + h))));
    }

    // ---- Stage A: h32[pair][n] = silu(gate*gs13) * (up*gs13), fp32 (NO rounding)
    std::vector<float> h32((size_t)pairs * I);
#pragma omp parallel for schedule(static)
    for (long pairn = 0; pairn < (long)pairs * I; pairn++) {
        int pair = (int)(pairn / I);
        int n = (int)(pairn % I);
        int token = pair / K;
        int e = ids[pair];
        const uint8_t *wg = w13p + (size_t)e * W13EXP + (size_t)n * W13ROW;
        const uint8_t *wu = wg + (size_t)I * W13ROW;
        const uint8_t *sg = s13 + (size_t)e * S13EXP + (size_t)n * S13ROW;
        const uint8_t *su = sg + (size_t)I * S13ROW;
        const float *a = xf.data() + (size_t)token * H;
        __m512 g0 = _mm512_setzero_ps(), g1 = _mm512_setzero_ps();
        __m512 u0 = _mm512_setzero_ps(), u1 = _mm512_setzero_ps();
        for (int k = 0; k < H; k += 32) {
            __m512 a0 = _mm512_loadu_ps(a + k);
            __m512 a1 = _mm512_loadu_ps(a + k + 16);
            __m512 cg = _mm512_set1_ps(p2[sg[k / 32]]);
            __m512 cu = _mm512_set1_ps(p2[su[k / 32]]);
            g0 = _mm512_fmadd_ps(a0, _mm512_mul_ps(decode16(*(const uint32_t *)(wg + k / 4), lut), cg), g0);
            g1 = _mm512_fmadd_ps(a1, _mm512_mul_ps(decode16(*(const uint32_t *)(wg + k / 4 + 4), lut), cg), g1);
            u0 = _mm512_fmadd_ps(a0, _mm512_mul_ps(decode16(*(const uint32_t *)(wu + k / 4), lut), cu), u0);
            u1 = _mm512_fmadd_ps(a1, _mm512_mul_ps(decode16(*(const uint32_t *)(wu + k / 4 + 4), lut), cu), u1);
        }
        float gx = (_mm512_reduce_add_ps(g0) + _mm512_reduce_add_ps(g1)) * gs13;
        float ux = (_mm512_reduce_add_ps(u0) + _mm512_reduce_add_ps(u1)) * gs13;
        float xc = gx < -20.f ? -20.f : (gx > 20.f ? 20.f : gx);
        h32[(size_t)pair * I + n] = (gx / (1.f + expf(-xc))) * ux;
    }

    // ---- Quantize: q32 (fp32 integers) + e8 bytes per 16-block
    const int NB = I / 16;
    std::vector<float> q32((size_t)pairs * I);
    std::vector<uint8_t> e8((size_t)pairs * NB);
#pragma omp parallel for schedule(static)
    for (long pb = 0; pb < (long)pairs * NB; pb++) {
        int pair = (int)(pb / NB), b = (int)(pb % NB);
        __m512 v = _mm512_loadu_ps(h32.data() + (size_t)pair * I + (size_t)b * 16);
        float amax = _mm512_reduce_max_ps(_mm512_andnot_ps(_mm512_set1_ps(-0.f), v));
        int ee;
        if (amax == 0.f) {
            ee = -127;
        } else {
            uint32_t bits;
            memcpy(&bits, &amax, 4);
            ee = (int)((bits >> 23) & 0xFF) - 127 - 6;
            if (ee < -127) ee = -127;
            if (ee > 127) ee = 127;
        }
        __m512i qi = _mm512_cvtps_epi32(_mm512_mul_ps(v, _mm512_set1_ps(exp2f((float)-ee))));
        qi = _mm512_max_epi32(_mm512_min_epi32(qi, _mm512_set1_epi32(127)), _mm512_set1_epi32(-127));
        _mm512_storeu_ps(q32.data() + (size_t)pair * I + (size_t)b * 16, _mm512_cvtepi32_ps(qi));
        e8[(size_t)pair * NB + b] = (uint8_t)(ee + 127);
    }

    // ---- Stage B: exact block dots; 4-row tiling; per-block 2^eb scaling
#pragma omp parallel for schedule(static)
    for (int m0 = 0; m0 < H; m0 += 4) {
        int nr = m0 + 3 < H ? 4 : H - m0;
        for (int t = 0; t < T; t++) {
            float tot[4] = {0.f, 0.f, 0.f, 0.f};
            for (int k = 0; k < K; k++) {
                int pair = t * K + k;
                int e = ids[pair];
                const uint8_t *wp = w2p + (size_t)e * W2EXP + (size_t)m0 * W2ROW;
                const uint8_t *sp = s2 + (size_t)e * S2EXP + (size_t)m0 * S2ROW;
                const float *a = q32.data() + (size_t)pair * I;
                const uint8_t *eb = e8.data() + (size_t)pair * NB;
                for (int kk = 0; kk < I; kk += 32) {
                    __m512 q0 = _mm512_loadu_ps(a + kk);
                    __m512 q1 = _mm512_loadu_ps(a + kk + 16);
                    // fold the pair's token weight into the pair-specific pow2
                    // activation scale (pow2 * tw is exact)
                    float e0 = p2[eb[kk / 16]] * tw[pair], e1 = p2[eb[kk / 16 + 1]] * tw[pair];
                    bool same = eb[kk / 16] == eb[kk / 16 + 1];
                    for (int r = 0; r < nr; r++) {
                        float cw = p2[sp[(size_t)r * S2ROW + kk / 32]];  // per-row weight scale
                        __m512 w0 = _mm512_mul_ps(decode16(*(const uint32_t *)(wp + (size_t)r * W2ROW + kk / 4), lut),
                                                  _mm512_set1_ps(cw));
                        __m512 w1 = _mm512_mul_ps(decode16(*(const uint32_t *)(wp + (size_t)r * W2ROW + kk / 4 + 4), lut),
                                                  _mm512_set1_ps(cw));
                        if (same) {
                            tot[r] += _mm512_reduce_add_ps(_mm512_fmadd_ps(q1, w1, _mm512_mul_ps(q0, w0))) * e0;
                        } else {
                            float d0 = _mm512_reduce_add_ps(_mm512_mul_ps(q0, w0));
                            float d1 = _mm512_reduce_add_ps(_mm512_mul_ps(q1, w1));
                            tot[r] += d0 * e0 + d1 * e1;
                        }
                    }
                }
            }
            for (int r = 0; r < nr; r++) out32[(size_t)t * H + m0 + r] = tot[r] * gs2;
        }
    }
}
