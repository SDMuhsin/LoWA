/* LoWA-TA L1 hwmodel core: the chip's arithmetic, bit for bit.
 *
 * Arithmetic contract (docs/arith.md): IEEE binary32, round-to-nearest-even, subnormal inputs
 * read as +-0 (DAZ) and subnormal results written as +-0 (FTZ, tininess after rounding), fused
 * multiply-add with ONE rounding.  Implemented with the x86 SSE/AVX-512 units under
 * MXCSR = RNE | FTZ | DAZ, set on entry to every exported function and every OpenMP region and
 * restored on exit.  test_arith.py proves add/mul/fma against an exact integer reference.
 * Every reduction runs in the dataflow.md order; parallelism is only ever across independent
 * outputs.  Build: see Makefile in this directory (-O2 -mavx512f -mfma -ffp-contract=off). */
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef float f32;

/* ------------------------------------------------------------------ MXCSR, bits */
static inline unsigned csr_on(void) {
    unsigned old = _mm_getcsr();
    _mm_setcsr((old & ~0x6000u) | 0x8040u);           /* RC = RNE, FTZ (bit 15), DAZ (bit 6) */
    return old;
}
static inline void csr_off(unsigned old) { _mm_setcsr(old); }
#define PAR_FOR(sched, ...)                                                                   \
    _Pragma("omp parallel") {                                                                 \
        unsigned _cs = csr_on();                                                              \
        _Pragma(sched) __VA_ARGS__                                                            \
        csr_off(_cs);                                                                         \
    }

static inline uint32_t B(f32 x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static inline f32 F(uint32_t u) { f32 x; memcpy(&x, &u, 4); return x; }

/* --------------------------------------------------------- the three rounded primitives */
/* Every rounded primitive returns the canonical quiet NaN 0x7FC00000 when its result is a NaN
 * (docs/arith.md section 1).  x86 generates the "real indefinite" 0xFFC00000 for 0 * inf, inf - inf
 * and friends; P4 moved the canonicalisation from the exported element-wise entry points into the
 * primitives so that the composite ops (softmax of a fully masked row: P = mul(0, recip(+0)) = NaN)
 * meet the contract too.  Non-NaN results are unchanged. */
static inline f32 canon(f32 y) { return y != y ? F(0x7FC00000u) : y; }
static inline f32 fadd(f32 a, f32 b) { return canon(_mm_cvtss_f32(_mm_add_ss(_mm_set_ss(a), _mm_set_ss(b)))); }
static inline f32 fsub(f32 a, f32 b) { return canon(_mm_cvtss_f32(_mm_sub_ss(_mm_set_ss(a), _mm_set_ss(b)))); }
static inline f32 fmul(f32 a, f32 b) { return canon(_mm_cvtss_f32(_mm_mul_ss(_mm_set_ss(a), _mm_set_ss(b)))); }
static inline f32 ffma(f32 a, f32 b, f32 c) {
    return canon(_mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c))));
}
static inline f32 fneg(f32 a) { return F(B(a) ^ 0x80000000u); }
static inline f32 fmax_(f32 a, f32 b) { return a > b ? a : b; }   /* inputs never NaN here */
static inline f32 pow2(int k) { return F((uint32_t)(k + 127) << 23); }  /* -126 <= k <= 127 */
static inline f32 daz(f32 x) { return (B(x) & 0x7f800000u) ? x : F(B(x) & 0x80000000u); }

#define QNAN F(0x7FC00000u)
#define PINF F(0x7F800000u)

/* ------------------------------------------------------------------ constants (hex) */
#define C_L2E    F(0x3FB8AA3Bu)   /* log2(e)                 */
#define C_MAGIC  F(0x4B400000u)   /* 1.5 * 2^23              */
#define C_LN2HI  F(0x3F317218u)   /* fp32(ln 2)              */
#define C_LN2LO  F(0xB102E308u)   /* ln 2 - fp32(ln 2)       */
#define C_R0     F(0x3FB4B4B5u)   /* 24/17                   */
#define C_R1     F(0x3EF0F0F1u)   /* 8/17                    */
#define C_GELU_A F(0x3F3504F3u)   /* fp32(1/sqrt 2)          */
#define C_GELU_B F(0x3ECC422Au)   /* fp32(1/sqrt(2 pi))      */
static const uint32_t EXP_C[7] = {0x3F800000, 0x3F000000, 0x3E2AAAAA, 0x3D2AAA41,
                                  0x3C088986, 0x3AB6CAB2, 0x394F4769};
static const uint32_t TANH_C[9] = {0xBEAAAAAB, 0x3E08887D, 0xBD5D0B49, 0x3CB30437, 0xBC1019E8,
                                   0x3B604FFA, 0xBA9B451C, 0x39A05C7B, 0xB82CB570};
static const uint32_t ERFS_C[7] = {0x3E0375D4, 0xBEC093A2, 0x3DE71710, 0xBCDC083E,
                                   0x3BAA6D6D, 0xBA5574D0, 0x38AF88C7};
static const uint32_t ERFL_C[10] = {0x3E037DA2, 0x3F22ECFB, 0x3DD393B0, 0xBCA3B3D8, 0x3A233D73,
                                    0x3AA0702B, 0xBA021F57, 0x38D82E17, 0xB746876D, 0x351F8958};
static const uint32_t LOG_C[9] = {0xBEFFFFFC, 0x3EAAAAA8, 0xBE800213, 0x3E4CD1FA, 0xBE2A13B6,
                                  0x3E1138C6, 0xBE079BB2, 0x3E04245D, 0xBD9C187A};

static inline f32 horner(const uint32_t *c, int deg, f32 x) {
    f32 p = F(c[deg]);
    for (int k = deg - 1; k >= 0; k--) p = ffma(p, x, F(c[k]));
    return p;
}

/* ------------------------------------------------------------------ SFU algorithms */
/* exp: n = rint(x log2e) by the 1.5*2^23 trick, Cody-Waite r = x - n ln2 (two FMAs),
 * e^r = fma(G(r), r, 1) with G degree 6, result = (p * 2^n1) * 2^n2, n1 = n>>1, n2 = n-n1. */
static inline f32 sfu_exp(f32 x) {
    x = daz(x);
    if (x != x) return QNAN;
    if (x > 89.0f) return PINF;
    if (x < -104.0f) return 0.0f;
    f32 t = ffma(x, C_L2E, C_MAGIC);
    int n = (int)(B(t) - B(C_MAGIC));
    f32 nf = fsub(t, C_MAGIC);
    f32 r = ffma(nf, fneg(C_LN2HI), x);
    r = ffma(nf, fneg(C_LN2LO), r);
    f32 p = ffma(horner(EXP_C, 6, r), r, 1.0f);
    int n1 = n >> 1, n2 = n - n1;
    return fmul(fmul(p, pow2(n1)), pow2(n2));
}

/* recip: m in [1,2) from the significand, y0 = 24/17 - 8/17 m, 3 Newton steps
 * e = fma(-m, y, 1), y = fma(y, e, y), then scaled by 2^-E in two exact muls (FTZ on underflow). */
static inline f32 sfu_recip(f32 x) {
    uint32_t u = B(x), s = u & 0x80000000u, a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return QNAN;
    if (a == 0x7f800000u) return F(s);
    if (a < 0x00800000u) return F(s | 0x7f800000u);
    int E = (int)(a >> 23) - 127;
    f32 m = F((a & 0x7fffffu) | 0x3f800000u);
    f32 y = ffma(fneg(m), C_R1, C_R0);
    for (int i = 0; i < 3; i++) { f32 e = ffma(fneg(m), y, 1.0f); y = ffma(y, e, y); }
    int k = -E, k1 = k >> 1, k2 = k - k1;
    y = fmul(fmul(y, pow2(k1)), pow2(k2));
    return F(B(y) | s);
}

/* rsqrt core on m in [1,4): y0 = 0x5F3759DF - (bits(m) >> 1); 3 Newton steps
 * t = y*y, e = fma(-m, t, 1), y = fma(y*0.5, e, y). */
static inline f32 rsqrt_m(f32 m) {
    f32 y = F(0x5F3759DFu - (B(m) >> 1));
    for (int i = 0; i < 3; i++) {
        f32 t = fmul(y, y);
        f32 e = ffma(fneg(m), t, 1.0f);
        y = ffma(fmul(y, 0.5f), e, y);
    }
    return y;
}
/* x = m 2^(2k), m in [1,4) (exponent split, integer ops).  rsqrt = rsqrt_m(m) * 2^-k.
 * sqrt: y = rsqrt_m(m), s = m*y, r = fma(-s, s, m), s = fma(r, y*0.5, s), result s * 2^k. */
static inline int split_even(uint32_t a, f32 *m) {
    int E = (int)(a >> 23) - 127, k = E >> 1;
    *m = F((a & 0x7fffffu) | ((uint32_t)(127 + E - 2 * k) << 23));
    return k;
}
static inline f32 sfu_rsqrt(f32 x) {
    uint32_t u = B(x), a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return QNAN;
    if (a < 0x00800000u) return F((u & 0x80000000u) | 0x7f800000u);
    if (u & 0x80000000u) return QNAN;
    if (a == 0x7f800000u) return 0.0f;
    f32 m;
    int k = split_even(a, &m);
    return fmul(rsqrt_m(m), pow2(-k));
}
static inline f32 sfu_sqrt(f32 x) {
    uint32_t u = B(x), a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return QNAN;
    if (a < 0x00800000u) return F(u & 0x80000000u);
    if (u & 0x80000000u) return QNAN;
    if (a == 0x7f800000u) return PINF;
    f32 m;
    int k = split_even(a, &m);
    f32 y = rsqrt_m(m);
    f32 s = fmul(m, y);
    f32 r = ffma(fneg(s), s, m);
    return fmul(ffma(r, fmul(y, 0.5f), s), pow2(k));
}

/* tanh: |x| >= 9 -> +-1; |x| < 1: fma(x*z, P(z), x), z = x*x, P degree 8;
 * else 1 - 2 / (exp(2|x|) + 1) = fma(-2, recip(exp(2a) + 1), 1), sign restored. */
static inline f32 sfu_tanh(f32 x) {
    x = daz(x);
    if (x != x) return QNAN;
    uint32_t s = B(x) & 0x80000000u;
    f32 a = F(B(x) & 0x7fffffffu);
    if (a >= 9.0f) return F(s | 0x3f800000u);
    if (a < 1.0f) {
        f32 z = fmul(x, x);
        return F(B(ffma(fmul(x, z), horner(TANH_C, 8, z), x)) | s);
    }
    f32 e = sfu_exp(fmul(a, 2.0f));
    f32 r = ffma(-2.0f, sfu_recip(fadd(e, 1.0f)), 1.0f);
    return F(B(r) | s);
}

/* erf: |x| >= 3.92 -> +-1; |x| < 0.921875: fma(x, Q(z), x), Q degree 6 in z = x*x;
 * else 1 - exp(-(a + a R(a))), R degree 9 in a, sign restored. */
static inline f32 sfu_erf(f32 x) {
    x = daz(x);
    if (x != x) return QNAN;
    uint32_t s = B(x) & 0x80000000u;
    f32 a = F(B(x) & 0x7fffffffu);
    if (a >= 3.92f) return F(s | 0x3f800000u);
    if (a < 0.921875f) return ffma(x, horner(ERFS_C, 6, fmul(x, x)), x);
    f32 t = ffma(horner(ERFL_C, 9, a), a, a);
    f32 r = fsub(1.0f, sfu_exp(fneg(t)));
    return F(B(r) | s);
}

/* ln (loss value only): m in (sqrt.5, sqrt2], f = m - 1 (exact), ln(1+f) = fma(f*f, Q(f), f),
 * result = fma(E, ln2_hi, fma(E, ln2_lo, ln(1+f))). */
static inline f32 sfu_log(f32 x) {
    uint32_t u = B(x), a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return QNAN;
    if (a < 0x00800000u) return F(0xff800000u);
    if (u & 0x80000000u) return QNAN;
    if (a == 0x7f800000u) return PINF;
    int E = (int)(a >> 23) - 127;
    uint32_t mb = (a & 0x7fffffu) | 0x3f800000u;
    if (mb > 0x3FB504F3u) { mb -= 0x00800000u; E += 1; }
    f32 f = fsub(F(mb), 1.0f);
    f32 r = ffma(fmul(f, f), horner(LOG_C, 8, f), f);
    f32 Ef = (f32)E;
    r = ffma(Ef, C_LN2LO, r);
    return ffma(Ef, C_LN2HI, r);
}

static inline f32 gelu(f32 u) {
    return fmul(fmul(u, 0.5f), fadd(1.0f, sfu_erf(fmul(u, C_GELU_A))));
}
static inline f32 gelu_grad(f32 u) {  /* cdf + u * pdf (torch GeluCUDAKernelImpl backward) */
    f32 cdf = fmul(0.5f, fadd(1.0f, sfu_erf(fmul(u, C_GELU_A))));
    f32 pdf = fmul(sfu_exp(fmul(fmul(u, -0.5f), u)), C_GELU_B);
    return ffma(u, pdf, cdf);
}

/* ======================================================= exported element-wise entry points */
#define ELEM1(name, expr)                                                                   \
    void name(long n, const f32 *x, f32 *y) {                                               \
        PAR_FOR("omp for schedule(static)", for (long i = 0; i < n; i++) {                  \
            f32 a = x[i]; y[i] = canon(expr); })                                            \
    }
#define ELEM2(name, expr)                                                                   \
    void name(long n, const f32 *x, const f32 *z, f32 *y) {                                 \
        PAR_FOR("omp for schedule(static)", for (long i = 0; i < n; i++) {                  \
            f32 a = x[i], b = z[i]; y[i] = canon(expr); })                                  \
    }
ELEM2(lw_add, fadd(a, b))
ELEM2(lw_sub, fsub(a, b))
ELEM2(lw_mul, fmul(a, b))
ELEM2(lw_max, (daz(a) > daz(b) ? daz(a) : daz(b)))
ELEM2(lw_min, (daz(a) > daz(b) ? daz(b) : daz(a)))         /* P2: min(c, 1) = clip's c > 1 ? 1 : c */
ELEM1(lw_exp, sfu_exp(a))
ELEM1(lw_recip, sfu_recip(a))
ELEM1(lw_rsqrt, sfu_rsqrt(a))
ELEM1(lw_sqrt, sfu_sqrt(a))
ELEM1(lw_tanh, sfu_tanh(a))
ELEM1(lw_erf, sfu_erf(a))
ELEM1(lw_log, sfu_log(a))
ELEM1(lw_gelu, gelu(a))
ELEM2(lw_gelu_bwd, fmul(a, gelu_grad(b)))                  /* a = dg, b = u */
ELEM2(lw_tanh_bwd, fmul(a, ffma(fneg(b), b, 1.0f)))        /* a = d(tanh out), b = tanh out */

/* P2 golden entry points (no new numerics): IEEE compares on DAZ'd operands, bit0 a<b, bit1 a==b,
 * bit2 a>b, bit3 unordered; and signed int32 -> fp32 (cvtsi2ss under MXCSR RNE). */
void lw_cmp(long n, const f32 *x, const f32 *z, uint8_t *y) {
    PAR_FOR("omp for schedule(static)", for (long i = 0; i < n; i++) {
        f32 a = daz(x[i]), b = daz(z[i]);
        y[i] = (uint8_t)((a < b) | ((a == b) << 1) | ((a > b) << 2) | ((a != a || b != b) << 3)); })
}
void lw_i2f(long n, const int32_t *x, f32 *y) {
    PAR_FOR("omp for schedule(static)", for (long i = 0; i < n; i++) {
        y[i] = _mm_cvtss_f32(_mm_cvtsi32_ss(_mm_setzero_ps(), x[i])); })
}
void lw_fma(long n, const f32 *a, const f32 *b, const f32 *c, f32 *y) {
    PAR_FOR("omp for schedule(static)",
            for (long i = 0; i < n; i++) y[i] = canon(ffma(a[i], b[i], c[i]));)
}
/* y = x*s (s a power of two or any fp32 constant) */
void lw_scale(long n, const f32 *x, f32 s, f32 *y) {
    PAR_FOR("omp for schedule(static)", for (long i = 0; i < n; i++) y[i] = fmul(x[i], s);)
}
/* dropout fwd / bwd (same op): y = keep ? x * DROP_SCALE : +0 */
void lw_drop(long n, const f32 *x, const uint8_t *keep, f32 ds, f32 *y) {
    PAR_FOR("omp for schedule(static)",
            for (long i = 0; i < n; i++) y[i] = keep[i] ? fmul(x[i], ds) : 0.0f;)
}
/* y[r, c] = add(x[r, c], bias[c]) */
void lw_add_bias(long rows, long cols, const f32 *x, const f32 *bias, f32 *y) {
    PAR_FOR("omp for schedule(static)", for (long r = 0; r < rows; r++) {
        for (long c = 0; c < cols; c++) y[r * cols + c] = fadd(x[r * cols + c], bias[c]); })
}
/* out[c] = sum over r ascending of x[r, c], from +0 (bias gradients) */
void lw_colsum(long rows, long cols, const f32 *x, f32 *out) {
    PAR_FOR("omp for schedule(static)", for (long c = 0; c < cols; c++) {
        f32 s = 0.0f;
        for (long r = 0; r < rows; r++) s = fadd(s, x[r * cols + c]);
        out[c] = s; })
}

/* ================================================================== GEMM (dataflow.md 2)
 * C(i,j) = sum_k A(i,k) B(j,k), K split into chunks of KC = 64 (the last may be short):
 *   p = +0; p = fma(A(i,k), B(j,k), p) for k ascending within the chunk;   (PE accumulator)
 *   acc = add(acc, p) for chunks ascending, acc starting at +0              (ACC drain adder)
 * Element addresses: A + b1*sA1 + b2*sA2 + i*a_r + k*a_k, B + b1*sB1 + b2*sB2 + j*b_n + k*b_k,
 * C + b1*sC1 + b2*sC2 + i*c_r + j*c_c.  Vectorised ACROSS j (16 lanes x 2) and i (8 rows):
 * every lane is its own sequential FMA chain, so the per-element order is the contract's. */
#define NR 32
#define MR 8
#define MCH 256
static int GEMM_KC = 64;               /* chunk length of the contract; 0 = one chain over all K */
void lw_set_kc(int kc) { GEMM_KC = kc; }
void lw_gemm(int nb1, int nb2, int M, int N, int K,
             const f32 *A, long sA1, long sA2, long a_r, long a_k,
             const f32 *Bm, long sB1, long sB2, long b_n, long b_k,
             f32 *C, long sC1, long sC2, long c_r, long c_c) {
    int np = (N + NR - 1) / NR, nm = (M + MCH - 1) / MCH;
    int kc = GEMM_KC > 0 ? GEMM_KC : (K > 0 ? K : 1);
    long ntask = (long)nb1 * nb2 * np * nm;
    _Pragma("omp parallel") {
        unsigned cs = csr_on();
        f32 *Bp = aligned_alloc(64, (size_t)(K > 0 ? K : 1) * NR * sizeof(f32));
        _Pragma("omp for schedule(dynamic)")
        for (long t = 0; t < ntask; t++) {
            long r = t;
            int mc = r % nm; r /= nm;
            int jp = r % np; r /= np;
            int b2 = r % nb2, b1 = (int)(r / nb2);
            const f32 *Ab = A + b1 * sA1 + b2 * sA2;
            const f32 *Bb = Bm + b1 * sB1 + b2 * sB2;
            f32 *Cb = C + b1 * sC1 + b2 * sC2;
            int j0 = jp * NR, nj = N - j0 < NR ? N - j0 : NR;
            for (int jj = 0; jj < NR; jj++) {
                if (jj < nj) {
                    const f32 *src = Bb + (long)(j0 + jj) * b_n;
                    for (int k = 0; k < K; k++) Bp[(long)k * NR + jj] = src[(long)k * b_k];
                } else {
                    for (int k = 0; k < K; k++) Bp[(long)k * NR + jj] = 0.0f;
                }
            }
            int i_end = (mc + 1) * MCH < M ? (mc + 1) * MCH : M;
            for (int i0 = mc * MCH; i0 < i_end; i0 += MR) {
                int ni = i_end - i0 < MR ? i_end - i0 : MR;
                const f32 *ap[MR];
                for (int q = 0; q < MR; q++) ap[q] = Ab + (long)(i0 + (q < ni ? q : 0)) * a_r;
                __m512 tot[MR][2];
                for (int q = 0; q < MR; q++) tot[q][0] = tot[q][1] = _mm512_setzero_ps();
                for (int k0 = 0; k0 < K || k0 == 0; k0 += kc) {
                    int k1 = k0 + kc < K ? k0 + kc : K;
                    __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
                    __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
                    __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
                    __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
                    __m512 c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
                    __m512 c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();
                    __m512 c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps();
                    __m512 c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps();
                    const f32 *bp = Bp + (long)k0 * NR;
                    long ko = (long)k0 * a_k;
                    for (int k = k0; k < k1; k++, bp += NR, ko += a_k) {
                        __m512 b0 = _mm512_load_ps(bp), b1v = _mm512_load_ps(bp + 16);
                        __m512 a;
#define ROW(q, x0, x1) a = _mm512_set1_ps(ap[q][ko]); x0 = _mm512_fmadd_ps(a, b0, x0); x1 = _mm512_fmadd_ps(a, b1v, x1);
                        ROW(0, c00, c01) ROW(1, c10, c11) ROW(2, c20, c21) ROW(3, c30, c31)
                        ROW(4, c40, c41) ROW(5, c50, c51) ROW(6, c60, c61) ROW(7, c70, c71)
#undef ROW
                    }
                    __m512 cc[MR][2] = {{c00, c01}, {c10, c11}, {c20, c21}, {c30, c31},
                                        {c40, c41}, {c50, c51}, {c60, c61}, {c70, c71}};
                    /* drain: ACC = add(ACC, chunk partial); ACC starts at +0 */
                    for (int q = 0; q < MR; q++) {
                        tot[q][0] = _mm512_add_ps(tot[q][0], cc[q][0]);
                        tot[q][1] = _mm512_add_ps(tot[q][1], cc[q][1]);
                    }
                    if (K == 0) break;
                }
                for (int q = 0; q < ni; q++) {
                    f32 *cp = Cb + (long)(i0 + q) * c_r + (long)j0 * c_c;
                    f32 tmp[NR] __attribute__((aligned(64)));
                    _mm512_store_ps(tmp, tot[q][0]);
                    _mm512_store_ps(tmp + 16, tot[q][1]);
                    for (int jj = 0; jj < nj; jj++) cp[(long)jj * c_c] = tmp[jj];
                }
            }
        }
        free(Bp);
        csr_off(cs);
    }
}

/* ===================================================================== LayerNorm (dataflow 5) */
/* Chunked reductions (P1): each run of KC elements is summed from +0 in order, and the chunk
 * partials are added in order into a total that starts at +0.  KC = GEMM_KC (64; 0 = one chain). */
#define CSUM_BEGIN f32 _tot = 0.0f, _p = 0.0f; int _c = 0;
#define CSUM_STEP(update) { update; if (++_c == kc) { _tot = fadd(_tot, _p); _p = 0.0f; _c = 0; } }
#define CSUM_END(out) { if (_c) _tot = fadd(_tot, _p); out = _tot; }
/* rows x d; stats out: mu, rstd per row */
void lw_ln_fwd(long rows, int d, const f32 *x, const f32 *g, const f32 *b, f32 eps, f32 inv_d,
               f32 *y, f32 *mu_out, f32 *rstd_out) {
    int kc = GEMM_KC > 0 ? GEMM_KC : d;
    PAR_FOR("omp for schedule(static)", for (long r = 0; r < rows; r++) {
        const f32 *xr = x + r * d;
        f32 s, q;
        { CSUM_BEGIN for (int i = 0; i < d; i++) CSUM_STEP(_p = fadd(_p, xr[i])) CSUM_END(s) }
        f32 mu = fmul(s, inv_d);
        { CSUM_BEGIN for (int i = 0; i < d; i++) { f32 t = fsub(xr[i], mu); CSUM_STEP(_p = ffma(t, t, _p)) } CSUM_END(q) }
        f32 var = fmul(q, inv_d);
        f32 rstd = sfu_rsqrt(fadd(var, eps));
        for (int i = 0; i < d; i++) y[r * d + i] = ffma(fmul(fsub(xr[i], mu), rstd), g[i], b[i]);
        mu_out[r] = mu; rstd_out[r] = rstd; })
}
/* dx from z (pre-LN input), stats, gamma, dy; xhat recomputed exactly as in forward */
void lw_ln_bwd(long rows, int d, const f32 *z, const f32 *mu, const f32 *rstd, const f32 *g,
               const f32 *dy, f32 inv_d, f32 *dx) {
    int kc = GEMM_KC > 0 ? GEMM_KC : d;
    PAR_FOR("omp for schedule(static)", for (long r = 0; r < rows; r++) {
        const f32 *zr = z + r * d, *dr = dy + r * d;
        f32 m = mu[r], rs = rstd[r], a, bb;
        { CSUM_BEGIN for (int i = 0; i < d; i++) CSUM_STEP(_p = fadd(_p, fmul(dr[i], g[i]))) CSUM_END(a) }
        { CSUM_BEGIN for (int i = 0; i < d; i++) {
            f32 gy = fmul(dr[i], g[i]);
            f32 xh = fmul(fsub(zr[i], m), rs);
            CSUM_STEP(_p = ffma(gy, xh, _p)) } CSUM_END(bb) }
        f32 ad = fmul(a, inv_d), bd = fmul(bb, inv_d);
        for (int i = 0; i < d; i++) {
            f32 gy = fmul(dr[i], g[i]);
            f32 xh = fmul(fsub(zr[i], m), rs);
            dx[r * d + i] = fmul(rs, fsub(fsub(gy, ad), fmul(xh, bd)));
        } })
}

/* ======================================================================= softmax (dataflow 5)
 * S: [nrow_groups][L][L] with nrow_groups = B*H, kmask[b][j] (1 = real key), keep (NULL = no
 * dropout) same shape as S.  Writes P (pre-dropout), Pd (= P'), stats m, r per row. */
void lw_smax_fwd(int Bn, int H, int L, const f32 *S, const uint8_t *kmask, const uint8_t *keep,
                 f32 ds, f32 *P, f32 *Pd, f32 *mrow, f32 *rrow) {
    long rows = (long)Bn * H * L;
    PAR_FOR("omp for schedule(static)", for (long r = 0; r < rows; r++) {
        int b = (int)(r / ((long)H * L));
        const f32 *s = S + r * L;
        const uint8_t *km = kmask + (long)b * L;
        f32 m = F(0xff800000u);
        for (int j = 0; j < L; j++) if (km[j]) m = fmax_(m, s[j]);
        f32 sum = 0.0f;
        for (int j = 0; j < L; j++) {
            f32 e = km[j] ? sfu_exp(fsub(s[j], m)) : 0.0f;
            P[r * L + j] = e;
            sum = fadd(sum, e);
        }
        f32 rc = sfu_recip(sum);
        for (int j = 0; j < L; j++) {
            f32 p = fmul(P[r * L + j], rc);
            P[r * L + j] = p;
            Pd[r * L + j] = (!keep || keep[r * L + j]) ? (keep ? fmul(p, ds) : p) : 0.0f;
        }
        mrow[r] = m; rrow[r] = rc; })
}
/* dS = (P * (dP - rd)) * scale, dP = drop-bwd(dP'), rd = sum_j fma(P, dP, .) */
void lw_smax_bwd(long rows, int L, const f32 *P, const f32 *dPd, const uint8_t *keep, f32 ds,
                 f32 scale, f32 *dS) {
    PAR_FOR("omp for schedule(static)", for (long r = 0; r < rows; r++) {
        const f32 *p = P + r * L, *g = dPd + r * L;
        f32 rd = 0.0f;
        for (int j = 0; j < L; j++) {
            f32 dp = (!keep || keep[r * L + j]) ? (keep ? fmul(g[j], ds) : g[j]) : 0.0f;
            rd = ffma(p[j], dp, rd);
        }
        for (int j = 0; j < L; j++) {
            f32 dp = (!keep || keep[r * L + j]) ? (keep ? fmul(g[j], ds) : g[j]) : 0.0f;
            dS[r * L + j] = fmul(fmul(p[j], fsub(dp, rd)), scale);
        } })
}

/* ===================================================================== LoWA (dataflow 3, 4) */
/* merge-on-load: W[eflat[e]] = add(W[eflat[e]], mul(mul(gamma, c[e / s]), h[e % s])), e ascending */
void lw_lowa_merge(long ne, int s, const int64_t *eflat, const f32 *coef, const f32 *h, f32 gamma,
                   f32 *W) {
    unsigned cs = csr_on();
    for (long e = 0; e < ne; e++) {
        f32 v = fmul(gamma, coef[e / s]);
        W[eflat[e]] = fadd(W[eflat[e]], fmul(v, h[e % s]));
    }
    csr_off(cs);
}
/* sampled coefficient gradient, atom-major (D5):
 * for t ascending, for tap ascending: acc_j = fma(h_tap, mul(dY[t, row[j,tap]], X[t, col[j,tap]]), acc_j)
 * grad_j = mul(gamma, acc_j).  rows/cols: [k][s] cell coordinates of atom j's taps. */
void lw_lowa_sg(long N, int m, int n, int k, int s, const f32 *dY, const f32 *X,
                const int64_t *rows, const int64_t *cols, const f32 *h, f32 gamma, f32 *grad) {
    PAR_FOR("omp for schedule(static)", for (int j = 0; j < k; j++) {
        f32 acc = 0.0f;
        const int64_t *rj = rows + (long)j * s, *cj = cols + (long)j * s;
        for (long t = 0; t < N; t++) {
            const f32 *dy = dY + t * m, *x = X + t * n;
            for (int e = 0; e < s; e++) acc = ffma(h[e], fmul(dy[rj[e]], x[cj[e]]), acc);
        }
        grad[j] = fmul(gamma, acc); })
}

/* ============================================================================ loss (spec 4) */
/* CE: per row b: m = max z, e = exp(z - m), s = sum e, lse = m + ln s, l_b = (lse - z_y) * w_b;
 * loss = (sum_b l_b) * inv_b;  dz = ((e * recip(s)) - onehot) * w_b * inv_b */
void lw_ce(int Bn, int Cn, const f32 *z, const int64_t *y, const f32 *w, f32 inv_b, f32 *loss,
           f32 *dz) {
    unsigned cs = csr_on();
    f32 tot = 0.0f;
    for (int b = 0; b < Bn; b++) {
        const f32 *zb = z + (long)b * Cn;
        f32 m = zb[0];
        for (int c = 1; c < Cn; c++) m = fmax_(m, zb[c]);
        f32 e[16], s = 0.0f;
        for (int c = 0; c < Cn; c++) { e[c] = sfu_exp(fsub(zb[c], m)); s = fadd(s, e[c]); }
        f32 lse = fadd(m, sfu_log(s));
        tot = fadd(tot, fmul(fsub(lse, zb[y[b]]), w[b]));
        f32 rc = sfu_recip(s);
        for (int c = 0; c < Cn; c++) {
            f32 p = fmul(e[c], rc);
            dz[(long)b * Cn + c] = fmul(fmul(fsub(p, c == y[b] ? 1.0f : 0.0f), w[b]), inv_b);
        }
    }
    *loss = fmul(tot, inv_b);
    csr_off(cs);
}

/* MSE (STS-B, C = 1, spec.md section 4): d = sub(z_b, y_b); l_b = mul(mul(d, d), w_b); tot = add(tot, l_b)
 * for b ascending; loss = mul(tot, inv_b); dz_b = mul(mul(d, inv2b), w_b), inv2b = fp32(2 / B_ACT) = mul(2, inv_b)
 * (exact).  Added in P4 (no entry point existed); evaluated left to right as written in spec.md. */
void lw_mse(int Bn, const f32 *z, const f32 *y, const f32 *w, f32 inv_b, f32 inv2b, f32 *loss, f32 *dz) {
    unsigned cs = csr_on();
    f32 tot = 0.0f;
    for (int b = 0; b < Bn; b++) {
        f32 d = fsub(z[b], y[b]);
        tot = fadd(tot, fmul(fmul(d, d), w[b]));
        dz[b] = fmul(fmul(d, inv2b), w[b]);
    }
    *loss = fmul(tot, inv_b);
    csr_off(cs);
}

/* ======================================================================= optimizer (dataflow 6) */
/* Per tensor i, viewed as rows of length rowlen[i] (its last dim): rs = chunked sum of fma(g, g, .)
 * along the row, ss_i = chunked sum of add(rs) over the rows (both KC = 64, in order, from +0);
 * n_i = sqrt(ss_i); G = sqrt(sum fma(n_i, n_i, .)) (28 terms); c = min(1, recip(G + 1e-6)); g *= c. */
void lw_gnorm_clip(int nt, const int64_t *off, const int64_t *rowlen, f32 *g, f32 max_norm_eps,
                   f32 *norms, f32 *Gout, f32 *cout) {
    PAR_FOR("omp for schedule(dynamic)", for (int i = 0; i < nt; i++) {
        long nr = (off[i + 1] - off[i]) / rowlen[i];
        int kc = GEMM_KC > 0 ? GEMM_KC : (int)(rowlen[i] > nr ? rowlen[i] : nr);
        f32 ss;
        { CSUM_BEGIN
          for (long r = 0; r < nr; r++) {
              const f32 *gr = g + off[i] + r * rowlen[i];
              f32 rs;
              { f32 _tot = 0.0f, _p = 0.0f; int _c = 0;
                for (long e = 0; e < rowlen[i]; e++) CSUM_STEP(_p = ffma(gr[e], gr[e], _p))
                CSUM_END(rs) }
              CSUM_STEP(_p = fadd(_p, rs))
          }
          CSUM_END(ss) }
        norms[i] = sfu_sqrt(ss); })
    unsigned cs = csr_on();
    f32 t = 0.0f;
    for (int i = 0; i < nt; i++) t = ffma(norms[i], norms[i], t);
    f32 G = sfu_sqrt(t);
    f32 c = sfu_recip(fadd(G, max_norm_eps));
    if (c > 1.0f) c = 1.0f;
    csr_off(cs);
    long n = off[nt];
    PAR_FOR("omp for schedule(static)", for (long e = 0; e < n; e++) g[e] = fmul(g[e], c);)
    *Gout = G; *cout = c;
}
/* AdamW per element (torch _multi_tensor_adamw order).  sc = {decay, step, recip(bc2s), b1c, b2,
 * b2c, eps}: p = p*decay; m = fma(b1c, g - m, m); v = fma(b2c, g*g, v*b2);
 * den = sqrt(v)*rbc2 + eps; p = p - step*(m*recip(den)) */
void lw_adamw(long n, f32 *p, const f32 *g, f32 *m, f32 *v, const f32 *sc) {
    f32 decay = sc[0], step = sc[1], rbc2 = sc[2], b1c = sc[3], b2 = sc[4], b2c = sc[5], eps = sc[6];
    PAR_FOR("omp for schedule(static)", for (long i = 0; i < n; i++) {
        f32 pp = fmul(p[i], decay);
        f32 mm = ffma(b1c, fsub(g[i], m[i]), m[i]);
        f32 vv = ffma(b2c, fmul(g[i], g[i]), fmul(v[i], b2));
        f32 den = fadd(fmul(sfu_sqrt(vv), rbc2), eps);
        p[i] = fsub(pp, fmul(step, fmul(mm, sfu_recip(den))));
        m[i] = mm; v[i] = vv; })
}

int lw_omp_threads(void) { return omp_get_max_threads(); }

/* ======================================================= measurement only (test_arith.py)
 * Exhaustive ulp scan of one SFU function over the fp32 bit patterns [lo, hi] (inclusive),
 * against a double-precision libm reference.  ulp(v) = 2^(floor(log2|v|) - 23).  Points whose
 * exact result is outside the normal fp32 range are skipped (flush / overflow region) and counted. */
static double ref_fn(int fn, double x) {
    switch (fn) {
    case 0: return exp(x);
    case 1: return 1.0 / x;
    case 2: return 1.0 / sqrt(x);
    case 3: return sqrt(x);
    case 4: return tanh(x);
    case 5: return erf(x);
    default: return log(x);
    }
}
static f32 chip_fn(int fn, f32 x) {
    switch (fn) {
    case 0: return sfu_exp(x);
    case 1: return sfu_recip(x);
    case 2: return sfu_rsqrt(x);
    case 3: return sfu_sqrt(x);
    case 4: return sfu_tanh(x);
    case 5: return sfu_erf(x);
    default: return sfu_log(x);
    }
}
/* out: [max_ulp, argmax_bits, n_tested, n_skipped, n_not_faithful(>=1 ulp), n_not_cr] */
void lw_ulp_scan(int fn, uint32_t lo, uint32_t hi, double *out) {
    double best = 0; uint32_t arg = 0; long nt = 0, ns = 0, nf = 0, ncr = 0;
    _Pragma("omp parallel") {
        unsigned cs = csr_on();
        double lb = 0; uint32_t la = 0; long t = 0, s = 0, f1 = 0, c1 = 0;
        _Pragma("omp for schedule(static)")
        for (int64_t u = lo; u <= (int64_t)hi; u++) {
            f32 x = F((uint32_t)u);
            double r = ref_fn(fn, (double)x);
            double ar = fabs(r);
            if (!(ar >= 0x1p-126 && ar <= 0x1.fffffep127)) { s++; continue; }
            f32 y = chip_fn(fn, x);
            int e; frexp(ar, &e);
            double ulp = ldexp(1.0, e - 24);
            double err = fabs((double)y - r) / ulp;
            t++;
            if (err >= 1.0) f1++;
            if ((f32)r != y) c1++;
            if (err > lb) { lb = err; la = (uint32_t)u; }
        }
        _Pragma("omp critical") {
            if (lb > best) { best = lb; arg = la; }
            nt += t; ns += s; nf += f1; ncr += c1;
        }
        csr_off(cs);
    }
    out[0] = best; out[1] = arg; out[2] = nt; out[3] = ns; out[4] = nf; out[5] = ncr;
}
