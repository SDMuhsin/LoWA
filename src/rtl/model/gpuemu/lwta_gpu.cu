/* LoWA-TA L1 hwmodel on the GPU: csrc/lwta.c ported op by op to CUDA (P9d).
 *
 * Every rounded operation is one IEEE binary32 add / mul / fma with round-to-nearest-even
 * (__fadd_rn, __fmul_rn, __fmaf_rn; built with -fmad=false so nothing is contracted) and the
 * GPU's flush-to-zero mode (-ftz=true: subnormal inputs read as zero, subnormal results written
 * as zero).  NaN results are canonicalised to 0x7FC00000 exactly as in lwta.c.  Every reduction
 * keeps lwta.c's per-element order (chunked GEMM with KC = 64, sequential softmax / LoWA / CE /
 * colsum chains, chunked LN and GNORM sums); parallelism is only ever across independent outputs.
 * Bit-exactness against L1 is not assumed: gpuemu/check_gpu.py proves it (primitives on the
 * test_arith.py case set including the 2^-126 boundary, SFU sweeps, every kernel, whole steps).
 * Build: see Makefile in this directory. */
#include <cuda_runtime.h>
#include <stdint.h>

typedef float f32;

/* ------------------------------------------------------------------ primitives */
__device__ __forceinline__ uint32_t B(f32 x) { return __float_as_uint(x); }
__device__ __forceinline__ f32 F(uint32_t u) { return __uint_as_float(u); }
__device__ __forceinline__ f32 canon(f32 y) { return y != y ? F(0x7FC00000u) : y; }
__device__ __forceinline__ f32 fadd(f32 a, f32 b) { return canon(__fadd_rn(a, b)); }
__device__ __forceinline__ f32 fsub(f32 a, f32 b) { return canon(__fsub_rn(a, b)); }
__device__ __forceinline__ f32 fmul(f32 a, f32 b) { return canon(__fmul_rn(a, b)); }
__device__ __forceinline__ f32 ffma(f32 a, f32 b, f32 c) { return canon(__fmaf_rn(a, b, c)); }
__device__ __forceinline__ f32 fneg(f32 a) { return F(B(a) ^ 0x80000000u); }
__device__ __forceinline__ f32 daz(f32 x) { return (B(x) & 0x7f800000u) ? x : F(B(x) & 0x80000000u); }
/* lwta.c fmax_: a > b ? a : b, compiled under MXCSR DAZ (operands never subnormal here) */
__device__ __forceinline__ f32 fmax_(f32 a, f32 b) { return daz(a) > daz(b) ? a : b; }
__device__ __forceinline__ f32 pow2(int k) { return F((uint32_t)(k + 127) << 23); }

#define QNAN F(0x7FC00000u)
#define PINF F(0x7F800000u)
#define C_L2E    F(0x3FB8AA3Bu)
#define C_MAGIC  F(0x4B400000u)
#define C_LN2HI  F(0x3F317218u)
#define C_LN2LO  F(0xB102E308u)
#define C_R0     F(0x3FB4B4B5u)
#define C_R1     F(0x3EF0F0F1u)
#define C_GELU_A F(0x3F3504F3u)
#define C_GELU_B F(0x3ECC422Au)
__constant__ uint32_t EXP_C[7] = {0x3F800000, 0x3F000000, 0x3E2AAAAA, 0x3D2AAA41,
                                  0x3C088986, 0x3AB6CAB2, 0x394F4769};
__constant__ uint32_t TANH_C[9] = {0xBEAAAAAB, 0x3E08887D, 0xBD5D0B49, 0x3CB30437, 0xBC1019E8,
                                   0x3B604FFA, 0xBA9B451C, 0x39A05C7B, 0xB82CB570};
__constant__ uint32_t ERFS_C[7] = {0x3E0375D4, 0xBEC093A2, 0x3DE71710, 0xBCDC083E,
                                   0x3BAA6D6D, 0xBA5574D0, 0x38AF88C7};
__constant__ uint32_t ERFL_C[10] = {0x3E037DA2, 0x3F22ECFB, 0x3DD393B0, 0xBCA3B3D8, 0x3A233D73,
                                    0x3AA0702B, 0xBA021F57, 0x38D82E17, 0xB746876D, 0x351F8958};
__constant__ uint32_t LOG_C[9] = {0xBEFFFFFC, 0x3EAAAAA8, 0xBE800213, 0x3E4CD1FA, 0xBE2A13B6,
                                  0x3E1138C6, 0xBE079BB2, 0x3E04245D, 0xBD9C187A};

__device__ __forceinline__ f32 horner(const uint32_t *c, int deg, f32 x) {
    f32 p = F(c[deg]);
    for (int k = deg - 1; k >= 0; k--) p = ffma(p, x, F(c[k]));
    return p;
}

/* ------------------------------------------------------------------ SFU (lwta.c verbatim) */
__device__ f32 sfu_exp(f32 x) {
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
__device__ f32 sfu_recip(f32 x) {
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
__device__ f32 rsqrt_m(f32 m) {
    f32 y = F(0x5F3759DFu - (B(m) >> 1));
    for (int i = 0; i < 3; i++) {
        f32 t = fmul(y, y);
        f32 e = ffma(fneg(m), t, 1.0f);
        y = ffma(fmul(y, 0.5f), e, y);
    }
    return y;
}
__device__ __forceinline__ int split_even(uint32_t a, f32 *m) {
    int E = (int)(a >> 23) - 127, k = E >> 1;
    *m = F((a & 0x7fffffu) | ((uint32_t)(127 + E - 2 * k) << 23));
    return k;
}
__device__ f32 sfu_rsqrt(f32 x) {
    uint32_t u = B(x), a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return QNAN;
    if (a < 0x00800000u) return F((u & 0x80000000u) | 0x7f800000u);
    if (u & 0x80000000u) return QNAN;
    if (a == 0x7f800000u) return 0.0f;
    f32 m;
    int k = split_even(a, &m);
    return fmul(rsqrt_m(m), pow2(-k));
}
__device__ f32 sfu_sqrt(f32 x) {
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
__device__ f32 sfu_tanh(f32 x) {
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
__device__ f32 sfu_erf(f32 x) {
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
__device__ f32 sfu_log(f32 x) {
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
    f32 Ef = __int2float_rn(E);
    r = ffma(Ef, C_LN2LO, r);
    return ffma(Ef, C_LN2HI, r);
}
__device__ __forceinline__ f32 gelu(f32 u) {
    return fmul(fmul(u, 0.5f), fadd(1.0f, sfu_erf(fmul(u, C_GELU_A))));
}
__device__ __forceinline__ f32 gelu_grad(f32 u) {
    f32 cdf = fmul(0.5f, fadd(1.0f, sfu_erf(fmul(u, C_GELU_A))));
    f32 pdf = fmul(sfu_exp(fmul(fmul(u, -0.5f), u)), C_GELU_B);
    return ffma(u, pdf, cdf);
}

#define GRID(n) (unsigned)(((n) + 255) / 256), 256
#define TID long i = (long)blockIdx.x * blockDim.x + threadIdx.x

/* ======================================================= element-wise (lwta.c ELEM1/ELEM2) */
enum { OP_ADD, OP_SUB, OP_MUL, OP_MAX, OP_MIN, OP_EXP, OP_RECIP, OP_RSQRT, OP_SQRT, OP_TANH,
       OP_ERF, OP_LOG, OP_GELU, OP_GELU_BWD, OP_TANH_BWD, OP_FMA };
__global__ void k_elem(int op, long n, const f32 *x, const f32 *z, const f32 *w, f32 *y) {
    TID;
    if (i >= n) return;
    f32 a = x[i], b = z ? z[i] : 0.0f, r;
    switch (op) {
    case OP_ADD: r = fadd(a, b); break;
    case OP_SUB: r = fsub(a, b); break;
    case OP_MUL: r = fmul(a, b); break;
    case OP_MAX: r = daz(a) > daz(b) ? daz(a) : daz(b); break;
    case OP_MIN: r = daz(a) > daz(b) ? daz(b) : daz(a); break;
    case OP_EXP: r = sfu_exp(a); break;
    case OP_RECIP: r = sfu_recip(a); break;
    case OP_RSQRT: r = sfu_rsqrt(a); break;
    case OP_SQRT: r = sfu_sqrt(a); break;
    case OP_TANH: r = sfu_tanh(a); break;
    case OP_ERF: r = sfu_erf(a); break;
    case OP_LOG: r = sfu_log(a); break;
    case OP_GELU: r = gelu(a); break;
    case OP_GELU_BWD: r = fmul(a, gelu_grad(b)); break;
    case OP_TANH_BWD: r = fmul(a, ffma(fneg(b), b, 1.0f)); break;
    default: r = ffma(a, b, w[i]); break;
    }
    y[i] = canon(r);
}
extern "C" void gw_elem(int op, long n, const f32 *x, const f32 *z, const f32 *w, f32 *y) {
    if (n > 0) k_elem<<<GRID(n)>>>(op, n, x, z, w, y);
}
__global__ void k_scale(long n, const f32 *x, f32 s, f32 *y) { TID; if (i < n) y[i] = fmul(x[i], s); }
extern "C" void gw_scale(long n, const f32 *x, f32 s, f32 *y) { if (n > 0) k_scale<<<GRID(n)>>>(n, x, s, y); }
__global__ void k_drop(long n, const f32 *x, const uint8_t *keep, f32 ds, f32 *y) {
    TID; if (i < n) y[i] = keep[i] ? fmul(x[i], ds) : 0.0f;
}
extern "C" void gw_drop(long n, const f32 *x, const uint8_t *keep, f32 ds, f32 *y) {
    if (n > 0) k_drop<<<GRID(n)>>>(n, x, keep, ds, y);
}
__global__ void k_add_bias(long n, long cols, const f32 *x, const f32 *bias, f32 *y) {
    TID; if (i < n) y[i] = fadd(x[i], bias[i % cols]);
}
extern "C" void gw_add_bias(long rows, long cols, const f32 *x, const f32 *bias, f32 *y) {
    long n = rows * cols;
    if (n > 0) k_add_bias<<<GRID(n)>>>(n, cols, x, bias, y);
}
__global__ void k_colsum(long rows, long cols, const f32 *x, f32 *out) {
    TID;
    if (i >= cols) return;
    f32 s = 0.0f;
    for (long r = 0; r < rows; r++) s = fadd(s, x[r * cols + i]);
    out[i] = s;
}
extern "C" void gw_colsum(long rows, long cols, const f32 *x, f32 *out) { k_colsum<<<GRID(cols)>>>(rows, cols, x, out); }

/* ================================================================== GEMM (lwta.c lw_gemm)
 * C(i,j) = sum_k A(i,k) B(j,k): per output p = +0; p = fma(A, B, p) over k ascending inside each
 * KC = 64 chunk; ACC = add(ACC, p) over chunks ascending from +0.  Tiles BM x BN x BK with BK | 64,
 * so chunk boundaries fall on tile boundaries; a partial last tile runs only the real k (no padded
 * fma, which could turn a -0 partial into +0). */
#define BM 64
#define BN 64
#define BK 16
#define KC 64
__global__ void __launch_bounds__(256) k_gemm(int nb2, int M, int N, int K,
        const f32 *A, long sA1, long sA2, long a_r, long a_k,
        const f32 *Bm, long sB1, long sB2, long b_n, long b_k,
        f32 *C, long sC1, long sC2, long c_r, long c_c) {
    int bz = blockIdx.z, b1 = bz / nb2, b2 = bz % nb2;
    A += b1 * sA1 + b2 * sA2;
    Bm += b1 * sB1 + b2 * sB2;
    C += b1 * sC1 + b2 * sC2;
    int i0 = blockIdx.y * BM, j0 = blockIdx.x * BN;
    __shared__ f32 As[BK][BM + 4], Bs[BK][BN + 4];
    int t = threadIdx.x, tx = t % 16, ty = t / 16;
    f32 acc[4][4], tot[4][4];
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc[r][c] = tot[r][c] = 0.0f;
    bool a_kmaj = (a_k == 1), b_kmaj = (b_k == 1);
    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int l = 0; l < 4; l++) {
            int e = t + 256 * l, kk, ii;
            if (a_kmaj) { kk = e % BK; ii = e / BK; } else { ii = e % BM; kk = e / BM; }
            int gi = i0 + ii, gk = k0 + kk;
            As[kk][ii] = (gi < M && gk < K) ? A[(long)gi * a_r + (long)gk * a_k] : 0.0f;
            int jj;
            if (b_kmaj) { kk = e % BK; jj = e / BK; } else { jj = e % BN; kk = e / BN; }
            int gj = j0 + jj;
            gk = k0 + kk;
            Bs[kk][jj] = (gj < N && gk < K) ? Bm[(long)gj * b_n + (long)gk * b_k] : 0.0f;
        }
        __syncthreads();
        int kmax = K - k0 < BK ? K - k0 : BK;
        if (kmax == BK) {
#pragma unroll
            for (int kk = 0; kk < BK; kk++) {
                f32 a[4], b[4];
#pragma unroll
                for (int r = 0; r < 4; r++) a[r] = As[kk][ty + 16 * r];
#pragma unroll
                for (int c = 0; c < 4; c++) b[c] = Bs[kk][tx + 16 * c];
#pragma unroll
                for (int r = 0; r < 4; r++)
#pragma unroll
                    for (int c = 0; c < 4; c++) acc[r][c] = __fmaf_rn(a[r], b[c], acc[r][c]);
            }
        } else {
            for (int kk = 0; kk < kmax; kk++) {
                f32 a[4], b[4];
#pragma unroll
                for (int r = 0; r < 4; r++) a[r] = As[kk][ty + 16 * r];
#pragma unroll
                for (int c = 0; c < 4; c++) b[c] = Bs[kk][tx + 16 * c];
#pragma unroll
                for (int r = 0; r < 4; r++)
#pragma unroll
                    for (int c = 0; c < 4; c++) acc[r][c] = __fmaf_rn(a[r], b[c], acc[r][c]);
            }
        }
        __syncthreads();
        if ((k0 + BK) % KC == 0 || k0 + BK >= K) {
#pragma unroll
            for (int r = 0; r < 4; r++)
#pragma unroll
                for (int c = 0; c < 4; c++) { tot[r][c] = __fadd_rn(tot[r][c], acc[r][c]); acc[r][c] = 0.0f; }
        }
    }
#pragma unroll
    for (int r = 0; r < 4; r++) {
        int gi = i0 + ty + 16 * r;
        if (gi >= M) continue;
#pragma unroll
        for (int c = 0; c < 4; c++) {
            int gj = j0 + tx + 16 * c;
            if (gj < N) C[(long)gi * c_r + (long)gj * c_c] = canon(tot[r][c]);
        }
    }
}
/* Larger tiles for the big projections: 128 x 128 x 8, 8 x 8 outputs per thread.  Same per-output
 * chain: fma over k ascending, drain into ACC with add at every multiple of KC and at the end. */
#define B2M 128
#define B2N 128
#define B2K 8
__global__ void __launch_bounds__(256) k_gemm128(int nb2, int M, int N, int K,
        const f32 *A, long sA1, long sA2, long a_r, long a_k,
        const f32 *Bm, long sB1, long sB2, long b_n, long b_k,
        f32 *C, long sC1, long sC2, long c_r, long c_c) {
    int bz = blockIdx.z, b1 = bz / nb2, b2 = bz % nb2;
    A += b1 * sA1 + b2 * sA2;
    Bm += b1 * sB1 + b2 * sB2;
    C += b1 * sC1 + b2 * sC2;
    int i0 = blockIdx.y * B2M, j0 = blockIdx.x * B2N;
    __shared__ f32 As[B2K][B2M + 4], Bs[B2K][B2N + 4];
    int t = threadIdx.x, tx = t % 16, ty = t / 16;
    f32 acc[8][8], tot[8][8];
#pragma unroll
    for (int r = 0; r < 8; r++)
#pragma unroll
        for (int c = 0; c < 8; c++) acc[r][c] = tot[r][c] = 0.0f;
    bool a_kmaj = (a_k == 1), b_kmaj = (b_k == 1);
    for (int k0 = 0; k0 < K; k0 += B2K) {
#pragma unroll
        for (int l = 0; l < 4; l++) {
            int e = t + 256 * l, kk, ii;
            if (a_kmaj) { kk = e % B2K; ii = e / B2K; } else { ii = e % B2M; kk = e / B2M; }
            int gi = i0 + ii, gk = k0 + kk;
            As[kk][ii] = (gi < M && gk < K) ? A[(long)gi * a_r + (long)gk * a_k] : 0.0f;
            int jj;
            if (b_kmaj) { kk = e % B2K; jj = e / B2K; } else { jj = e % B2N; kk = e / B2N; }
            int gj = j0 + jj;
            gk = k0 + kk;
            Bs[kk][jj] = (gj < N && gk < K) ? Bm[(long)gj * b_n + (long)gk * b_k] : 0.0f;
        }
        __syncthreads();
        int kmax = K - k0 < B2K ? K - k0 : B2K;
        if (kmax == B2K) {
#pragma unroll
            for (int kk = 0; kk < B2K; kk++) {
                f32 a[8], b[8];
#pragma unroll
                for (int r = 0; r < 8; r++) a[r] = As[kk][ty + 16 * r];
#pragma unroll
                for (int c = 0; c < 8; c++) b[c] = Bs[kk][tx + 16 * c];
#pragma unroll
                for (int r = 0; r < 8; r++)
#pragma unroll
                    for (int c = 0; c < 8; c++) acc[r][c] = __fmaf_rn(a[r], b[c], acc[r][c]);
            }
        } else {
            for (int kk = 0; kk < kmax; kk++) {
                f32 a[8], b[8];
#pragma unroll
                for (int r = 0; r < 8; r++) a[r] = As[kk][ty + 16 * r];
#pragma unroll
                for (int c = 0; c < 8; c++) b[c] = Bs[kk][tx + 16 * c];
#pragma unroll
                for (int r = 0; r < 8; r++)
#pragma unroll
                    for (int c = 0; c < 8; c++) acc[r][c] = __fmaf_rn(a[r], b[c], acc[r][c]);
            }
        }
        __syncthreads();
        if ((k0 + B2K) % KC == 0 || k0 + B2K >= K) {
#pragma unroll
            for (int r = 0; r < 8; r++)
#pragma unroll
                for (int c = 0; c < 8; c++) { tot[r][c] = __fadd_rn(tot[r][c], acc[r][c]); acc[r][c] = 0.0f; }
        }
    }
#pragma unroll
    for (int r = 0; r < 8; r++) {
        int gi = i0 + ty + 16 * r;
        if (gi >= M) continue;
#pragma unroll
        for (int c = 0; c < 8; c++) {
            int gj = j0 + tx + 16 * c;
            if (gj < N) C[(long)gi * c_r + (long)gj * c_c] = canon(tot[r][c]);
        }
    }
}
extern "C" void gw_gemm(int nb1, int nb2, int M, int N, int K,
                        const f32 *A, long sA1, long sA2, long a_r, long a_k,
                        const f32 *Bm, long sB1, long sB2, long b_n, long b_k,
                        f32 *C, long sC1, long sC2, long c_r, long c_c) {
    if (M >= 128 && N >= 128) {
        dim3 g2((N + B2N - 1) / B2N, (M + B2M - 1) / B2M, nb1 * nb2);
        k_gemm128<<<g2, 256>>>(nb2, M, N, K, A, sA1, sA2, a_r, a_k, Bm, sB1, sB2, b_n, b_k, C, sC1, sC2, c_r, c_c);
        return;
    }
    dim3 g((N + BN - 1) / BN, (M + BM - 1) / BM, nb1 * nb2);
    k_gemm<<<g, 256>>>(nb2, M, N, K, A, sA1, sA2, a_r, a_k, Bm, sB1, sB2, b_n, b_k, C, sC1, sC2, c_r, c_c);
}

/* ===================================================================== LayerNorm (lwta.c) */
#define CSUM_BEGIN f32 _tot = 0.0f, _p = 0.0f; int _c = 0;
#define CSUM_STEP(update) { update; if (++_c == kc) { _tot = fadd(_tot, _p); _p = 0.0f; _c = 0; } }
#define CSUM_END(out) { if (_c) _tot = fadd(_tot, _p); out = _tot; }
__global__ void k_ln_fwd(long rows, int d, const f32 *x, const f32 *g, const f32 *b, f32 eps, f32 inv_d,
                         f32 *y, f32 *mu_out, f32 *rstd_out) {
    TID;
    if (i >= rows) return;
    const int kc = KC;
    long r = i;
    const f32 *xr = x + r * d;
    f32 s, q;
    { CSUM_BEGIN for (int e = 0; e < d; e++) CSUM_STEP(_p = fadd(_p, xr[e])) CSUM_END(s) }
    f32 mu = fmul(s, inv_d);
    { CSUM_BEGIN for (int e = 0; e < d; e++) { f32 t = fsub(xr[e], mu); CSUM_STEP(_p = ffma(t, t, _p)) } CSUM_END(q) }
    f32 var = fmul(q, inv_d);
    f32 rstd = sfu_rsqrt(fadd(var, eps));
    for (int e = 0; e < d; e++) y[r * d + e] = ffma(fmul(fsub(xr[e], mu), rstd), g[e], b[e]);
    mu_out[r] = mu; rstd_out[r] = rstd;
}
extern "C" void gw_ln_fwd(long rows, int d, const f32 *x, const f32 *g, const f32 *b, f32 eps, f32 inv_d,
                          f32 *y, f32 *mu, f32 *rs) {
    k_ln_fwd<<<GRID(rows)>>>(rows, d, x, g, b, eps, inv_d, y, mu, rs);
}
__global__ void k_ln_bwd(long rows, int d, const f32 *z, const f32 *mu, const f32 *rstd, const f32 *g,
                         const f32 *dy, f32 inv_d, f32 *dx) {
    TID;
    if (i >= rows) return;
    const int kc = KC;
    long r = i;
    const f32 *zr = z + r * d, *dr = dy + r * d;
    f32 m = mu[r], rs = rstd[r], a, bb;
    { CSUM_BEGIN for (int e = 0; e < d; e++) CSUM_STEP(_p = fadd(_p, fmul(dr[e], g[e]))) CSUM_END(a) }
    { CSUM_BEGIN for (int e = 0; e < d; e++) {
        f32 gy = fmul(dr[e], g[e]);
        f32 xh = fmul(fsub(zr[e], m), rs);
        CSUM_STEP(_p = ffma(gy, xh, _p)) } CSUM_END(bb) }
    f32 ad = fmul(a, inv_d), bd = fmul(bb, inv_d);
    for (int e = 0; e < d; e++) {
        f32 gy = fmul(dr[e], g[e]);
        f32 xh = fmul(fsub(zr[e], m), rs);
        dx[r * d + e] = fmul(rs, fsub(fsub(gy, ad), fmul(xh, bd)));
    }
}
extern "C" void gw_ln_bwd(long rows, int d, const f32 *z, const f32 *mu, const f32 *rstd, const f32 *g,
                          const f32 *dy, f32 inv_d, f32 *dx) {
    k_ln_bwd<<<GRID(rows)>>>(rows, d, z, mu, rstd, g, dy, inv_d, dx);
}

/* ======================================================================= softmax (lwta.c) */
__global__ void k_smax_fwd(int Bn, int H, int L, const f32 *S, const uint8_t *kmask, const uint8_t *keep,
                           f32 ds, f32 *P, f32 *Pd, f32 *mrow, f32 *rrow) {
    TID;
    long rows = (long)Bn * H * L;
    if (i >= rows) return;
    long r = i;
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
    mrow[r] = m; rrow[r] = rc;
}
/* Tiled versions: a block stages RB whole rows in shared memory with coalesced loads / stores, and
 * thread r runs row r's sequential chain on the staged copy.  Same per-row ops in the same order. */
#define SRB 64
#define LMAX 128
__global__ void k_smax_fwd_t(int Bn, int H, int L, const f32 *S, const uint8_t *kmask, const uint8_t *keep,
                             f32 ds, f32 *P, f32 *Pd, f32 *mrow, f32 *rrow) {
    __shared__ f32 sS[SRB][LMAX + 1];
    long rows = (long)Bn * H * L, r0 = (long)blockIdx.x * SRB;
    int nr = rows - r0 < SRB ? (int)(rows - r0) : SRB, t = threadIdx.x;
    for (int q = t; q < nr * L; q += SRB) sS[q / L][q % L] = S[r0 * L + q];
    __syncthreads();
    if (t < nr) {
        long r = r0 + t;
        int b = (int)(r / ((long)H * L));
        const uint8_t *km = kmask + (long)b * L;
        f32 *s = sS[t];
        f32 m = F(0xff800000u);
        for (int j = 0; j < L; j++) if (km[j]) m = fmax_(m, s[j]);
        f32 sum = 0.0f;
        for (int j = 0; j < L; j++) {
            f32 e = km[j] ? sfu_exp(fsub(s[j], m)) : 0.0f;
            s[j] = e;
            sum = fadd(sum, e);
        }
        f32 rc = sfu_recip(sum);
        for (int j = 0; j < L; j++) s[j] = fmul(s[j], rc);
        mrow[r] = m; rrow[r] = rc;
    }
    __syncthreads();
    for (int q = t; q < nr * L; q += SRB) {
        f32 p = sS[q / L][q % L];
        long gi = r0 * L + q;
        P[gi] = p;
        Pd[gi] = (!keep || keep[gi]) ? (keep ? fmul(p, ds) : p) : 0.0f;
    }
}
extern "C" void gw_smax_fwd(int Bn, int H, int L, const f32 *S, const uint8_t *kmask, const uint8_t *keep,
                            f32 ds, f32 *P, f32 *Pd, f32 *m, f32 *r) {
    long rows = (long)Bn * H * L;
    if (L <= LMAX) k_smax_fwd_t<<<(unsigned)((rows + SRB - 1) / SRB), SRB>>>(Bn, H, L, S, kmask, keep, ds, P, Pd, m, r);
    else k_smax_fwd<<<GRID(rows)>>>(Bn, H, L, S, kmask, keep, ds, P, Pd, m, r);
}
__global__ void k_smax_bwd(long rows, int L, const f32 *P, const f32 *dPd, const uint8_t *keep, f32 ds,
                           f32 scale, f32 *dS) {
    TID;
    if (i >= rows) return;
    long r = i;
    const f32 *p = P + r * L, *g = dPd + r * L;
    f32 rd = 0.0f;
    for (int j = 0; j < L; j++) {
        f32 dp = (!keep || keep[r * L + j]) ? (keep ? fmul(g[j], ds) : g[j]) : 0.0f;
        rd = ffma(p[j], dp, rd);
    }
    for (int j = 0; j < L; j++) {
        f32 dp = (!keep || keep[r * L + j]) ? (keep ? fmul(g[j], ds) : g[j]) : 0.0f;
        dS[r * L + j] = fmul(fmul(p[j], fsub(dp, rd)), scale);
    }
}
#define BRB 32
__global__ void k_smax_bwd_t(long rows, int L, const f32 *P, const f32 *dPd, const uint8_t *keep, f32 ds,
                             f32 scale, f32 *dS) {
    __shared__ f32 sP[BRB][LMAX + 1], sG[BRB][LMAX + 1];
    long r0 = (long)blockIdx.x * BRB;
    int nr = rows - r0 < BRB ? (int)(rows - r0) : BRB, t = threadIdx.x;
    for (int q = t; q < nr * L; q += BRB) {
        long gi = r0 * L + q;
        f32 g = dPd[gi];
        sP[q / L][q % L] = P[gi];
        sG[q / L][q % L] = (!keep || keep[gi]) ? (keep ? fmul(g, ds) : g) : 0.0f;   /* dp */
    }
    __syncthreads();
    if (t < nr) {
        f32 *p = sP[t], *dp = sG[t];
        f32 rd = 0.0f;
        for (int j = 0; j < L; j++) rd = ffma(p[j], dp[j], rd);
        for (int j = 0; j < L; j++) dp[j] = fmul(fmul(p[j], fsub(dp[j], rd)), scale);
    }
    __syncthreads();
    for (int q = t; q < nr * L; q += BRB) dS[r0 * L + q] = sG[q / L][q % L];
}
extern "C" void gw_smax_bwd(long rows, int L, const f32 *P, const f32 *dPd, const uint8_t *keep, f32 ds,
                            f32 scale, f32 *dS) {
    if (L <= LMAX) k_smax_bwd_t<<<(unsigned)((rows + BRB - 1) / BRB), BRB>>>(rows, L, P, dPd, keep, ds, scale, dS);
    else k_smax_bwd<<<GRID(rows)>>>(rows, L, P, dPd, keep, ds, scale, dS);
}

/* ===================================================================== LoWA (lwta.c) */
/* merge-on-load, e ascending.  One thread per distinct cell walks that cell's hits in ascending e
 * (host-built lists: cell_start[c]..cell_start[c+1] index into order[], ascending e within a cell);
 * the result equals lwta.c's single sequential loop because hits on different cells never interact. */
__global__ void k_lowa_merge(int ncell, const int64_t *cell_start, const int64_t *order, int s,
                             const int64_t *eflat, const f32 *coef, const f32 *h, f32 gamma, f32 *W) {
    TID;
    if (i >= ncell) return;
    for (long q = cell_start[i]; q < cell_start[i + 1]; q++) {
        long e = order[q];
        f32 v = fmul(gamma, coef[e / s]);
        W[eflat[e]] = fadd(W[eflat[e]], fmul(v, h[e % s]));
    }
}
extern "C" void gw_lowa_merge(int ncell, const int64_t *cell_start, const int64_t *order, int s,
                              const int64_t *eflat, const f32 *coef, const f32 *h, f32 gamma, f32 *W) {
    k_lowa_merge<<<GRID(ncell)>>>(ncell, cell_start, order, s, eflat, coef, h, gamma, W);
}
__global__ void k_lowa_sg(long N, int m, int n, int k, int s, const f32 *dY, const f32 *X,
                          const int64_t *rows, const int64_t *cols, const f32 *h, f32 gamma, f32 *grad) {
    TID;
    if (i >= k) return;
    int j = (int)i;
    f32 acc = 0.0f;
    const int64_t *rj = rows + (long)j * s, *cj = cols + (long)j * s;
    for (long t = 0; t < N; t++) {
        const f32 *dy = dY + t * m, *x = X + t * n;
        for (int e = 0; e < s; e++) acc = ffma(h[e], fmul(dy[rj[e]], x[cj[e]]), acc);
    }
    grad[j] = fmul(gamma, acc);
}
/* Same result in two kernels: the products mul(dY[t, row], X[t, col]) do not depend on the chain,
 * so k_sg_prod forms all of them in parallel (prod[(t*s + e)*k + j]) and k_sg_chain runs each atom's
 * fma chain over (t, e) ascending on the stored products.  Identical rounded ops in identical order. */
__global__ void k_sg_prod(long N, int m, int n, int k, int s, const f32 *dY, const f32 *X,
                          const int64_t *rows, const int64_t *cols, f32 *prod) {
    TID;
    long tot = N * s * (long)k;
    if (i >= tot) return;
    int j = (int)(i % k);
    long te = i / k;
    int e = (int)(te % s);
    long t = te / s;
    prod[i] = fmul(dY[t * m + rows[(long)j * s + e]], X[t * n + cols[(long)j * s + e]]);
}
__global__ void k_sg_chain(long N, int k, int s, const f32 *prod, const f32 *h, f32 gamma, f32 *grad) {
    TID;
    if (i >= k) return;
    f32 hh[32];
    for (int e = 0; e < s; e++) hh[e] = h[e];
    f32 acc = 0.0f;
    const f32 *p = prod + i;
    for (long t = 0; t < N; t++, p += (long)s * k)
        for (int e = 0; e < s; e++) acc = ffma(hh[e], p[(long)e * k], acc);
    grad[i] = fmul(gamma, acc);
}
/* s = 25 (K = 3 hops): taps unrolled, next token's products prefetched while the chain runs */
template <int S>
__global__ void k_sg_chain_s(long N, int k, const f32 *prod, const f32 *h, f32 gamma, f32 *grad) {
    TID;
    if (i >= k) return;
    f32 hh[S], cur[S], nxt[S];
#pragma unroll
    for (int e = 0; e < S; e++) { hh[e] = h[e]; cur[e] = prod[(long)e * k + i]; }
    f32 acc = 0.0f;
    for (long t = 0; t < N; t++) {
        const f32 *pn = prod + (t + 1 < N ? t + 1 : t) * (long)S * k + i;
#pragma unroll
        for (int e = 0; e < S; e++) nxt[e] = pn[(long)e * k];
#pragma unroll
        for (int e = 0; e < S; e++) acc = ffma(hh[e], cur[e], acc);
#pragma unroll
        for (int e = 0; e < S; e++) cur[e] = nxt[e];
    }
    grad[i] = fmul(gamma, acc);
}
extern "C" void gw_lowa_sg(long N, int m, int n, int k, int s, const f32 *dY, const f32 *X,
                           const int64_t *rows, const int64_t *cols, const f32 *h, f32 gamma, f32 *grad,
                           f32 *work) {
    if (work && s == 25) {
        long tot = N * s * (long)k;
        k_sg_prod<<<GRID(tot)>>>(N, m, n, k, s, dY, X, rows, cols, work);
        k_sg_chain_s<25><<<(k + 31) / 32, 32>>>(N, k, work, h, gamma, grad);
        return;
    }
    if (!work || s > 32) { k_lowa_sg<<<(k + 63) / 64, 64>>>(N, m, n, k, s, dY, X, rows, cols, h, gamma, grad); return; }
    long tot = N * s * (long)k;
    k_sg_prod<<<GRID(tot)>>>(N, m, n, k, s, dY, X, rows, cols, work);
    k_sg_chain<<<(k + 31) / 32, 32>>>(N, k, s, work, h, gamma, grad);
}

/* ============================================================================ loss (lwta.c) */
__global__ void k_ce(int Bn, int Cn, const f32 *z, const int64_t *y, const f32 *w, f32 inv_b, f32 *loss,
                     f32 *dz) {
    if (threadIdx.x || blockIdx.x) return;
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
}
extern "C" void gw_ce(int Bn, int Cn, const f32 *z, const int64_t *y, const f32 *w, f32 inv_b, f32 *loss,
                      f32 *dz) {
    k_ce<<<1, 1>>>(Bn, Cn, z, y, w, inv_b, loss, dz);
}

/* ======================================================================= optimizer (lwta.c) */
__global__ void k_gnorm_part(int nt, const int64_t *off, const int64_t *rowlen, const f32 *g, f32 *norms) {
    TID;
    if (i >= nt) return;
    const int kc = KC;
    long nr = (off[i + 1] - off[i]) / rowlen[i];
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
    norms[i] = sfu_sqrt(ss);
}
__global__ void k_gnorm_tot(int nt, const f32 *norms, f32 eps, f32 *Gc) {
    if (threadIdx.x || blockIdx.x) return;
    f32 t = 0.0f;
    for (int i = 0; i < nt; i++) t = ffma(norms[i], norms[i], t);
    f32 G = sfu_sqrt(t);
    f32 c = sfu_recip(fadd(G, eps));
    if (c > 1.0f) c = 1.0f;
    Gc[0] = G; Gc[1] = c;
}
__global__ void k_gscale(long n, f32 *g, const f32 *Gc) { TID; if (i < n) g[i] = fmul(g[i], Gc[1]); }
/* Same sums split in two kernels: each row's chunked sum rs is independent of the other rows
 * (k_gnorm_rows, one thread per row of all tensors), then per tensor the chunked sum over its rows
 * in row order (k_gnorm_outer).  rowbase[i] = index of tensor i's first row in rs[]. */
__global__ void k_gnorm_rows(long nrows_all, int nt, const int64_t *off, const int64_t *rowlen,
                             const int64_t *rowbase, const f32 *g, f32 *rs_out) {
    TID;
    if (i >= nrows_all) return;
    const int kc = KC;
    int ti = 0;
    while (ti + 1 < nt && rowbase[ti + 1] <= i) ti++;
    long r = i - rowbase[ti];
    const f32 *gr = g + off[ti] + r * rowlen[ti];
    f32 rs;
    { CSUM_BEGIN for (long e = 0; e < rowlen[ti]; e++) CSUM_STEP(_p = ffma(gr[e], gr[e], _p)) CSUM_END(rs) }
    rs_out[i] = rs;
}
__global__ void k_gnorm_outer(int nt, const int64_t *off, const int64_t *rowlen, const int64_t *rowbase,
                              const f32 *rs, f32 *norms) {
    TID;
    if (i >= nt) return;
    const int kc = KC;
    long nr = (off[i + 1] - off[i]) / rowlen[i];
    f32 ss;
    { CSUM_BEGIN for (long r = 0; r < nr; r++) CSUM_STEP(_p = fadd(_p, rs[rowbase[i] + r])) CSUM_END(ss) }
    norms[i] = sfu_sqrt(ss);
}
extern "C" void gw_gnorm_clip2(int nt, const int64_t *off, const int64_t *rowlen, const int64_t *rowbase,
                               long nrows_all, f32 *rs_work, long n, f32 *g, f32 eps, f32 *norms, f32 *Gc) {
    k_gnorm_rows<<<GRID(nrows_all)>>>(nrows_all, nt, off, rowlen, rowbase, g, rs_work);
    k_gnorm_outer<<<1, 32>>>(nt, off, rowlen, rowbase, rs_work, norms);
    k_gnorm_tot<<<1, 1>>>(nt, norms, eps, Gc);
    k_gscale<<<GRID(n)>>>(n, g, Gc);
}
extern "C" void gw_gnorm_clip(int nt, const int64_t *off, const int64_t *rowlen, long n, f32 *g, f32 eps,
                              f32 *norms, f32 *Gc) {
    k_gnorm_part<<<1, 32>>>(nt, off, rowlen, g, norms);
    k_gnorm_tot<<<1, 1>>>(nt, norms, eps, Gc);
    k_gscale<<<GRID(n)>>>(n, g, Gc);
}
__global__ void k_adamw(long n, f32 *p, const f32 *g, f32 *m, f32 *v, f32 decay, f32 step, f32 rbc2,
                        f32 b1c, f32 b2, f32 b2c, f32 eps) {
    TID;
    if (i >= n) return;
    f32 pp = fmul(p[i], decay);
    f32 mm = ffma(b1c, fsub(g[i], m[i]), m[i]);
    f32 vv = ffma(b2c, fmul(g[i], g[i]), fmul(v[i], b2));
    f32 den = fadd(fmul(sfu_sqrt(vv), rbc2), eps);
    p[i] = fsub(pp, fmul(step, fmul(mm, sfu_recip(den))));
    m[i] = mm; v[i] = vv;
}
extern "C" void gw_adamw(long n, f32 *p, const f32 *g, f32 *m, f32 *v, const f32 *sc) {
    k_adamw<<<GRID(n)>>>(n, p, g, m, v, sc[0], sc[1], sc[2], sc[3], sc[4], sc[5], sc[6]);
}

/* ======================================================= Threefry-2x32-13 masks (model/rng.py) */
__global__ void k_keep(long n, uint32_t c1, uint32_t k0, uint32_t k1, uint32_t thresh, uint8_t *out) {
    TID;
    long np_ = (n + 1) / 2;
    if (i >= np_) return;
    const int ROT[8] = {13, 15, 26, 6, 17, 29, 16, 24};
    uint32_t ks[3] = {k0, k1, 0x1BD11BDAu ^ k0 ^ k1};
    uint32_t x0 = (uint32_t)i + ks[0], x1 = c1 + ks[1];
    for (int r = 0; r < 13; r++) {
        x0 += x1;
        int s = ROT[r % 8];
        x1 = (x1 << s) | (x1 >> (32 - s));
        x1 ^= x0;
        if (r % 4 == 3) {
            int j = r / 4 + 1;
            x0 += ks[j % 3];
            x1 += ks[(j + 1) % 3] + (uint32_t)j;
        }
    }
    out[2 * i] = (x0 >> 8) >= thresh;
    if (2 * i + 1 < n) out[2 * i + 1] = (x1 >> 8) >= thresh;
}
extern "C" void gw_keep(long n, uint32_t c1, uint32_t k0, uint32_t k1, uint32_t thresh, uint8_t *out) {
    long np_ = (n + 1) / 2;
    k_keep<<<GRID(np_)>>>(n, c1, k0, k1, thresh, out);
}

extern "C" int gw_sync(void) { return (int)cudaDeviceSynchronize(); }
extern "C" int gw_last_error(void) { return (int)cudaGetLastError(); }
