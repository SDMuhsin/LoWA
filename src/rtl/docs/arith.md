# LoWA-TA arithmetic contract (P1; binding for P2-P5)

The reference implementation is `model/hwmodel/csrc/lwta.c` (L1). RTL must match it bit for bit.
Evidence: `make model-arith` (`model/hwmodel/test_arith.py`).

## 1. Number format and the three rounded primitives
- IEEE-754 binary32, round to nearest, ties to even (RNE).
- **DAZ:** every subnormal operand reads as a zero of the same sign.
- **FTZ:** first round with unbounded exponent. If the rounded magnitude is below 2^-126, the
  result is a zero with the sign of the exact result (tininess after rounding, as on x86). This
  includes results that are exactly representable subnormals.
- Overflow gives +-inf. Any NaN result is the canonical quiet NaN `0x7FC00000` (no payloads).
  inf*0 and inf-inf give NaN.
- `add(a,b)`, `sub(a,b)`, `mul(a,b)` round once. `fma(a,b,c)` = a*b+c, exact product, one rounding.
- Exact zero sums: +0, except (-0)+(-0) = -0. `max` and compares use the DAZ'd operands.
- Proof: add, mul and fma (311,565 cases each) are compared bit for bit with an exact integer
  reference: decode, form the exact result in integers, round once, then FTZ. The cases cover
  random bit patterns, normals, catastrophic cancellation, exact half-ulp ties, results around
  2^-126 (both sides of the rounding boundary), subnormal inputs, and all specials crossed.
  **0 mismatches.** L1 runs on SSE/AVX-512 under MXCSR = RNE|FTZ|DAZ, set in every thread and
  region and restored on exit. The test is what shows that this hardware mode is exactly the
  semantics above.

## 2. Reductions (P1 change to dataflow.md, see p1_report.md section 3)
**Chunked sum, KC = 64:** a reduction of n terms is split into runs of 64 (the last run may be
short). Each run is accumulated in order from +0 by its op (`fma` or `add`), then the run partials
are added in order into a total that starts at +0. This rule applies to:
- GEMM over K. The PE accumulator holds one K chunk; the ACC drain does `ACC = add(ACC, p)`.
- LayerNorm forward: sum and sum of squares.
- LayerNorm backward: `a` and `b`.
- GNORM: along each row of the last dimension, then over the row sums.

These reductions stay sequential (no chunking): softmax (n <= 128), the LoWA atom accumulators
(D5), the head bias column sums (n = B), the 28-term total norm, and CE over C.

## 3. SFU algorithms (all constants are fp32 bit patterns)
Polynomials use Horner, `p = c_d; p = fma(p, x, c_{k})` for k = d-1..0. "2^k" means the exact
power-of-two constant. `scale2(y, n)` = `mul(mul(y, 2^(n>>1)), 2^(n - (n>>1)))`, two exact
multiplies that underflow to 0 through FTZ. Coefficients come from `model/hwmodel/fit_poly.py`
(weighted minimax, rounded to fp32 from high degree to low with refit).

| fn | algorithm |
|---|---|
| exp | NaN->NaN; x>89 -> +inf; x<-104 -> +0. t = fma(x, L2E=0x3FB8AA3B, M=0x4B400000); n = int(bits(t)-bits(M)); nf = sub(t, M); r = fma(nf, -0x3F317218, x); r = fma(nf, -0xB102E308, r); p = fma(G(r), r, 1), G degree 6 = {0x3F800000, 0x3F000000, 0x3E2AAAAA, 0x3D2AAA41, 0x3C088986, 0x3AB6CAB2, 0x394F4769} (c0..c6); result scale2(p, n) |
| recip | NaN->NaN; +-inf -> +-0; +-0 or subnormal -> +-inf. m = significand in [1,2) (exponent field 127), E = unbiased exponent; y = fma(-m, 0x3EF0F0F1, 0x3FB4B4B5) (24/17 - 8/17 m); 3 x {e = fma(-m, y, 1); y = fma(y, e, y)}; result scale2(y, -E), sign OR'ed in |
| rsqrt | NaN, x<0 (non-zero) -> NaN; +-0/subnormal -> +-inf; +inf -> +0. E = unbiased exp, k = E>>1 (floor), m = x 2^-2k in [1,4) (exponent field 127+E-2k); y = bits 0x5F3759DF - (bits(m)>>1); 3 x {t = mul(y,y); e = fma(-m, t, 1); y = fma(mul(y,0.5), e, y)}; result mul(y, 2^-k) |
| sqrt | +-0/subnormal -> +-0; x<0 -> NaN; +inf -> +inf. Same m, k and y as rsqrt; s = mul(m, y); r = fma(-s, s, m); result mul(fma(r, mul(y,0.5), s), 2^k) |
| tanh | NaN->NaN; a = abs(x); a >= 9 -> +-1. If a < 1: z = mul(x,x); r = fma(mul(x,z), P(z), x), P degree 8 = {0xBEAAAAAB, 0x3E08887D, 0xBD5D0B49, 0x3CB30437, 0xBC1019E8, 0x3B604FFA, 0xBA9B451C, 0x39A05C7B, 0xB82CB570}. Else r = fma(-2, recip(add(exp(mul(a,2)), 1)), 1). Sign of x OR'ed into r |
| erf | NaN->NaN; a = abs(x); a >= 3.92 (0x407AE148) -> +-1. If a < 0.921875: r = fma(x, Q(mul(x,x)), x), Q degree 6 = {0x3E0375D4, 0xBEC093A2, 0x3DE71710, 0xBCDC083E, 0x3BAA6D6D, 0xBA5574D0, 0x38AF88C7}. Else t = fma(R(a), a, a), R degree 9 = {0x3E037DA2, 0x3F22ECFB, 0x3DD393B0, 0xBCA3B3D8, 0x3A233D73, 0x3AA0702B, 0xBA021F57, 0x38D82E17, 0xB746876D, 0x351F8958}; r = sub(1, exp(-t)), sign OR'ed |
| ln (loss value only) | NaN, x<0 -> NaN; +-0/subnormal -> -inf; +inf -> +inf. E = unbiased exp, mb = significand bits with exponent field 127; if mb > 0x3FB504F3: mb -= 1<<23, E += 1; f = sub(mb, 1) (exact); r = fma(mul(f,f), Q(f), f), Q degree 8 = {0xBEFFFFFC, 0x3EAAAAA8, 0xBE800213, 0x3E4CD1FA, 0xBE2A13B6, 0x3E1138C6, 0xBE079BB2, 0x3E04245D, 0xBD9C187A}; r = fma(float(E), 0xB102E308, r); result fma(float(E), 0x3F317218, r) |

**Measured error.** Every fp32 input in the domain is tested exhaustively against a double libm
reference. ulp(v) = 2^(floor(log2 abs(v)) - 23). The worst point was re-checked in mpmath at 50
digits and agrees to 1e-4 ulp. Domain = all inputs whose exact result is a normal fp32 number.
Outside that domain the result is +-0 or +-inf by the rules above.

| fn | inputs tested | domain | max ulp | at x | inputs >= 1 ulp | not correctly rounded |
|---|---|---|---|---|---|---|
| exp | 2,237,668,968 | [-87.336, 88.722] | **0.879** | -71.04119 | 0 | 0.45 % |
| recip | 4,227,858,434 | abs(x) in [2^-126, 2^126] | **0.502** | 1.0023887 * 2^k | 0 | 0.040 % |
| rsqrt | 2,130,706,432 | all normal x > 0 | **0.850** | 1.0490096e-22 | 0 | 13.0 % |
| sqrt | 2,130,706,432 | all normal x > 0 | **0.500** | (none) | 0 | **0 (correctly rounded)** |
| tanh | 4,261,412,864 | all finite x | **1.082** | -0.9122643 | 280 | 0.10 % |
| erf | 4,261,412,864 | all finite x | **1.013** | -1.0600996 | 8 | 0.71 % |
| ln | 2,130,706,431 | all normal x > 0 | **0.921** | 1.571041 | 0 | 0.57 % |

## 4. Composite element-wise ops (evaluated exactly as written)
- dropout, forward and backward (one op): `keep ? mul(x, DS) : +0`, DS = `0x3F8E38E4` = fp32(1/0.9).
- GELU: `g = mul(mul(u, 0.5), add(1, erf(mul(u, 0x3F3504F3))))`.
- GELU': `cdf = mul(0.5, add(1, erf(mul(u, 0x3F3504F3))))`;
  `pdf = mul(exp(mul(mul(u, -0.5), u)), 0x3ECC422A)`; `du = mul(dg, fma(u, pdf, cdf))`.
- tanh': `dx = mul(dy, fma(-a, a, 1))`, a = tanh output.
- CE (C classes, row b): m = max, `e_c = exp(sub(z_c, m))`, s = sum of e_c with add,
  `lse = add(m, ln(s))`, `loss = mul(sum_b mul(sub(lse, z_y), w_b), INV_B)`.
  `dz_c = mul(mul(sub(mul(e_c, recip(s)), onehot), w_b), INV_B)`, INV_B = recip(B_ACT).
- LN, softmax, LoWA, GNORM, step scalars, AdamW: dataflow.md sections 3-6, with the chunked sums
  of section 2 above.

## 5. Golden-vector entry points (`import hwmodel as H`, numpy float32 in and out)
`add sub mul fma fmax fmin cmp i2f scale drop exp recip rsqrt sqrt tanh erf log gelu gelu_bwd tanh_bwd add_bias
colsum gemm gemm_nn gemm_tn gemm_raw (batched, arbitrary strides) ln_fwd ln_bwd smax_fwd smax_bwd
lowa_merge lowa_sg ce gnorm_clip adamw`, plus `ulp_scan`. `H.set_kc(0)` selects P0's single-chain
reductions, for comparison only.
P2 added three entry points, no new numerics: `fmin(a, b)` = `daz(a) > daz(b) ? daz(b) : daz(a)` (so
`fmin(c, 1)` is exactly the clip `c > 1 ? 1 : c` of dataflow.md section 6, NaN included), `cmp(a, b)` =
IEEE compare flags on the DAZ'd operands (bit0 <, bit1 ==, bit2 >, bit3 unordered) and `i2f(x)` =
signed int32 to fp32 under RNE (cvtsi2ss), for the step-scalar integers.
