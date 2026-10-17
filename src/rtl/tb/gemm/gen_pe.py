#!/usr/bin/env python3
"""PE test vectors (tb/gemm/tb_gm_pe.sv).  Expected chunk partials come ONLY from the L1 `fma`
entry point applied in the contract order (docs/dataflow.md section 2: p = +0; p = fma(a_k, b_k, p),
k ascending within the chunk).  Nothing here computes a result.

    gen_pe.py directed|random NCHUNKS SEED OUT [LAMBDA]
File format (hex): per chunk: one line "K", then K lines "a b_0 .. b_{LAMBDA-1}", then one line
"p_0 .. p_{LAMBDA-1}"; the b_q are the LAMBDA interleaved W rows the PE multiplies the held a by.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model"))
sys.path.insert(0, os.path.join(HERE, "..", "fp"))
import hwmodel as H  # noqa: E402
import gen_vectors as G  # noqa: E402

U32 = np.uint32


def chunks_expected(chunks):
    """[(a [K], b [K, L])] uint32 -> [p [L]] by the L1 fma chain, one L1 call per k step over all
    chunks that are still running (batched only across independent chunks: each chain's order is
    exactly p = +0; p = fma(a_k, b_k, p), k ascending)."""
    L = chunks[0][1].shape[1]
    n = len(chunks)
    p = np.zeros((n, L), np.float32)
    kmax = max(c[0].shape[0] for c in chunks)
    for k in range(kmax):
        idx = np.array([i for i, c in enumerate(chunks) if c[0].shape[0] > k])
        a = np.stack([np.full(L, chunks[i][0][k], U32) for i in idx])
        b = np.stack([chunks[i][1][k] for i in idx])
        p[idx] = H.fma(G.u2f(a), G.u2f(b), p[idx])
    return [G.f2u(p[i]) for i in range(n)]


def main():
    kind, n, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    L = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    r = np.random.default_rng(seed)
    lines = []
    chunks = []
    if kind == "directed":
        sp = np.asarray(G.SPECIAL, U32)
        ks = [1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 33, 63, 64] * 3
        for i, K in enumerate(ks):
            if i % 3 == 0:          # specials everywhere: NaN / inf / zero propagation
                a = sp[r.integers(0, len(sp), K)]
                b = sp[r.integers(0, len(sp), (K, L))]
            elif i % 3 == 1:        # tiny operands: FTZ inside the chain, cancellation
                a = G.rand_normal(r, K, 1, 70)
                b = np.stack([G.rand_normal(r, K, 1, 70) for _ in range(L)], 1)
                b[:, 0] = (b[:, 0] ^ U32(0x80000000))
            else:                   # mixed families + exact ties
                m = G.mix_operands(r, K * L)
                a = m[:K, 0]
                b = m[:K * L, 1].reshape(K, L)
            chunks.append((a.astype(U32), b.astype(U32)))
    else:
        for _ in range(n):
            K = int(r.integers(1, 65)) if r.integers(0, 4) else 64
            fam = r.integers(0, 3)
            if fam == 0:
                m = G.mix_operands(r, K * L)
                a, b = m[:K, 0], m[:K * L, 1].reshape(K, L)
            elif fam == 1:
                a = G.rand_normal(r, K, 100, 154)
                b = np.stack([G.rand_normal(r, K, 100, 154) for _ in range(L)], 1)
            else:
                a = G.rand_normal(r, K, 120, 134)
                b = np.stack([G.rand_normal(r, K, 120, 134) for _ in range(L)], 1)
            chunks.append((a.astype(U32), b.astype(U32)))
    exp = chunks_expected(chunks)
    for (a, b), p in zip(chunks, exp):
        K = a.shape[0]
        lines.append(f"{K:x}")
        for k in range(K):
            lines.append(" ".join(f"{int(v):08x}" for v in [a[k]] + list(b[k])))
        lines.append(" ".join(f"{int(v):08x}" for v in p))
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
