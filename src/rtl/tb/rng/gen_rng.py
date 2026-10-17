#!/usr/bin/env python3
"""Vectors for tb/rng (P6b).  Goldens come only from model/rng.py (threefry2x32_13, keep_mask, KAT).

  gen_rng.py tf <out> <n> <seed>        rn_tf vectors: Random123 KAT + n random (counter, key); keys change every
                                        256 vectors (rn_mkgen holds a key per stream), counters every vector.
                                        line: c0 c1 k0 k1 x0 x1 keep(2 bits, from x >> 8 >= DROP_THRESH)
  gen_rng.py mk <out> <case> <mkw>      rn_mkgen request streams: line "idx keep" (keep = MKW bits, bit i =
                                        element idx + i, from rng.keep_mask), plus <out>.cfg (seed step site layer)
     cases: real_attn0  step-B attention-probability mask of layer 0, (32, 12, 128, 128), vu_mask order
            real_ffn5   step-B FFN-output hidden-state mask of layer 5, (4096, 768), GEMM epilogue order
            rand        random seed / step / site / layer, 2^20 elements, 60k random (odd / even) indices
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "model"))
import rng  # noqa: E402

RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "model", "runs")


def tf(out, n, seed):
    r = np.random.default_rng(seed)
    rows = [(c[0], c[1], k[0], k[1]) for c, k, _ in rng.KAT]
    nk = (n + 255) // 256
    keys = r.integers(0, 2 ** 32, size=(nk, 2), dtype=np.uint64)
    ctr = r.integers(0, 2 ** 32, size=(nk * 256, 2), dtype=np.uint64)
    # corner counters / keys
    keys[0] = (0, 0); keys[1] = (0xFFFFFFFF, 0xFFFFFFFF)
    ctr[:8] = [(0, 0), (0xFFFFFFFF, 0xFFFFFFFF), (1, 0), (0, 1), (0x7FFFFFFF, 0x80000000), (0xFFFFFFFF, 0), (0, 0xFFFFFFFF), (2, 3)]
    lines = []
    for c0, c1, k0, k1 in rows:                                   # KAT, one key each
        x0, x1 = rng.threefry2x32_13([c0], [c1], k0, k1)
        lines.append((c0, c1, k0, k1, int(x0[0]), int(x1[0])))
    for i in range(nk):
        k0, k1 = int(keys[i, 0]), int(keys[i, 1])
        c = ctr[256 * i: 256 * (i + 1)]
        x0, x1 = rng.threefry2x32_13(c[:, 0].astype(np.uint32), c[:, 1].astype(np.uint32), k0, k1)
        for j in range(len(c)):
            lines.append((int(c[j, 0]), int(c[j, 1]), k0, k1, int(x0[j]), int(x1[j])))
    th = rng.DROP_THRESH
    with open(out, "w") as f:
        for c0, c1, k0, k1, x0, x1 in lines:
            kp = (int((x1 >> 8) >= th) << 1) | int((x0 >> 8) >= th)
            f.write(f"{c0:08x} {c1:08x} {k0:08x} {k1:08x} {x0:08x} {x1:08x} {kp:x}\n")
    print(f"tf {len(lines)} vectors ({len(rng.KAT)} KAT)")


def write_mk(out, mask, idxs, mkw, cfg):
    m = np.concatenate([np.asarray(mask, np.uint8).reshape(-1), np.zeros(mkw, np.uint8)])
    w = 1 << np.arange(mkw, dtype=np.uint64)
    with open(out, "w") as f:
        for i in idxs:
            v = int((m[i:i + mkw].astype(np.uint64) * w).sum())
            f.write(f"{int(i):08x} {v:x}\n")
    with open(out + ".cfg", "w") as f:
        seed, step, site, layer = cfg
        f.write(f"+mkseed={seed:016x} +mkstep={step:x} +mksite={rng.SITE[site]:x} +mklayer={layer:x}\n")
    print(f"mk {len(idxs)} requests, {m.size - mkw} mask bits, keep rate {1 - mask.mean():.5f} dropped")


def mk(out, case, mkw):
    if case.startswith("real_"):
        z = np.load(os.path.join(RUNS, "stepB_l0f32.npz"))
        seed, step = int(z["seed"]), int(z["step_s"])
        Bn, L = z["input_ids"].shape
    if case == "real_attn0":
        cfg = (seed, step, "attn", 0)
        mask = rng.keep_mask((Bn, 12, L, L), seed, step, "attn", 0)
        rows = Bn * 12 * L
        idx = [((t * 16 + r) * L + mkw * g) for t in range(rows // 16) for g in range(L // mkw) for r in range(16)]
    elif case == "real_ffn5":
        cfg = (seed, step, "ffn_out", 5)
        mask = rng.keep_mask((Bn, L, 768), seed, step, "ffn_out", 5)
        idx = list(range(0, mask.size, mkw))
    elif case == "rand":
        r = np.random.default_rng(mkw)
        cfg = (int(r.integers(0, 2 ** 63)) * 2 + 1, int(r.integers(0, 2 ** 24)), ["emb", "attn", "attn_out", "ffn_out", "cls1", "cls2"][int(r.integers(0, 6))], int(r.integers(0, 12)))
        n = 1 << 20
        mask = rng.keep_mask((n,), *cfg)
        idx = list(r.integers(0, n - mkw, size=60000)) + [0, 1, n - mkw - 1, n - mkw]
    else:
        raise SystemExit("unknown case " + case)
    write_mk(out, mask, idx, mkw, cfg)


if __name__ == "__main__":
    if sys.argv[1] == "tf":
        tf(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        mk(sys.argv[2], sys.argv[3], int(sys.argv[4]))
