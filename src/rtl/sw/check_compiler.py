#!/usr/bin/env python3
"""Compiler golden test (P6c, `make compiler-check`): compile the full step (roberta-base, B 32, L 128, 12 layers,
step-B batch and state), run the program through the interpreter (sw/interp.py: sequencer decode + L1 per-op entry
points on the DRAM image), run the L1 step (model/hwmodel/step.py) on the same inputs, and compare bit for bit:
loss, logits, all 24 x 256 coefficient gradients, the head gradients, grad norm / clip coefficient, post-step
coefficients + head, post-step Adam m / v.  Prints one line CHECK <name> PASS|FAIL and a final COMPILER-CHECK line.
  python3 sw/check_compiler.py [--layers 12 --batch 32 --seq 128]
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))
import compile as CP  # noqa: E402
import interp  # noqa: E402

f32, U32 = np.float32, np.uint32


def run_interp(c, r, verbose=False):
    m = interp.Machine(CP.IMG_END, r["csr"]["SEED"], r["csr"]["STEP_T"], r["csr"]["SLOT_DELTA"], r["csr"]["DOORBELL_SLOT"])
    m.load(r["img"])
    t0 = time.time()
    n = m.run(r["csr"]["PROG_BASE"], verbose=verbose)
    return m, n, time.time() - t0


def results(c, m):
    """the step's results as stored in the DRAM image (names as step.py's output keys)."""
    nmod = 2 * c.layers
    R = CP.REG
    v = lambda a, n: m.mf[a // 4:a // 4 + n].copy()
    out = {"loss": v(R["LOSS_OUT"], 1)[0], "logits": v(c.cls["z"], c.B * c.C).reshape(c.B, c.C),
           "g_coef": v(R["GRAD"], 256 * nmod).reshape(nmod, 256),
           "gnorm": v(R["OPT_SCAL"] + 4 * CP.O_GFIN, 1)[0], "clip": v(R["OPT_SCAL"] + 4 * CP.O_GFIN + 4, 1)[0]}
    shapes = {"dw": (768, 768), "db": (768,), "ow": (c.C, 768), "ob": (c.C,)}
    for k, sh in shapes.items():
        out["g_" + k] = v(R["GRAD"] + c.head(k), int(np.prod(sh))).reshape(sh)
    for reg, tag in (("PARAM", "post_"), ("ADAM_M", "post_m_"), ("ADAM_V", "post_v_")):
        out[tag + "coef"] = v(R[reg], 256 * nmod).reshape(nmod, 256)
        for k, sh in shapes.items():
            out[tag + k] = v(R[reg] + c.head(k), int(np.prod(sh))).reshape(sh)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=128)
    a = ap.parse_args()
    c = CP.Cfg(a.layers, a.batch, a.seq)
    t0 = time.time()
    r = CP.compile_all(c)
    print(f"compiled: {r['ndesc']} descriptors, {r['prog'].size * 4} bytes, {r['flat_ops']} ops unrolled "
          f"({time.time() - t0:.1f} s)", flush=True)
    m, n, dt = run_interp(c, r)
    print(f"interpreted: {n} descriptors executed in {dt:.1f} s", flush=True)
    got = results(c, m)
    # golden: the L1 step on the same inputs
    import hwmodel.step as S
    assert (a.layers, a.batch, a.seq) == (12, 32, 128), "the L1 step golden exists for the full configuration only"
    W = S.Weights(os.path.join(CP.RUNS, "model_fp32.npz"))
    want = S.step(W, r["inp"], verbose=True)
    nfail = 0
    for k in ["loss", "logits", "g_coef", "g_dw", "g_db", "g_ow", "g_ob", "gnorm", "clip", "post_coef", "post_dw",
              "post_db", "post_ow", "post_ob", "post_m_coef", "post_m_dw", "post_m_db", "post_m_ow", "post_m_ob",
              "post_v_coef", "post_v_dw", "post_v_db", "post_v_ow", "post_v_ob"]:
        g = np.asarray(got[k], f32).view(U32)
        w = np.asarray(want[k], f32).view(U32)
        bad = int((g.reshape(-1) != w.reshape(-1)).sum()) if g.shape == w.shape else -1
        nfail += bad != 0
        print(f"CHECK {k:12s} {w.size:7d} words  {'PASS' if bad == 0 else f'FAIL {bad} mismatches'}")
    print(f"loss interp {float(got['loss']):.9g} L1 {float(want['loss']):.9g}")
    print(f"COMPILER-CHECK {'PASS' if nfail == 0 else 'FAIL'} ({nfail} of 24 result tensors differ)")
    sys.exit(0 if nfail == 0 else 1)


if __name__ == "__main__":
    main()
