"""Gate L1 vs L0 (P1).  Truth = L0-64 (float64 CPU).  For every quantity Q:

  e32 = |L0-fp32 - L0-64| / |L0-64|,   e1 = |L1 - L0-64| / |L0-64|      (element-wise)
  PASS iff  median(e1) <= 4 * median(e32)   and   max(e1) <= 10 * max(e32)      (fixed in P1)

Elements with L0-64 == 0 use the absolute error (both then must be 0 for a 0 reference).  A
quantity whose L0-fp32 error is exactly 0 passes only if L1's is exactly 0 too.
  python check_l1_l0.py [RUN_DIR] [A B] [--l1 TAG]     (TAG l1_kc0: the P0 single-chain variant)
"""
import os
import sys

import numpy as np

HEAD = ("dw", "db", "ow", "ob")
QUANT = (["loss", "logits", "g_coef"] + ["g_" + k for k in HEAD] + ["gnorm", "clip"]
         + ["post_coef"] + ["post_" + k for k in HEAD]
         + ["post_m_coef"] + ["post_m_" + k for k in HEAD] + ["post_v_coef"] + ["post_v_" + k for k in HEAD])


def relerr(x, ref):
    x, ref = np.asarray(x, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    d = np.abs(x - ref)
    nz = ref != 0
    d[nz] /= np.abs(ref[nz])
    return d


def gate(run_dir, name, l1="l1"):
    z = {t: np.load(os.path.join(run_dir, f"step{name}_{t}.npz")) for t in ("l0f32", "l0f64")}
    z["l1"] = np.load(os.path.join(run_dir, f"step{name}_{l1}.npz"))
    rows, ok = [], True
    for q in QUANT:
        ref = z["l0f64"][q]
        e32, e1 = relerr(z["l0f32"][q], ref), relerr(z["l1"][q], ref)
        m32, m1, x32, x1 = np.median(e32), np.median(e1), e32.max(), e1.max()
        if x32 == 0:
            p = x1 == 0
        else:
            p = (m1 <= 4 * m32) and (x1 <= 10 * x32)
        ok &= bool(p)
        rows.append((q, ref.size, m32, m1, x32, x1, p))
    return rows, ok, z


def fmt(rows, name):
    out = [f"step {name}: quantity, n, median rel err (L0-fp32 | L1) vs L0-64, max (L0-fp32 | L1), ratio med / max, PASS",
           f"{'quantity':12s} {'n':>7s} {'med32':>9s} {'medL1':>9s} {'max32':>9s} {'maxL1':>9s} {'r_med':>6s} {'r_max':>6s}  gate"]
    for q, n, m32, m1, x32, x1, p in rows:
        rm = m1 / m32 if m32 else (0.0 if m1 == 0 else np.inf)
        rx = x1 / x32 if x32 else (0.0 if x1 == 0 else np.inf)
        out.append(f"{q:12s} {n:7d} {m32:9.2e} {m1:9.2e} {x32:9.2e} {x1:9.2e} {rm:6.2f} {rx:6.2f}  {'PASS' if p else 'FAIL'}")
    return "\n".join(out)


if __name__ == "__main__":
    av = sys.argv[1:]
    tag = "l1"
    if "--l1" in av:
        i = av.index("--l1")
        tag = av[i + 1]
        del av[i:i + 2]
    run_dir = av[0] if av else os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    names = av[1:] or ["A", "B"]
    all_ok = True
    for nm in names:
        rows, ok, z = gate(run_dir, nm, tag)
        print(fmt(rows, nm))
        sub = z["l0f32"]["subnormals"]
        print(f"  L0-fp32 subnormal values: module outputs {sub[0]}/{sub[1]}, output grads {sub[2]}/{sub[3]}, "
              f"param grads {sub[4]};  L1 step time {float(z['l1']['time'][-1]):.1f} s")
        print(f"  step {nm}: {'PASS' if ok else 'FAIL'}\n")
        all_ok &= ok
    print("GATE L1 vs L0:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)
