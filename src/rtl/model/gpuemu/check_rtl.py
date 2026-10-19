"""P9d: the GPU port against the RTL itself, on the result dumps of a finished end-to-end simulation.

  python gpuemu/check_rtl.py E2E_DIR [--layers 12 --batch 2 --seq 32 --steps 2]

E2E_DIR = a tb/top/run_e2e.py output directory (dump_1.hex, dump_2.hex of the 12-layer chip RTL, e.g.
build/e2e/l12_b2_s32_n2 of make test-e2e-full).  Rebuilds that run's inputs exactly as run_e2e.py does
(compile_all, the same per-step batches, step-2 pre-state = step-1 post-state), runs the chained steps
with gpuemu.step_gpu, decodes the RTL's 24 result tensors per step (check_compiler.results on the
dumped write ranges) and compares them bit for bit.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(RTL, "tb", "top"))
import run_e2e as E  # noqa: E402  (puts sw/ and model/ on the path)
import gpuemu.step_gpu as SG  # noqa: E402
import hwmodel.step as S  # noqa: E402
from gpuemu.check_step import gpu_outputs  # noqa: E402

CP, CC, isa = E.CP, E.CC, E.isa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("e2e_dir")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2)
    a = ap.parse_args()
    assert a.layers == 12
    c = CP.Cfg(a.layers, a.batch, a.seq)
    r = CP.compile_all(c)
    inp, s0 = r["inp"], int(r["inp"]["step_s"])
    full = dict(np.load(os.path.join(CP.RUNS, "stepB_l0f32.npz")))
    batches = [(full["input_ids"][k * c.B:(k + 1) * c.B, :c.L].astype(np.int64),
                full["attention_mask"][k * c.B:(k + 1) * c.B, :c.L].astype(np.uint8),
                full["labels"][k * c.B:(k + 1) * c.B]) for k in range(a.steps)]
    items = CP.program(c, CP.EPOCHS * int(inp["n_batches"]))
    wr_iv = []
    for name, f, _, _ in CP.flatten(items, c, None, s0):
        if isa.OPS[name][1] == isa.UNIT_SEQ:
            continue
        wr_iv += CP.ranges(name, f, c)[1]
    wregs = E.merge_regions(wr_iv, 100000, align=4)
    ndump = sum(hi - lo for lo, hi in wregs) // 4
    W = SG.GWeights(os.path.join(CP.RUNS, "model_fp32.npz"))
    cur = dict(inp)
    bad = 0
    for k in range(a.steps):
        d = dict(cur)
        d["input_ids"], d["attention_mask"], d["labels"] = batches[k]
        g, _ = gpu_outputs(W, d, S.WARMUP, S.EPOCHS * int(inp["n_batches"]))
        dump = E.read_hex(os.path.join(a.e2e_dir, f"dump_{k + 1}.hex"), ndump)
        mm = E.Mem()
        o = 0
        for lo, hi in wregs:
            mm.mem[lo // 4:hi // 4] = dump[o:o + (hi - lo) // 4]
            o += (hi - lo) // 4
        res = CC.results(c, mm)
        nb = E.cmp_tensors(f"step {k + 1} RTL vs GPU", res, g)
        nval = sum(np.asarray(res[n]).size for n in E.NAMES)
        print(f"step {k + 1} (s = {int(d['step_s'])}): {nval} values, loss RTL {float(res['loss']):.9g} "
              f"GPU {float(g['loss']):.9g}", flush=True)
        bad += nb
        cur = dict(d)
        cur["step_s"] = int(d["step_s"]) + 1
        for k2 in ["coef"] + E.HEAD:
            for t in ("", "m_", "v_"):
                cur["pre_" + t + k2] = g["post_" + t + k2]
    print("RTL-CHECK", "PASS" if bad == 0 else "FAIL", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
