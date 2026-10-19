"""P9d: whole-step bit-exactness of the GPU port against L1.

  python gpuemu/check_step.py dumps            the P1 steps A (s = 0) and B (s = 140) of runs/:
                                               GPU step vs the stored L1 results step{A,B}_l1.npz
  python gpuemu/check_step.py ckpt FILE...     MRPC training checkpoints written by run_mrpc.py
                                               (state before step s + the batch): run L1
                                               (hwmodel/step.py) and the GPU step on them, compare
Every output of the step is compared bit for bit: loss, logits, coefficient and head gradients,
total norm, clip factor, learning rates, and the post-step coefficients, head, Adam m and v.
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
import gpuemu.step_gpu as SG  # noqa: E402
import hwmodel as H  # noqa: E402
import hwmodel.step as S  # noqa: E402

f32 = np.float32
HEAD = S.HEAD


def bits(a):
    return np.ascontiguousarray(np.asarray(a, np.float32)).view(np.uint32).reshape(-1)


def compare(tag, l1, gp):
    """l1: dict of L1 outputs; gp: dict of GPU outputs (numpy).  Returns number of mismatching values."""
    tot = bad = 0
    worst = []
    for k in sorted(l1):
        if k == "time" or k not in gp:
            continue
        a, b = bits(l1[k]), bits(gp[k])
        assert a.size == b.size, (k, a.size, b.size)
        nb = int((a != b).sum())
        tot += a.size
        bad += nb
        if nb:
            worst.append(f"{k}:{nb}/{a.size}")
    print(f"[{tag}] {tot} values over {len([k for k in l1 if k in gp])} outputs, {bad} mismatches "
          + (" ".join(worst) if worst else "(bit-exact)"), flush=True)
    return bad


def gpu_outputs(W, inp, warmup, t_total):
    s = int(inp["step_s"])
    b1t = b2t = f32(1.0)
    for _ in range(s):
        b1t, b2t = H.mul(b1t, S.BETA1)[0], H.mul(b2t, S.BETA2)[0]
    st = SG.State(inp["pre_coef"], {k: inp["pre_" + k] for k in HEAD}, inp["pre_m_coef"],
                  np.concatenate([inp["pre_m_" + k].reshape(-1) for k in HEAD]), inp["pre_v_coef"],
                  np.concatenate([inp["pre_v_" + k].reshape(-1) for k in HEAD]), b1t, b2t)
    Bn = inp["input_ids"].shape[0]
    lw = inp["loss_w"] if "loss_w" in inp else np.ones(Bn, f32)
    b_act = int(inp["b_act"]) if "b_act" in inp else Bn
    torch.cuda.synchronize()
    t0 = time.time()
    d = SG.step(W, st, torch.as_tensor(inp["input_ids"]), torch.as_tensor(inp["attention_mask"]),
                inp["labels"], lw, b_act, s, int(inp["seed"]), warmup, t_total)
    torch.cuda.synchronize()
    dt = time.time() - t0
    out = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in d.items()}
    out["lr"] = np.array(d["lr"], f32)
    for k, v in st.to_np().items():
        out["post_" + k] = v
    return out, dt


def main():
    mode = sys.argv[1]
    W = SG.GWeights(os.path.join(MODEL, "runs", "model_fp32.npz"))
    bad = 0
    if mode == "dumps":
        for name in "AB":
            inp = dict(np.load(os.path.join(MODEL, "runs", f"step{name}_l0f32.npz")))
            l1 = dict(np.load(os.path.join(MODEL, "runs", f"step{name}_l1.npz")))
            out, dt = gpu_outputs(W, inp, S.WARMUP, S.EPOCHS * int(inp["n_batches"]))
            bad += compare(f"step {name} s={int(inp['step_s'])} GPU {dt:.2f} s, L1 {l1['time'][-1]:.1f} s", l1, out)
    else:
        HW = S.Weights(os.path.join(MODEL, "runs", "model_fp32.npz"))
        for path in sys.argv[2:]:
            inp = dict(np.load(path))
            hd = {k: inp["pre_" + k] for k in HEAD}
            HW.z.update({"classifier." + k: v for k, v in hd.items()})
            t0 = time.time()
            l1 = S.step(HW, inp, verbose=False)
            tl1 = time.time() - t0
            out, dt = gpu_outputs(W, inp, int(inp["warmup"]), int(inp["t_total"]))
            bad += compare(f"{os.path.basename(path)} s={int(inp['step_s'])} GPU {dt:.2f} s, L1 {tl1:.1f} s", l1, out)
            run = path.replace(".npz", "_gpu.npz")
            if os.path.exists(run):          # the outputs the training run itself produced at this step
                bad += compare(f"{os.path.basename(run)} (in-run GPU step) vs L1", l1, dict(np.load(run)))
    print("STEP-CHECK", "PASS" if bad == 0 else "FAIL", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
