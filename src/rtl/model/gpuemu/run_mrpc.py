"""P9d: MRPC fine-tuning in which every training step is the chip's step (L1 arithmetic, on the GPU).

  CUDA_VISIBLE_DEVICES=0 python gpuemu/run_mrpc.py --seed 42 --out DIR [--ckpt-steps 0,114,...]

Run = the paper's MRPC LoWA row (scratchpad/s3/jobs_mrpc.tsv, mrpc-sgwa-seedN): roberta-base, LoWA on
query/value, k 256, K 3, gamma 0.552427, lr 0.166667 / head 5e-3, wd 0.01, B 32, 30 epochs, warmup
206, clipping 1.0, AdamW (0.9, 0.999, 1e-8).  What the chip changes (spec.md D1-D6): every step is
gpuemu.step_gpu.step = hwmodel/step.py bit for bit (fp32 RNE with FTZ/DAZ, the chip's reduction
orders, SFU exp/recip/rsqrt/sqrt/tanh/erf/ln, Threefry dropout keyed by (seed, step, site, layer),
sampled coefficient gradient, GNORM/clip and AdamW and the step scalars in fp32 on chip), on the
chip's fixed program shape B 32 x L 128 (every batch padded to 128 tokens; the short last batch of
an epoch, 20 rows, padded with dummy rows of loss weight 0, loss / B_ACT).
Model init (classifier head), LoWA tables and the data order come from train_glue's own code path
(l0_ref.build: same seeding, model, adapter and seeded shuffling loader).
Evaluation after every epoch on the 408 dev pairs, two ways: (a) the chip's forward arithmetic,
dropout off (gpuemu.step_gpu.forward_eval); (b) ordinary fp32 PyTorch (the HF model of train_glue,
SDPA, eval mode) with the same trained coefficients and head.  Both are logged; best epoch by F1.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
CWD0 = os.getcwd()
import l0_ref  # noqa: E402  (chdir to the repo root, imports train_glue)
import gpuemu.step_gpu as SG  # noqa: E402
import hwmodel.step as S  # noqa: E402

f32 = np.float32
HEAD = S.HEAD
L_PAD, B_FIX = 128, 32
ARGV = ("--model_name_or_path roberta-base --task_name mrpc --dtype float32 --max_length 128 "
        "--adapter_target_modules query,value --per_device_train_batch_size 32 --num_train_epochs 30 "
        "--num_warmup_steps 206 --optimizer adamw-sgwa --sgwa_k 256 --sgwa_K 3 --sgwa_seed 777 "
        "--sgwa_materialise 1 --sgwa_target_modules query,value --sgwa_scaling 0.552427 "
        "--learning_rate 0.166667 --classifier_lr 5e-3 --weight_decay 0.01 --sgwa_sampled_grad 1").split()


def pad_batch(b):
    """-> ids [32, 128], mask [32, 128] uint8, labels [32], loss_w [32], b_act (spec.md D3)."""
    ids, am, y = b["input_ids"], b["attention_mask"], b["labels"]
    n, L = ids.shape
    assert n <= B_FIX and L <= L_PAD
    I = torch.ones((B_FIX, L_PAD), dtype=torch.int64)          # pad token 1
    M = torch.zeros((B_FIX, L_PAD), dtype=torch.uint8)
    I[:n, :L], M[:n, :L] = ids, am.to(torch.uint8)
    I[n:, 0], M[n:, 0] = 0, 1                                  # dummy rows: <s> + pads, mask [1, 0, ...]
    Y = np.zeros(B_FIX, np.int64)
    Y[:n] = y.numpy()
    w = np.zeros(B_FIX, f32)
    w[:n] = 1.0
    return I, M, Y, w, n


def metrics(pred, ref):
    pred, ref = np.asarray(pred), np.asarray(ref)
    tp = int(((pred == 1) & (ref == 1)).sum())
    fp = int(((pred == 1) & (ref == 0)).sum())
    fn = int(((pred == 0) & (ref == 1)).sum())
    return dict(accuracy=float((pred == ref).mean()), f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop early (timing / smoke)")
    ap.add_argument("--ckpt-steps", default="0,1,113,114,115,205,206,207,1000,2000,3449")
    a = ap.parse_args()
    out = os.path.abspath(os.path.join(CWD0, a.out))
    os.makedirs(out, exist_ok=True)
    ck_steps = {int(x) for x in a.ckpt_steps.split(",") if x}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # --- train_glue's model / adapter / data, seeded as run_single_seed does
    l0_ref.SEED = a.seed
    sys.argv = ["train_glue.py"] + ARGV
    args = l0_ref.train_glue.parse_args()
    model, loader = l0_ref.build(args, attn="sdpa")
    nb = len(loader)
    t_total, warmup = args.num_train_epochs * nb, args.num_warmup_steps
    assert nb == 115 and t_total == 3450 and warmup == 206, (nb, t_total, warmup)
    mods = l0_ref.lowa_modules(model)
    W = SG.GWeights(os.path.join(MODEL, "runs", "model_fp32.npz"))
    # the frozen backbone and LoWA tables of model_fp32.npz are this model's (pretrained roberta-base,
    # sgwa_seed 777): check every tensor bit for bit
    sd = {k: v for k, v in model.model.state_dict().items()}
    nchk = 0
    for k, v in W.z.items():
        assert torch.equal(sd[k].to(v.device), v), k
        nchk += 1
    for i, m in enumerate(mods):
        assert torch.equal(m.eflat.cpu(), W.lw[i]["plan"].eflat.cpu()), i
        assert torch.equal(m.hval.cpu().float(), W.lw[i]["h"].cpu()), i
        assert f32(m.scaling) == S.GAMMA, (m.scaling, S.GAMMA)
    hp = dict(model.model.named_parameters())
    head0 = {k: hp[n].detach().cpu().numpy() for k, n in l0_ref.HEAD}
    coef0 = torch.stack([m.coef.detach() for m in mods]).cpu().numpy()
    assert not coef0.any()
    print(f"[init] seed {a.seed}: {nchk} backbone tensors equal to model_fp32.npz; head init from train_glue's "
          f"build; {nb} batches/epoch, T {t_total}, warmup {warmup}", flush=True)

    # eval data: train_glue's validation split, same tokenisation
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, DataCollatorWithPadding
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    raw = load_dataset("glue", "mrpc")["validation"]
    k1, k2 = l0_ref.train_glue.task_to_keys["mrpc"]

    def prep(ex):
        r = tok(ex[k1], ex[k2], padding=False, max_length=128, truncation=True)
        r["labels"] = ex["label"]
        return r
    dev = raw.map(prep, batched=True, remove_columns=raw.column_names)
    dev_loader = DataLoader(dev, collate_fn=DataCollatorWithPadding(tok), batch_size=32)
    dev_batches = [pad_batch(b) for b in dev_loader]
    dev_hf = [{k: v.to("cuda") for k, v in b.items()} for b in dev_loader]

    st = SG.State(coef0, head0)
    s = 0
    ep0 = 0
    log = open(os.path.join(out, "log.jsonl"), "a")
    # resume from the last finished epoch (state + the torch RNG that drives the loader's shuffle)
    done = sorted(int(f[8:10]) for f in os.listdir(out) if f.startswith("state_ep") and f.endswith(".npz"))
    if done:
        z = dict(np.load(os.path.join(out, f"state_ep{done[-1]:02d}.npz")))
        st = SG.State(z["coef"], {k: z[k] for k in HEAD}, z["m_coef"],
                      np.concatenate([z["m_" + k].reshape(-1) for k in HEAD]), z["v_coef"],
                      np.concatenate([z["v_" + k].reshape(-1) for k in HEAD]), z["b1t"], z["b2t"])
        torch.set_rng_state(torch.as_tensor(z["torch_rng"]))
        ep0, s = done[-1] + 1, int(z["s"])
        print(f"[resume] from epoch {done[-1]} (s = {s})", flush=True)

    t_start = time.time()
    for ep in range(ep0, a.epochs):
        tl, losses = time.time(), []
        for b in loader:
            I, M, Y, w, n = pad_batch(b)
            if s in ck_steps:
                pre = st.to_np()
                np.savez(os.path.join(out, f"ckpt_s{s:04d}.npz"), input_ids=I.numpy(), attention_mask=M.numpy(),
                         labels=Y, loss_w=w, b_act=np.array(n), step_s=np.array(s), seed=np.array(a.seed),
                         warmup=np.array(warmup), t_total=np.array(t_total), n_batches=np.array(nb),
                         **{"pre_" + k: v for k, v in pre.items() if k not in ("b1t", "b2t")})
            d = SG.step(W, st, I, M, Y, w, n, s, a.seed, warmup, t_total)
            if s in ck_steps:
                post = st.to_np()
                np.savez(os.path.join(out, f"ckpt_s{s:04d}_gpu.npz"), loss=d["loss"].cpu().numpy(),
                         logits=d["logits"].cpu().numpy(), g_coef=d["g_coef"].cpu().numpy(),
                         gnorm=d["gnorm"].cpu().numpy(), clip=d["clip"].cpu().numpy(), lr=np.array(d["lr"], f32),
                         **{"g_" + k: d["g_" + k].cpu().numpy() for k in HEAD},
                         **{"post_" + k: v for k, v in post.items() if k not in ("b1t", "b2t")})
            losses.append(d["loss"])
            s += 1
            if a.max_steps > 0 and s >= a.max_steps:
                break
        lv = torch.stack(losses).cpu().numpy()
        assert np.isfinite(lv).all(), "non-finite loss"
        ttrain = time.time() - tl
        # ---- evaluation: chip forward arithmetic, and fp32 PyTorch
        te = time.time()
        pc, refs = [], []
        for I, M, Y, w, n in dev_batches:
            z = SG.forward_eval(W, st, I, M)
            pc.append(z[:n].argmax(-1).cpu().numpy())
            refs.append(Y[:n])
        pc, refs = np.concatenate(pc), np.concatenate(refs)
        with torch.no_grad():
            for i, m in enumerate(mods):
                m.coef.copy_(st.coef[i])
            for k, nme in l0_ref.HEAD:
                hp[nme].copy_(st.head[k])
            model.eval()
            pt = np.concatenate([model(**b).logits.argmax(-1).cpu().numpy() for b in dev_hf])
        mc, mt = metrics(pc, refs), metrics(pt, refs)
        rec = dict(seed=a.seed, epoch=ep, steps=s, train_loss_mean=float(lv.mean()), last_loss=float(lv[-1]),
                   chip_eval=mc, fp32_eval=mt, pred_agree=float((pc == pt).mean()), t_train=ttrain,
                   t_eval=time.time() - te, s_per_step=ttrain / max(1, len(lv)))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[ep {ep:2d}] s={s} loss {lv.mean():.4f} chip acc {mc['accuracy']:.4f} f1 {mc['f1']:.4f} | "
              f"fp32 acc {mt['accuracy']:.4f} f1 {mt['f1']:.4f} | agree {rec['pred_agree']:.4f} | "
              f"{rec['s_per_step']:.2f} s/step, elapsed {(time.time() - t_start) / 60:.1f} min", flush=True)
        np.savez(os.path.join(out, f"state_ep{ep:02d}.npz"), s=np.array(s), torch_rng=torch.get_rng_state().numpy(),
                 **st.to_np())
        if a.max_steps > 0 and s >= a.max_steps:
            break
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
