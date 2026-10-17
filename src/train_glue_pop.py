#!/usr/bin/env python3
"""[S.4] population trainer -- R SGWA runs of ONE task in ONE process (S4-A/S4-B).

  env/bin/python src/train_glue_pop.py --task_name rte --runs_file runs.tsv [--mode merged]

runs.tsv: <label>\t<seed>\t<learning_rate>\t<sgwa_scaling>   (one run per line)

The training problem is train_glue.py's, constant for constant: roberta-base fp32,
batch 32 per run, 30 epochs, linear schedule with the task's warmup, AdamW
(betas .9/.999, eps 1e-8), head LR --classifier_lr, decoupled wd on adapter and
head, per-run global-norm clipping 1.0, dynamic padding, max_length 128, eval each
epoch, reported = MAX over epochs of the task's primary metric with its epoch.
Only the EXECUTION differs: one frozen backbone, one concatenated run-major batch,
R heads, one PopAdamW.  Per-run RNG: the run's seed drives its head init and its
data order; dropout draws from the process stream (per-element, so per-run anyway).
Output: one CSV per run in --out_dir (columns s3_read.py reads: the metric keys and
best_epoch), plus a population log.
"""
import argparse, csv, json, math, os, sys, time
from datetime import datetime

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TASK_KEYS = {"cola": ("sentence", None), "sst2": ("sentence", None), "mrpc": ("sentence1", "sentence2"),
             "qqp": ("question1", "question2"), "stsb": ("sentence1", "sentence2"),
             "mnli": ("premise", "hypothesis"), "qnli": ("question", "sentence"),
             "rte": ("sentence1", "sentence2"), "wnli": ("sentence1", "sentence2"),
             "boolq": ("question", "passage"), "cb": ("premise", "hypothesis")}
PRIMARY = {"cola": "matthews_correlation", "stsb": "pearson", "mrpc": "f1", "cb": "f1"}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--model_name_or_path", default="roberta-base")
    ap.add_argument("--runs_file", required=True)
    ap.add_argument("--mode", default="merged", choices=["merged", "stencil"])
    ap.add_argument("--chunks", type=int, default=1, help="split the population into this many "
                    "forward/backward chunks per step (exactly equivalent; trades memory for launches)")
    ap.add_argument("--per_run_batch", type=int, default=32)
    ap.add_argument("--eval_batch", type=int, default=32)
    ap.add_argument("--num_train_epochs", type=int, default=30)
    ap.add_argument("--num_warmup_steps", type=int, required=True)
    ap.add_argument("--classifier_lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clipping", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--sgwa_k", type=int, default=256)
    ap.add_argument("--sgwa_K", type=int, default=3)
    ap.add_argument("--sgwa_seed", type=int, default=777)
    ap.add_argument("--target_modules", default="query,value")
    ap.add_argument("--out_dir", default="scratchpad/s4/csv")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_train_samples", type=int, default=None)
    # [S.4b] branch-train-merge: this process is branch g of G (zero communication)
    ap.add_argument("--btm_G", type=int, default=1)
    ap.add_argument("--btm_branch", type=int, default=0)
    ap.add_argument("--btm_arm", default="disjoint", choices=["disjoint", "shared"],
                    help="disjoint: train only coefficients [g*k/G,(g+1)*k/G); shared: all k")
    ap.add_argument("--state_dir", default=None, help="save coef+head after every epoch here")
    ap.add_argument("--btm_sync_every", type=int, default=0, help="[S.4c] >0: torchrun ranks exchange their "
                    "coefficient blocks (exact concatenation) and average heads every this many steps and at "
                    "every epoch end (gloo, CPU tensors); rank = btm_branch; 0 = no communication (S.4b)")
    return ap.parse_args()


def load_runs(path):
    runs = []
    for line in open(path):
        if not line.strip() or line.startswith("#"):
            continue
        lab, seed, lr, sc = line.rstrip("\n").split("\t")
        runs.append(dict(label=lab, seed=int(seed), lr=float(lr), scaling=float(sc)))
    return runs


def main():
    a = parse()
    if a.btm_sync_every > 0:   # [S.4c] torchrun: rank = branch = device
        a.btm_branch = int(os.environ["LOCAL_RANK"]); a.device = f"cuda:{a.btm_branch}"
    runs = load_runs(a.runs_file)
    if a.btm_G > 1:
        for r_ in runs:
            r_["label"] += f"-b{a.btm_branch}"
    R = len(runs)
    assert R % a.chunks == 0
    dev = torch.device(a.device)
    os.makedirs(a.out_dir, exist_ok=True)
    log = open(os.path.join(os.path.dirname(a.out_dir.rstrip("/")) or ".", f"pop_{a.task_name}_{a.mode}_R{R}.log"), "a")

    def say(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    say(f"[pop] {datetime.utcnow().isoformat()}Z task={a.task_name} R={R} mode={a.mode} chunks={a.chunks} "
        f"device={dev}  runs={[r['label'] for r in runs]}")
    torch.manual_seed(runs[0]["seed"])

    from datasets import load_dataset
    import evaluate
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    from sgwa_parallel import PopAdamW, SGWAPopModel

    raw = load_dataset("super_glue" if a.task_name in ("boolq", "cb") else "glue", a.task_name)
    is_reg = a.task_name == "stsb"
    num_labels = 1 if is_reg else len(raw["train"].features["label"].names)
    tok = AutoTokenizer.from_pretrained(a.model_name_or_path)
    k1, k2 = TASK_KEYS[a.task_name]

    def prep(ex):
        texts = (ex[k1],) if k2 is None else (ex[k1], ex[k2])
        out = tok(*texts, padding=False, max_length=a.max_length, truncation=True)
        out["labels"] = ex["label"]
        return out

    proc = raw.map(prep, batched=True, remove_columns=raw["train"].column_names)
    train = proc["train"]
    if a.max_train_samples is not None and a.max_train_samples < len(train):
        train = train.shuffle(seed=runs[0]["seed"]).select(range(a.max_train_samples))
    evals = proc["validation_matched" if a.task_name == "mnli" else "validation"]
    if a.btm_G > 1:   # [S.4b] fixed data shard g of G, from the run's seed
        shard = torch.randperm(len(train), generator=torch.Generator().manual_seed(runs[0]["seed"]))
        train = train.select(shard[a.btm_branch::a.btm_G].tolist())
        say(f"[btm] branch {a.btm_branch}/{a.btm_G} arm={a.btm_arm}: {len(train)} train examples")
    tr_ids = [torch.tensor(x) for x in train["input_ids"]]
    tr_lab = torch.tensor(train["labels"], dtype=torch.float32 if is_reg else torch.long)
    ev_ids = [torch.tensor(x) for x in evals["input_ids"]]
    ev_lab = torch.tensor(evals["labels"], dtype=torch.float32 if is_reg else torch.long)
    Ntr, Nev = len(tr_ids), len(ev_ids)
    pad = tok.pad_token_id

    def collate(seqs):
        T = max(s.numel() for s in seqs)
        ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        am = torch.zeros(len(seqs), T, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :s.numel()] = s; am[i, :s.numel()] = 1
        return ids, am

    cfg = AutoConfig.from_pretrained(a.model_name_or_path, num_labels=num_labels)
    hf = AutoModelForSequenceClassification.from_pretrained(a.model_name_or_path, config=cfg)
    hf.to(dev)
    model = SGWAPopModel(hf, a.target_modules.split(","), R, k=a.sgwa_k, K=a.sgwa_K,
                         scaling=[r["scaling"] for r in runs], seed=a.sgwa_seed, mode=a.mode,
                         num_labels=num_labels, head_seeds=[r["seed"] for r in runs]).to(dev)
    del hf.classifier
    n_coef = sum(p.numel() for p in model.coef_params()) // R
    say(f"[pop] adapted modules {len(model.pops)}  coef params/run {n_coef}  head params/run "
        f"{sum(p.numel() for p in model.head.params()) // R}")

    mask = None
    if a.btm_G > 1 and a.btm_arm == "disjoint":   # [S.4b] mask: zero grad => zero Adam update => stays 0
        kb = a.sgwa_k // a.btm_G
        mask = torch.zeros(a.sgwa_k, device=dev)
        mask[a.btm_branch * kb:(a.btm_branch + 1) * kb] = 1.0
        for c_ in model.coef_params():
            c_.register_hook(lambda g, m_=mask: g * m_)
    dist = None
    sync_bytes = 0
    if a.btm_sync_every > 0:   # [S.4c]
        import torch.distributed as dist
        dist.init_process_group("gloo")
        assert dist.get_rank() == a.btm_branch and dist.get_world_size() == a.btm_G and mask is not None
        masks = [torch.zeros(a.sgwa_k) for _ in range(a.btm_G)]
        for g_ in range(a.btm_G):
            masks[g_][g_ * kb:(g_ + 1) * kb] = 1.0
        other = [None] * len(model.pops)   # the received blocks, restored after every optimiser step

        def sync():
            nonlocal sync_bytes
            t = time.time()
            with torch.no_grad():
                for i, p_ in enumerate(model.pops):
                    loc = p_.coef[0].detach().cpu()
                    got = [torch.zeros_like(loc) for _ in range(a.btm_G)]
                    dist.all_gather(got, loc)
                    new = sum(masks[g_] * got[g_] for g_ in range(a.btm_G))     # exact: disjoint supports
                    p_.coef[0].copy_(new.to(dev))
                    other[i] = (new * (1 - masks[a.btm_branch])).to(dev)
                    sync_bytes += loc.numel() * 4
                for t_ in model.head.params():
                    h = t_[0].detach().cpu()
                    dist.all_reduce(h); h /= a.btm_G
                    t_[0].copy_(h.to(dev))
                    sync_bytes += h.numel() * 4
            return time.time() - t

        def restore():   # keep the received blocks exactly as received between syncs (no decay, no update)
            with torch.no_grad():
                for i, p_ in enumerate(model.pops):
                    if other[i] is not None:
                        p_.coef[0].mul_(mask).add_(other[i])
    opt = PopAdamW([
        {"params": model.coef_params(), "lr": [r["lr"] for r in runs], "weight_decay": a.weight_decay},
        {"params": model.head.params(), "lr": [a.classifier_lr] * R, "weight_decay": a.weight_decay},
    ])
    steps_per_epoch = math.ceil(Ntr / a.per_run_batch)
    S = a.num_train_epochs * steps_per_epoch
    W = a.num_warmup_steps

    def lam(s):   # HF get_linear_schedule_with_warmup
        if s < W:
            return s / max(1, W)
        return max(0.0, (S - s) / max(1, S - W))

    gens = [torch.Generator().manual_seed(r["seed"]) for r in runs]
    metric = evaluate.load("super_glue" if a.task_name in ("boolq", "cb") else "glue", a.task_name)
    pkey = PRIMARY.get(a.task_name, "accuracy")
    best = [dict(val=float("-inf"), epoch=-1, metrics={}) for _ in runs]
    Rc = R // a.chunks
    rid_all = torch.arange(R, device=dev)
    t_start = time.time()
    step = 0
    torch.cuda.reset_peak_memory_stats(dev)
    for epoch in range(a.num_train_epochs):
        model.train()
        perms = [torch.randperm(Ntr, generator=g) for g in gens]
        t_ep = time.time()
        sync_s = 0.0
        for b in range(steps_per_epoch):
            opt.zero_grad()
            lo, hi = b * a.per_run_batch, min(Ntr, (b + 1) * a.per_run_batch)
            for c in range(a.chunks):
                rs = range(c * Rc, (c + 1) * Rc)
                idx = torch.cat([perms[r][lo:hi] for r in rs])
                ids, am = collate([tr_ids[i] for i in idx.tolist()])
                labels = tr_lab[idx].view(Rc, -1).to(dev)
                rid_seq = torch.arange(c * Rc, (c + 1) * Rc, device=dev).repeat_interleave(hi - lo)
                logits = _forward_chunk(model, ids.to(dev), am.to(dev), rid_seq, c, Rc, a.chunks)
                if is_reg:
                    loss = F.mse_loss(logits.squeeze(-1), labels, reduction="none").mean(1).sum()
                else:
                    loss = F.cross_entropy(logits.reshape(-1, num_labels), labels.reshape(-1),
                                           reduction="none").view(Rc, -1).mean(1).sum()
                loss.backward()
            opt.step(lam=lam(step), clip=a.grad_clipping)
            step += 1
            if dist is not None:
                restore()
                if step % a.btm_sync_every == 0 and b + 1 < steps_per_epoch:
                    sync_s += sync()
        if dist is not None:
            sync_s += sync()
        train_s = time.time() - t_ep
        # ---- eval: every run scores the same eval set ----
        model.eval()
        preds = [[] for _ in runs]
        with torch.no_grad():
            for lo in range(0, Nev, a.eval_batch):
                sl = list(range(lo, min(Nev, lo + a.eval_batch)))
                ids, am = collate([ev_ids[i] for i in sl])
                n = len(sl)
                ids = ids.repeat(R, 1).to(dev); am = am.repeat(R, 1).to(dev)
                rid_seq = rid_all.repeat_interleave(n)
                logits = model(ids, am, rid_seq)                     # (R, n, C)
                p = logits.squeeze(-1) if is_reg else logits.argmax(-1)
                for r in range(R):
                    preds[r].append(p[r].cpu())
        for r in range(R):
            pr = torch.cat(preds[r])
            md = metric.compute(predictions=pr.numpy(), references=ev_lab.numpy())
            v = md.get(pkey, float("-inf"))
            if v > best[r]["val"]:
                best[r] = dict(val=v, epoch=epoch, metrics=md)
        if a.state_dir:   # [S.4b] epoch-e state of every run
            os.makedirs(a.state_dir, exist_ok=True)
            for r, run in enumerate(runs):
                torch.save({"coef": torch.stack([p.coef[r].detach().cpu() for p in model.pops]),
                            "head": [t[r].detach().cpu() for t in model.head.params()],
                            "epoch": epoch, "scaling": run["scaling"], "seed": run["seed"]},
                           os.path.join(a.state_dir, f"{run['label']}_ep{epoch:02d}.pt"))
        say(f"[pop] epoch {epoch:2d}  train {train_s:6.1f}s  eval {time.time() - t_ep - train_s:5.1f}s  "
            f"peak {torch.cuda.max_memory_allocated(dev) / 2**20:,.0f} MiB  " +
            (f"sync {sync_s:.2f}s {sync_bytes / 2**20:.1f} MiB  " if dist is not None else "") +
            "  ".join(f"{runs[r]['label'].split('-')[-1]}:{best[r]['val']:.4f}" for r in range(R)))
    total = time.time() - t_start
    say(f"[pop] done in {total / 3600:.2f} h  ({total / R / 60:.1f} min per run-equivalent)")
    for r, run in enumerate(runs):
        row = dict(timestamp=datetime.utcnow().isoformat(), name=run["label"], task_name=a.task_name,
                   seed=run["seed"], lr=run["lr"], sgwa_scaling=run["scaling"], pop_R=R, pop_mode=a.mode,
                   pop_chunks=a.chunks, btm_G=a.btm_G, btm_branch=a.btm_branch, btm_arm=a.btm_arm,
                   btm_sync_every=a.btm_sync_every, sync_bytes=sync_bytes,
                   best_epoch=best[r]["epoch"], train_time_s=round(total, 1), steps_per_epoch=steps_per_epoch, n_train=Ntr,
                   peak_mem_mib=round(torch.cuda.max_memory_allocated(dev) / 2**20, 1))
        row.update({k_: float(v_) for k_, v_ in best[r]["metrics"].items()})
        with open(os.path.join(a.out_dir, f"{run['label']}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader(); w.writerow(row)
    say("[pop] ALL RUNS WRITTEN")


def _forward_chunk(model, ids, am, rid_seq, c, Rc, chunks):
    """Forward of one chunk of runs.  With chunks == 1 this is model.forward; with
    more, the heads of the chunk's runs are selected (coef rows are selected by
    rid inside the pop layers; merged mode needs the chunk's rows to be run-major
    over exactly its own runs, which the caller guarantees)."""
    if chunks == 1:
        return model(ids, am, rid_seq)
    for p in model.pops:
        p.set_rid(rid_seq, runs=torch.arange(c * Rc, (c + 1) * Rc, device=rid_seq.device))
    h = model.backbone(input_ids=ids, attention_mask=am).last_hidden_state[:, 0, :]
    head = model.head
    hh = h.view(Rc, -1, h.shape[-1])
    sl = slice(c * Rc, (c + 1) * Rc)
    hh = head.drop(hh)
    hh = torch.tanh(torch.bmm(hh, head.Wd[sl]) + head.bd[sl])
    hh = head.drop(hh)
    return torch.bmm(hh, head.Wo[sl]) + head.bo[sl]


if __name__ == "__main__":
    main()
