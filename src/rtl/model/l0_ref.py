"""L0 bridge: the real reference step, in PyTorch fp32 (cuda:0) and float64 (CPU, ground truth).

The model, data and optimizer are built by the code paths of src/train_glue.py (its parse_args on
the reference flags, its optimizer resolver and param groups, src/sgwa_adapter.get_sgwa_model);
the only change is ruling R2/D1a: eager attention with every dropout module replaced by one that
applies the chip's Threefry mask (model/rng.py).  Two steps are dumped (frozen before any run):

  A  the first optimizer step of training (s = 0): fresh coefficients, zero Adam state.  The linear
     schedule gives lr = 0 at s = 0, so A cannot exercise the merge (c = 0) or the update.
  B  step s = 140 (lr at its peak): the state after 140 real train_glue steps (stock dropout,
     batches 1..140 of the same seeded loader), then batch 141 with the chip's masks.
torch deterministic algorithms are on (warn_only) so the dumps are reproducible run to run.

  CUDA_VISIBLE_DEVICES is forced to 0.   python l0_ref.py [--out DIR]
Writes DIR/model_fp32.npz (frozen weights + LoWA tables, once) and DIR/step{A,B}_{l0f32,l0f64}.npz.
"""
import argparse
import copy
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"            # cuda:1 is forbidden
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)                                      # train_glue makes ./results at import (exists)

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
import torch.nn as nn                               # noqa: E402
from datasets import load_dataset                   # noqa: E402
from torch.utils.data import DataLoader             # noqa: E402
from transformers import (AutoConfig, AutoModelForSequenceClassification,  # noqa: E402
                          AutoTokenizer, DataCollatorWithPadding, get_scheduler)

import train_glue                                   # noqa: E402
from sgwa_adapter import SGWALinear, get_sgwa_model  # noqa: E402
import rng                                          # noqa: E402

SEED = 42                                           # first seed of the S.2 stage-D / S.3 protocol
REF_ARGV = ("--model_name_or_path roberta-base --task_name rte --dtype float32 --max_length 128 "
            "--adapter_target_modules query,value --per_device_train_batch_size 32 "
            "--num_train_epochs 30 --num_warmup_steps 140 --optimizer adamw-sgwa --sgwa_k 256 "
            "--sgwa_K 3 --sgwa_seed 777 --sgwa_materialise 1 --sgwa_target_modules query,value "
            "--sgwa_scaling 0.552427 --learning_rate 0.166667 --classifier_lr 5e-3 "
            "--weight_decay 0.01 --sgwa_sampled_grad 1").split()
STEPS = {"A": 0, "B": 140}
HEAD = [("dw", "classifier.dense.weight"), ("db", "classifier.dense.bias"),
        ("ow", "classifier.out_proj.weight"), ("ob", "classifier.out_proj.bias")]


def ref_args():
    sys.argv = ["train_glue.py"] + REF_ARGV
    return train_glue.parse_args()


def build(args, attn="eager"):
    """run_single_seed's seeding, model, adapter and loader, in its order (RNG-consuming steps)."""
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    raw = load_dataset("glue", args.task_name)
    num_labels = len(raw["train"].features["label"].names)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels,
                                        finetuning_task=args.task_name)
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config, attn_implementation=attn)
    tm = [m.strip() for m in args.adapter_target_modules.split(",")]
    model = get_sgwa_model(model=model, target_modules=tm, k=args.sgwa_k, K=args.sgwa_K,
                           scaling=args.sgwa_scaling, seed=args.sgwa_seed,
                           materialise=bool(args.sgwa_materialise),
                           sampled_grad=bool(args.sgwa_sampled_grad),
                           freeze_classifier_dense=args.freeze_classifier_dense)
    model.to("cuda")
    k1, k2 = train_glue.task_to_keys[args.task_name]

    def prep(ex):
        r = tok(ex[k1], ex[k2], padding=False, max_length=args.max_length, truncation=True)
        r["labels"] = ex["label"]
        return r
    ds = raw.map(prep, batched=True, remove_columns=raw["train"].column_names)["train"]
    loader = DataLoader(ds, shuffle=True, collate_fn=DataCollatorWithPadding(tok),
                        batch_size=args.per_device_train_batch_size)
    return model, loader


def make_opt(args, model, n_batches):
    """train_glue's classifier_lr param groups, resolver, kwargs and linear schedule."""
    cls = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" in n]
    ada = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" not in n]
    groups = [{"params": ada, "lr": args.learning_rate}, {"params": cls, "lr": args.classifier_lr}]
    opt = train_glue._resolve_optimizer_class(args.optimizer_base)(
        groups, lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = get_scheduler(args.lr_scheduler_type, optimizer=opt, num_warmup_steps=args.num_warmup_steps,
                          num_training_steps=args.num_train_epochs * n_batches)
    return opt, sched


class ChipDropout(nn.Module):
    """y = keep ? x * fp32(1/0.9) : 0 with the chip's mask for (site, layer); cls sites alternate."""

    def __init__(self, sites, layer, step, cache):
        super().__init__()
        self.sites, self.layer, self.step, self.cache, self.calls = sites, layer, step, cache, 0

    def forward(self, x):
        site = self.sites[self.calls % len(self.sites)]
        self.calls += 1
        key = (site, self.layer, tuple(x.shape))
        if key not in self.cache:
            self.cache[key] = torch.from_numpy(
                rng.keep_mask(tuple(x.shape), SEED, self.step, site, self.layer))
        keep = self.cache[key].to(x.device, x.dtype)
        return x * keep * torch.tensor(float(rng.DROP_SCALE), dtype=x.dtype, device=x.device)


def inject(model, step, cache):
    hf = model.model
    hf.roberta.embeddings.dropout = ChipDropout(["emb"], 0, step, cache)
    for l, lay in enumerate(hf.roberta.encoder.layer):
        lay.attention.self.dropout = ChipDropout(["attn"], l, step, cache)
        lay.attention.output.dropout = ChipDropout(["attn_out"], l, step, cache)
        lay.output.dropout = ChipDropout(["ffn_out"], l, step, cache)
    hf.classifier.dropout = ChipDropout(["cls1", "cls2"], 0, step, cache)


def lowa_modules(model):
    """the 24 SGWALinear modules in model.parameters() order: q0, v0, q1, v1, ..."""
    return [m for m in model.modules() if isinstance(m, SGWALinear)]


def head_params(model):
    d = dict(model.model.named_parameters())
    return [d[n] for _, n in HEAD]


def state_np(model, opt):
    """trainable params + Adam state as fp32 numpy, keyed like the dumps."""
    out = {}
    mods = lowa_modules(model)
    params = [m.coef for m in mods] + head_params(model)
    for tag, get in (("", lambda p: p.detach()),
                     ("m_", lambda p: opt.state[p]["exp_avg"] if p in opt.state else torch.zeros_like(p)),
                     ("v_", lambda p: opt.state[p]["exp_avg_sq"] if p in opt.state else torch.zeros_like(p))):
        coefs = [get(p) for p in params[:len(mods)]]
        out[tag + "coef"] = torch.stack(coefs).cpu().numpy()
        for (k, _), p in zip(HEAD, params[len(mods):]):
            out[tag + k] = get(p).detach().cpu().numpy().copy()
    return out


class SubnormalCounter:
    """FTZ question (spec 7.6): counts subnormal fp32 values in every module output and its gradient."""

    def __init__(self, model):
        self.n_out = self.n_grad = self.tot_out = self.tot_grad = 0
        self.hs = [m.register_forward_hook(self.fwd) for m in model.modules() if len(list(m.children())) == 0]

    @staticmethod
    def sub(t):
        a = t.detach().abs()
        return int(((a > 0) & (a < 2.0 ** -126)).sum()), t.numel()

    def fwd(self, mod, inp, out):
        if torch.is_tensor(out) and out.dtype == torch.float32:
            n, t = self.sub(out)
            self.n_out += n
            self.tot_out += t
            if out.requires_grad:
                out.register_hook(self.bwd)

    def bwd(self, g):
        n, t = self.sub(g)
        self.n_grad += n
        self.tot_grad += t

    def remove(self):
        for h in self.hs:
            h.remove()


def one_step(model, opt, batch, step, cache, count=False):
    """forward + backward + clip + AdamW, exactly train_glue's sequence; returns the dump dict."""
    model.train()
    inject(model, step, cache)
    ctr = SubnormalCounter(model) if count else None
    out = model(**batch)
    out.loss.backward()
    mods = lowa_modules(model)
    d = {"loss": out.loss.detach().cpu().numpy(), "logits": out.logits.detach().cpu().numpy(),
         "g_coef": torch.stack([m.coef.grad for m in mods]).cpu().numpy()}
    for (k, _), p in zip(HEAD, head_params(model)):
        d["g_" + k] = p.grad.detach().cpu().numpy().copy()
    if ctr:
        ctr.remove()
        sub_params = sum(SubnormalCounter.sub(p.grad)[0] for p in model.parameters() if p.grad is not None)
        d["subnormals"] = np.array([ctr.n_out, ctr.tot_out, ctr.n_grad, ctr.tot_grad, sub_params])
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    d["gnorm"] = gn.detach().cpu().numpy()
    d["clip"] = np.array(min(1.0, 1.0 / (float(gn) + 1e-6)))
    d["lr"] = np.array([g["lr"] for g in opt.param_groups])
    opt.step()
    for k, v in state_np(model, opt).items():
        d["post_" + k] = v
    opt.zero_grad()
    return d


def to_f64(model, opt, args, n_batches):
    """CPU float64 copy of model + optimizer (same state values), gamma = the fp32 run's fp32(gamma)."""
    m64 = copy.deepcopy(model).cpu().double()
    for mod in lowa_modules(m64):
        mod.scaling = float(np.float32(mod.scaling))
    o64, _ = make_opt(args, m64, n_batches)
    o64.load_state_dict(opt.state_dict())             # casts exp_avg / exp_avg_sq to float64
    for g64, g in zip(o64.param_groups, opt.param_groups):
        g64["lr"] = g["lr"]
    return m64, o64


def dump_model(model, path):
    sd = {k: v.detach().cpu().numpy() for k, v in model.model.state_dict().items()
          if v.dtype == torch.float32}
    for i, m in enumerate(lowa_modules(model)):
        sd[f"lowa.{i}.eflat"] = m.eflat.cpu().numpy()
        sd[f"lowa.{i}.hval"] = m.hval.cpu().numpy()
        sd[f"lowa.{i}.gamma"] = np.float32(m.scaling)
    np.savez(path, **sd)


def run(name, args, model, loader, opt, sched, batch, out_dir):
    s = STEPS[name]
    assert sched.last_epoch == s, (sched.last_epoch, s)
    pre = state_np(model, opt)
    inputs = {k: v.cpu().numpy() for k, v in batch.items()}
    meta = {"seed": np.array(SEED), "step_s": np.array(s), "n_batches": np.array(len(loader))}
    m64, o64 = to_f64(model, opt, args, len(loader))
    cache = {}
    d32 = one_step(model, opt, batch, s, cache, count=True)
    b64 = {k: v.cpu() for k, v in batch.items()}
    torch.set_num_threads(24)
    d64 = one_step(m64, o64, b64, s, cache)
    for tag, d in (("l0f32", d32), ("l0f64", d64)):
        np.savez(os.path.join(out_dir, f"step{name}_{tag}.npz"), **inputs, **meta, **d,
                 **{"pre_" + k: v for k, v in pre.items()})
    print(f"[L0] step {name}: s={s} L={batch['input_ids'].shape[1]} loss32={float(d32['loss']):.7f} "
          f"loss64={float(d64['loss']):.7f} gnorm={float(d32['gnorm']):.5f} lr={d32['lr']} "
          f"subnormals(out,grad,param)={d32['subnormals'][[0, 2, 4]]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False      # (train_glue sets these at import too)
    torch.backends.cudnn.allow_tf32 = False
    # reproducible dumps: CUDA index_add (the LoWA merge, which has colliding cells) otherwise sums
    # with atomics in a run-dependent order, so the 140-step state of B differed between runs.
    torch.use_deterministic_algorithms(True, warn_only=True)
    args = ref_args()

    # the eager build consumes the RNG exactly as train_glue's SDPA build: same head init
    m_sdpa, _ = build(args, attn="sdpa")
    m_eager, loader = build(args)
    for (_, n) in HEAD:
        assert torch.equal(dict(m_sdpa.model.named_parameters())[n], dict(m_eager.model.named_parameters())[n])
    del m_sdpa
    dump_model(m_eager, os.path.join(a.out, "model_fp32.npz"))

    # step A: the first batch of the first epoch, fresh state
    opt, sched = make_opt(args, m_eager, len(loader))
    batch = {k: v.to("cuda") for k, v in next(iter(loader)).items()}
    run("A", args, m_eager, loader, opt, sched, batch, a.out)
    del m_eager, opt

    # step B: 140 real steps (train_glue loop, stock dropout), then batch 141 with the chip's masks
    model, loader = build(args)
    opt, sched = make_opt(args, model, len(loader))
    model.train()
    done = 0
    while True:
        for batch in loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            if done == STEPS["B"]:
                run("B", args, model, loader, opt, sched, batch, a.out)
                return
            model(**batch).loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)
            opt.step()
            sched.step()
            opt.zero_grad()
            done += 1


if __name__ == "__main__":
    main()
