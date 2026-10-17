#!/usr/bin/env python3
"""[S.2] SGWA's hyperparameter search on RTE / roberta-base, at EQUAL EFFORT to
the baselines' [R.305] search (and SCoRA's [R.306]).

  env/bin/python scripts/s2_plan.py --selftest
  env/bin/python scripts/s2_plan.py --generate {A|B|C|D}

Exactly the R.306 pattern: `r305_plan`'s stage machinery is reused VERBATIM
(this repo's copy is byte-identical to the one that ran [R.305]); only the arm
dict and the state dir are re-pointed.  Every baseline arm got:

  A  the 5 x 4 plane in (P = lr*atom) x scale, ratio 2 on both axes, seed 41
  B  the coded edge-extension rule: <= 2 rounds, <= 40 cells, full rows/cols
  C  the single-knob block from the arm's OWN plane optimum: 4 shared knobs
     (wd0, cosine, clf1e-3, clf1e-2) + 1-6 arm-specific ones
  D  the top-2 of A u B u C, re-run at the 5 disjoint seeds 42-46
and the reported number is the confirmed candidate with the best 5-seed mean
(`r305_read.tuned`).  SGWA gets the same, with no extra freedom.

THE ONLY ARM-SPECIFIC INPUTS, fixed before any cell (prereg S2_sgwa_rte_prereg.md)
  atom(scale) = scale        SGWA's interior atom IS gamma (verify_sgwa G6), the
                             same identity LoCA and LYRA carry ([R.281]).
  p0  = 0.5 * 50 / sqrt(2mn) = 0.0230179   The baselines centred P on their own
        [R.237] screening argmax; SGWA has no screening history, so the centre
        is the a-priori rule [S.1] already used -- FourierFT's [R.305]-selected
        P, which is also the median P of the seven baseline rows.  Stage B moves
        the ladder if the argmax lands on an edge, under the baselines' rule.
  s0  = 150 / sqrt(2mn) = 0.138107, s0_at_bottom=False -> ladder
        [s0/4, s0/2, s0, 2 s0]: the derived scale is the 3rd rung, INTERIOR,
        the placement [R.306] gave SCoRA's derived scale.
  ofat (arm-specific) = SGWA's one structural knob, the Chebyshev order K, one
        step each way: K=2 (13-vertex atom) and K=4 (41-vertex atom).  Baselines
        got 1-6 arm-specific knobs (mean 2.0); 2 is inside that range.
  The [S.1] configuration (lr 1/6, gamma 0.138107, K 3) is plane cell p0-s2.
"""
import argparse, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import r305_plan as P                      # every helper, verbatim -- no forks

D = os.path.join(ROOT, "scratchpad", "s2")

SGWA_ATOM = 150.0 / P.SQRT_2MN                       # 0.138106793200498
P0 = 5e-1 * (50.0 / P.SQRT_2MN)                      # FourierFT's selected P


def _atom_sgwa(s):
    return s


ARMS_S2 = {
    "sgwa": dict(
        title="SGWA (spectral graph wavelet adapter)",
        base=("--optimizer adamw-sgwa --sgwa_k 256 --sgwa_K 3 --sgwa_seed 777"
              " --sgwa_materialise 1 --sgwa_target_modules query,value"),
        scale_flag="--sgwa_scaling",
        atom=_atom_sgwa,
        p0=P0,
        s0=SGWA_ATOM, s0_at_bottom=False,
        extra="",
        ofat={"K2": "--sgwa_K 2", "K4": "--sgwa_K 4"},
    ),
}


def install():
    """Point r305_plan's stage machinery at THIS arm and THIS state dir."""
    P.ARMS = ARMS_S2
    P.D = D
    os.makedirs(D, exist_ok=True)


def generate(stage):
    install()
    return P.generate(stage)


def selftest():
    ok = fail = 0

    def ck(cond, msg):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL: {msg}")

    import hashlib, importlib
    ref = "/workspace/lora_research_signal/scripts/r305_plan.py"
    if os.path.exists(ref):
        h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
        ck(h(ref) == h(os.path.join(HERE, "r305_plan.py")),
           "r305_plan.py must be byte-identical to the copy that ran [R.305]")
    install()
    spec = ARMS_S2["sgwa"]
    # -- 1. the same plane as every baseline ---------------------------------
    man = {}
    jobs = P.stage_a(man)
    ck(len(jobs) == 20 == len(P.P_INDICES_A) * len(P.S_INDICES_A),
       f"SGWA must get the baselines' 20-cell plane, got {len(jobs)}")
    ck(P.P_INDICES_A == [-2, -1, 0, 1, 2] and P.S_INDICES_A == [0, 1, 2, 3],
       "ladder shape must be [R.305]'s 5 x 4, ratio 2")
    ck(all(seed == 41 for _l, _a, seed in jobs), "plane cells run at the selection seed 41")
    # -- 2. the derived scale is interior; the S.1 config is a plane cell ----
    scales = [P.scale_at(spec, j) for j in P.S_INDICES_A]
    ck(min(scales) < SGWA_ATOM < max(scales), f"derived scale must be interior to {scales}")
    s1 = [a for l, a, _s in jobs if l == "sgwa-A-p0-s2"][0]
    toks = s1.split()
    ck(abs(float(toks[toks.index("--learning_rate") + 1]) - 1 / 6) < 1e-5
       and abs(float(toks[toks.index("--sgwa_scaling") + 1]) - SGWA_ATOM) < 1e-6,
       "the [S.1] configuration (lr 1/6, gamma 0.138107) must be plane cell p0-s2")
    # -- 3. lr * atom == P in every cell ------------------------------------
    bad = []
    for lab, args, _seed in jobs:
        t = args.split()
        lr = float(t[t.index("--learning_rate") + 1])
        sc = float(t[t.index("--sgwa_scaling") + 1])
        i = int(lab.split("-p")[1].split("-s")[0])
        if abs(lr * _atom_sgwa(sc) - P.p_at(spec, i)) > 1e-5 * P.p_at(spec, i):
            bad.append(lab)
    ck(not bad, f"lr*atom must equal P everywhere; violated in {bad[:3]}")
    ck(abs(P0 - 0.0230179) < 1e-6, f"P centre must be FourierFT's selected P, got {P0}")
    # -- 4. the shared protocol is the baselines', untouched -----------------
    ck("--num_warmup_steps 140" in P.COMMON and "--num_train_epochs 30" in P.COMMON
       and "--task_name rte" in P.COMMON, "COMMON must be [R.305]'s RTE protocol")
    ck(all(f"--classifier_lr {P.C_CLF} --weight_decay {P.C_WD}" in a for _l, a, _s in jobs),
       "head LR 5e-3 and wd 0.01 in every plane cell, as for every baseline")
    # -- 5. stage C: the 4 shared knobs + exactly the 2 declared ones --------
    fake = {l: {"acc": 0.6 + 0.001 * k, "collapsed": False, "near_floor": False}
            for k, (l, _a, _s) in enumerate(jobs)}
    man2 = dict(man)
    cj = P.stage_c(fake, man2)
    names = sorted(l.split("-C-")[1] for l, _a, _s in cj)
    ck(names == sorted(list(P.SHARED_OFAT) + ["K2", "K4"]),
       f"stage C must be shared OFAT + K2/K4, got {names}")
    best = P.argmax_cell(fake)
    ck(all(man2[l]["from"] == best for l, _a, _s in cj),
       "every OFAT cell departs from SGWA's OWN plane optimum")
    k2 = [a for l, a, _s in cj if l.endswith("-K2")][0]
    ck(k2.split().count("--sgwa_K") == 1 and "--sgwa_K 2" in k2,
       "an OFAT cell moves exactly one knob (no duplicated flag)")
    # -- 6. stage D: top-2 x 5 disjoint seeds --------------------------------
    for l, a, s in cj:
        fake[l] = {"acc": 0.5, "collapsed": False, "near_floor": False}
    dj = P.stage_d(fake, man2)
    ck(len(dj) == 10 and sorted({s for _l, _a, s in dj}) == [42, 43, 44, 45, 46],
       f"stage D must be top-2 x seeds 42-46 (10 cells), got {len(dj)}")
    ck(not any("fftstock" in l for l, _a, _s in dj), "no foreign FourierFT cells charged to SGWA")
    # -- 7. stage B: an edge argmax extends by the baselines' rule -----------
    edge = {l: {"acc": (0.9 if l == "sgwa-A-p2-s3" else 0.6), "collapsed": False,
                "near_floor": False} for l, _a, _s in jobs}
    man3 = dict(man)
    bj = P.stage_b(edge, man3, 0)
    ck(len(bj) == 4 + 5 + 1, f"a corner argmax must add one full row + one full column "
                             f"(4 + 5 + 1 = 10 cells), got {len(bj)}")
    ck(P.MAX_EXTENSION_ROUNDS == 2 and P.MAX_EXTENSION_CELLS == 40,
       "stage B caps must be the baselines' (2 rounds, 40 cells)")
    print(f"[s2] selftest: {ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--generate", choices=["A", "B", "C", "D"])
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.generate:
        generate(a.generate)
