#!/usr/bin/env python3
"""[S.2] reader -- SGWA's tuned RTE row, by [R.305]'s own reader and rules.

  env/bin/python scripts/s2_read.py

Selection cells (stages A/B/C, seed 41, n=1) are printed for the audit trail and
are NEVER quoted as a result.  The reported number is `r305_read.tuned`: the
confirmed candidate with the best mean over seeds 42-46.  It is compared with
the baselines' frozen [R.305]/[R.306] rows (results/sgwa/baselines_r305_rte.json)
through the 5/5 paired sign gate only.
"""
import json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import s2_plan as S2
S2.install()
import r305_plan as P
import r305_read as R


def main():
    res, man = P.load(), P.read_manifest()
    if not man:
        print("[s2] nothing planned yet")
        return
    rec = R.collect(res, man).get("sgwa")
    by_stage = {}
    for k, m in man.items():
        by_stage.setdefault(m["stage"], [0, 0])
        by_stage[m["stage"]][0] += 1
        by_stage[m["stage"]][1] += k in res
    print("=" * 84)
    print("[S.2] SGWA -- equal-effort search on RTE / roberta-base / q+v / k=256 / 30 epochs")
    print("cells done/planned by stage: " + "  ".join(f"{s} {d}/{n}" for s, (n, d) in sorted(by_stage.items())))
    print("=" * 84)
    if not rec:
        return
    plane = rec["plane"]
    pis = sorted({P._idx(k)[0] for k in plane})
    sjs = sorted({P._idx(k)[1] for k in plane})
    spec = S2.ARMS_S2["sgwa"]
    print("\nPLANE (seed 41, selection only):  rows P = lr*atom, cols scale gamma")
    print(f"{'P':>10s} " + "".join(f"{P.scale_at(spec, j):>10.4g}" for j in sjs))
    for i in pis:
        row = []
        for j in sjs:
            lab = next((k for k in plane if P._idx(k) == (i, j)), None)
            v = plane.get(lab)
            row.append(f"{v['acc']:10.4f}" + ("*" if lab == rec.get("plane_best") else " ")
                       if v else f"{'.':>10s} ")
        print(f"{P.p_at(spec, i):10.5f} " + "".join(r[:-1] + r[-1] for r in row))
    best = rec.get("plane_best")
    if best:
        ep = P.best_epoch(best)
        print(f"\nplane argmax {best}  acc {plane[best]['acc']:.4f}  argmax epoch {ep}"
              + ("  ⚠ TRUNCATED (last 10%)" if ep is not None and ep >= 27 else ""))
        print("edges after stage B:", rec["edges"] or "none -- bracketed on both axes")
        n_coll = sum(v["collapsed"] or v["near_floor"] for v in plane.values())
        print(f"collapsed / near-floor plane cells: {n_coll}/{len(plane)}")
    if rec["ofat"]:
        print("\nOFAT from the plane optimum (delta acc, n=1, selection only):")
        for k, d in sorted(rec["ofat"].items()):
            print(f"   {k:10s} {d:+.4f}")
    if rec["confirm"]:
        print("\nCONFIRMATION (seeds 42-46, out-of-sample):")
        for c, s in sorted(rec["confirm"].items()):
            v = [s[x] for x in P.CONFIRM_SEEDS if x in s]
            print(f"   {c:22s} n={len(v)}  mean {statistics.fmean(v):.4f}  per-seed "
                  + " ".join(f"{x:.4f}" for x in v))
    t = R.tuned(rec)
    if not t:
        return
    lab, mean, sd, n, vals = t
    args = man[lab]["args"]
    tk = args.split()
    lr = float(tk[tk.index("--learning_rate") + 1])
    sc = float(tk[tk.index("--sgwa_scaling") + 1])
    print(f"\n⭐ TUNED SGWA: {lab}  mean {mean:.4f} ± {sd:.4f}  (P = {lr*sc:.5g}, lr {lr:.6g}, "
          f"gamma {sc:.6g})\n   args: {args}")
    base = json.load(open(os.path.join(ROOT, "results", "sgwa", "baselines_r305_rte.json")))["arms"]
    print("\nAGAINST THE FROZEN [R.305]/[R.306] ROWS (paired over seeds 42-46):")
    rows = dict(base)
    rows["sgwa"] = {"mean": mean, "sd": sd, "per_seed": vals}
    for a in sorted(rows, key=lambda a: -rows[a]["mean"]):
        w, ti, nn = R.paired_gate(vals, rows[a]["per_seed"])
        tag = "" if a == "sgwa" else f"SGWA wins {w}/{nn} (ties {ti})"
        print(f"   {a:9s} {rows[a]['mean']:.4f} ± {rows[a]['sd']:.4f}   {tag}")


if __name__ == "__main__":
    main()
