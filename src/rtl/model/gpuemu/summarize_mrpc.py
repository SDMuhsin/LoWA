"""P9d: summary of the chip-exact MRPC runs (run_mrpc.py log.jsonl), the paper's protocol.

  python gpuemu/summarize_mrpc.py DIR [DIR ...]      (one run directory per seed)

Per seed: the best epoch = first epoch with the maximum dev F1 (train_glue: strictly greater replaces),
accuracy and F1 at that epoch; median over seeds.  Selection and scores are given for the chip-forward
evaluation and, separately, for the fp32 PyTorch evaluation of the same trained parameters.
"""
import json
import os
import statistics
import sys


def best(recs, key):
    b = None
    for r in recs:
        if b is None or r[key]["f1"] > b[key]["f1"]:
            b = r
    return b


def main():
    rows = []
    for d in sys.argv[1:]:
        recs = {}
        for ln in open(os.path.join(d, "log.jsonl")):
            r = json.loads(ln)
            recs[r["epoch"]] = r                      # a resumed epoch replaces its earlier record
        recs = [recs[e] for e in sorted(recs)]
        bc, bf = best(recs, "chip_eval"), best(recs, "fp32_eval")
        agree = min(r["pred_agree"] for r in recs)
        rows.append(dict(seed=recs[0]["seed"], epochs=len(recs), ep_c=bc["epoch"], acc_c=bc["chip_eval"]["accuracy"],
                         f1_c=bc["chip_eval"]["f1"], ep_f=bf["epoch"], acc_f=bf["fp32_eval"]["accuracy"],
                         f1_f=bf["fp32_eval"]["f1"], agree=agree, sps=statistics.median(r["s_per_step"] for r in recs)))
    print("| seed | epochs done | best epoch | acc | F1 | (fp32 eval) best epoch | acc | F1 | min pred. agreement |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['seed']} | {r['epochs']} | {r['ep_c']} | {100 * r['acc_c']:.2f} | {100 * r['f1_c']:.2f} | "
              f"{r['ep_f']} | {100 * r['acc_f']:.2f} | {100 * r['f1_f']:.2f} | {r['agree']:.4f} |")
    if rows:
        med = lambda k: 100 * statistics.median(r[k] for r in rows)  # noqa: E731
        print(f"| **median** | | | **{med('acc_c'):.2f}** | **{med('f1_c'):.2f}** | | {med('acc_f'):.2f} | "
              f"{med('f1_f'):.2f} | |")


if __name__ == "__main__":
    main()
