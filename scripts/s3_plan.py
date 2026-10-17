#!/usr/bin/env python3
"""[S.3] SGWA's [R.310] multi-task row: the RTE-selected configurations of [S.2]
carried to the paper's other six RoBERTa columns, exactly as the baselines' were.

  env/bin/python scripts/s3_plan.py      # writes scratchpad/s3/jobs_<task>.tsv

The per-task flags (task, 30 epochs, batch 32, the task's own derived warmup
ratio, fp32, q+v) are COPIED from [R.310]'s own job files, never re-derived, so
SGWA's cells differ from the baselines' only in the adapter flags.  Both [S.2]
rows run (neither replaces the other): at d=768 carrying P is carrying lr.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R310 = "/workspace/lora_research_signal/scratchpad/phaseR/r310"
D = os.path.join(ROOT, "scratchpad", "s3")
TASKS = ["cb", "cola", "mrpc", "stsb", "boolq", "sst2"]      # cheapest first
SEEDS = [42, 43, 44, 45, 46]
SGWA = ("--optimizer adamw-sgwa --sgwa_k 256 --sgwa_K 3 --sgwa_seed 777 --sgwa_materialise 1"
        " --sgwa_target_modules query,value")
CONFIGS = {   # [S.2] §2 and §7.1; P = lr * gamma = 0.0921 for both
    "sgwa":  "--sgwa_scaling 0.552427 --learning_rate 0.166667",   # full search (A u B u C)
    "sgwaA": "--sgwa_scaling 0.138107 --learning_rate 0.666667",   # grid-only (A)
}
TAIL = "--classifier_lr 5e-3 --weight_decay 0.01"


def common(task):
    line = open(os.path.join(R310, f"jobs_{task}.tsv")).readline().split("\t")[2]
    c = line.split(" --optimizer")[0]
    assert f"--task_name {task} " in c and "--num_train_epochs 30" in c
    return c


if __name__ == "__main__":
    os.makedirs(D, exist_ok=True)
    for t in TASKS:
        rows = [f"{t}-{arm}-seed{s}\t{s}\t{common(t)} {SGWA} {cfg} {TAIL}"
                for arm, cfg in CONFIGS.items() for s in SEEDS]
        with open(os.path.join(D, f"jobs_{t}.tsv"), "w") as f:
            f.write("\n".join(rows) + "\n")
        print(t, len(rows), "cells")
