#!/usr/bin/env python3
"""P7a end-to-end tests: the compiled program on the whole chip (hw/top/top_chip: sequencer + gm_top + vu_top + lw_top
+ mm_top with the on-chip dropout-mask generators) + the DRAM model (tb/mem/tb_dram.sv), driven through the chip's
AXI4-Lite CSR slave by a host model (tb/top/tb_top.sv).

  python3 tb/top/run_e2e.py --layers 1 --batch 2 --seq 32 [--steps 2]          (make test-e2e)
  python3 tb/top/run_e2e.py --layers 12 --batch 2 --seq 32 --steps 2 --l1      (make test-e2e-full)

Golden values come only from the interpreter (sw/interp.py = sequencer decode + L1 per-op entry points) and, with
--l1, from the L1 step (model/hwmodel/step.py) at the same (B, L), chained for step 2 (pre-state = L1 step-1
post-state, step_s + 1, the next B sequences of the step-B batch).  Checks after every step:
  ALLWRITES  every word of every byte range any op of the program writes (activation stash, scratch, grads,
             post-step state, loss), RTL DRAM vs interpreter memory;
  the 24 result tensors (check_compiler.results) RTL vs interpreter, and with --l1 RTL vs L1.
Host traffic: step 1 (slot 0) inputs are in the initial image; before step 2 the host writes only ids, key-mask
words and labels of slot 1 through the DRAM window, then rings the doorbell (slot 1) and acknowledges the IRQ.
Prints E2E PASS / FAIL.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "sw"))
sys.path.insert(0, os.path.join(ROOT, "model"))
import compile as CP  # noqa: E402
import check_compiler as CC  # noqa: E402
import interp  # noqa: E402
import isa  # noqa: E402

f32, U32 = np.float32, np.uint32
NAMES = ["loss", "logits", "g_coef", "g_dw", "g_db", "g_ow", "g_ob", "gnorm", "clip", "post_coef", "post_dw", "post_db",
         "post_ow", "post_ob", "post_m_coef", "post_m_dw", "post_m_db", "post_m_ow", "post_m_ob", "post_v_coef",
         "post_v_dw", "post_v_db", "post_v_ow", "post_v_ob"]
HEAD = ["dw", "db", "ow", "ob"]


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def build(memw, bdir, threads=1):
    obj = os.path.join(bdir, f"obj_m{memw}" + (f"_t{threads}" if threads > 1 else ""))
    exe = os.path.join(obj, "Vtb_top")
    hw = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(ROOT, "hw")) for f in fs if f.endswith(".v"))
    srcs = [os.path.join(ROOT, "tb/top/tb_top.sv"), os.path.join(ROOT, "tb/mem/tb_dram.sv")] + hw
    if os.path.exists(exe) and os.path.getmtime(exe) > max(os.path.getmtime(s) for s in srcs):
        return exe
    t0 = time.time()
    r = sh(f"verilator --binary --timing -O3 -j 16 --x-assign fast --x-initial fast -Wno-fatal -Wno-lint -Wno-style "
           f"-CFLAGS -O2 {f'--threads {threads}' if threads > 1 else ''} --top-module tb_top +define+MM_DMEMW={memw} -Mdir {obj} {' '.join(srcs)} > {obj}.build.log 2>&1")
    if not os.path.exists(exe):
        raise RuntimeError(f"verilator build failed: {obj}.build.log")
    print(f"built {exe} ({time.time() - t0:.0f} s)", flush=True)
    return exe


def kmask_words(c, am):
    kmw = np.zeros((c.B, c.kmw), U32)
    for b in range(c.B):
        for j in range(c.L):
            if am[b, j]:
                kmw[b, j >> 5] |= U32(1) << U32(j & 31)
    return kmw


def slot_words(c, ids, am, labels):
    """{byte offset in the slot: words} of the per-step host inputs (ids int32, key-mask words, labels int32)."""
    return {c.si["ids"]: ids.reshape(-1).astype(np.int32).view(U32), c.si["km"]: kmask_words(c, am).reshape(-1),
            c.si["lab"]: np.asarray(labels, np.int32).view(U32)}


def merge_regions(iv, maxn, align=4096):
    iv = sorted(((lo // align) * align, -(-hi // align) * align) for lo, hi in iv)
    out = []
    for lo, hi in iv:
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    while len(out) > maxn:                       # close the smallest gap
        g = int(np.argmin([out[i + 1][0] - out[i][1] for i in range(len(out) - 1)]))
        out[g][1] = out[g + 1][1]
        del out[g + 1]
    return [(lo, hi) for lo, hi in out]


def hexlines(words):
    w = np.ascontiguousarray(words, U32)
    nib = np.empty((w.size, 9), np.uint8)
    lut = np.frombuffer(b"0123456789abcdef", np.uint8)
    for i in range(8):
        nib[:, i] = lut[(w >> U32(28 - 4 * i)) & U32(15)]
    nib[:, 8] = 10
    return nib.tobytes()


def read_hex(path, n):
    a = np.frombuffer(open(path, "rb").read(), np.uint8)
    assert a.size == 9 * n, (path, a.size, n)
    a = a.reshape(n, 9)[:, :8].astype(np.int64)
    v = np.where(a >= 97, a - 87, a - 48)
    out = np.zeros(n, np.int64)
    for i in range(8):
        out = (out << 4) | v[:, i]
    return out.astype(U32)


class Mem:
    """sparse result view with interp.Machine's `mf` interface (zeros where nothing was dumped)."""

    def __init__(self):
        self.mem = np.zeros(CP.IMG_END // 4, U32)
        self.mf = self.mem.view(f32)


def l1_chain(c, r, nsteps, batches):
    import hwmodel.step as S
    W = S.Weights(os.path.join(CP.RUNS, "model_fp32.npz"))
    inp = dict(r["inp"])
    outs = []
    for k in range(nsteps):
        ids, am, lab = batches[k]
        d = dict(inp)
        d["input_ids"], d["attention_mask"], d["labels"] = ids, am, lab
        t0 = time.time()
        o = S.step(W, d, verbose=False)
        print(f"L1 step {k + 1} at B {c.B} L {c.L}: loss {float(o['loss']):.9g} ({time.time() - t0:.0f} s)", flush=True)
        outs.append(o)
        inp = dict(d)
        inp["step_s"] = int(d["step_s"]) + 1
        for k2 in ["coef"] + HEAD:
            inp["pre_" + k2], inp["pre_m_" + k2], inp["pre_v_" + k2] = o["post_" + k2], o["post_m_" + k2], o["post_v_" + k2]
    return outs


def cmp_tensors(tag, got, want):
    nbad = 0
    for k in NAMES:
        g = np.asarray(got[k], f32).view(U32).reshape(-1)
        w = np.asarray(want[k], f32).view(U32).reshape(-1)
        bad = int((g != w).sum()) if g.shape == w.shape else -1
        nbad += bad != 0
        if bad:
            print(f"CHECK {tag} {k:12s} {w.size:7d} words FAIL {bad} mismatches")
    print(f"CHECK {tag}: {24 - nbad}/24 result tensors bit-exact")
    return nbad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--l1", action="store_true", help="also check against the L1 step (needs --layers 12)")
    ap.add_argument("--maxcyc", type=int, default=50_000_000)
    ap.add_argument("--memw", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1, help="Verilator --threads")
    ap.add_argument("--stop-after", type=int, default=None, help="debug: stop after this unit op")
    ap.add_argument("--check-only", action="store_true", help="compare the dumps of an existing (running) simulation")
    ap.add_argument("--prep-only", action="store_true", help="image, goldens and build only; no simulation")
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "e2e"))
    a = ap.parse_args()
    assert not a.l1 or a.layers == 12, "the L1 step is the 12-layer model"
    c = CP.Cfg(a.layers, a.batch, a.seq)
    tag = f"l{a.layers}_b{a.batch}_s{a.seq}_n{a.steps}"
    bdir = os.path.join(a.out, tag)
    os.makedirs(bdir, exist_ok=True)
    t0 = time.time()
    r = CP.compile_all(c)
    inp, s0 = r["inp"], int(r["inp"]["step_s"])
    if a.stop_after is not None:                 # debug: END after the k-th unit op in program order (both sides)
        pr = next(arr for ad, arr in r["img"].reg if ad == CP.REG["PROG"])
        pc, k = 0, -1
        while True:
            h = isa.header(int(pr[pc // 4]))
            nsl = ((int(pr[pc // 4]) >> 21) & 3) + 1
            if k == a.stop_after:
                pr[pc // 4] = isa.OPS["END"][0] | (0xF << 12) | (0xF << 16)
                break
            if h["unit"] != isa.UNIT_SEQ:
                k += 1
            pc += 64 * nsl
        print(f"debug: END at {pc:#x}, after unit op {a.stop_after}")
    print(f"compiled {tag}: {r['ndesc']} descriptors, {r['prog'].size * 4} B, {r['flat_ops']} ops unrolled", flush=True)
    # per-step batches: step k uses rows [k B, (k+1) B) of the step-B batch
    full = dict(np.load(os.path.join(CP.RUNS, "stepB_l0f32.npz")))
    assert a.steps * c.B <= full["input_ids"].shape[0]
    batches = [(full["input_ids"][k * c.B:(k + 1) * c.B, :c.L].astype(np.int64),
                full["attention_mask"][k * c.B:(k + 1) * c.B, :c.L].astype(np.uint8),
                full["labels"][k * c.B:(k + 1) * c.B]) for k in range(a.steps)]
    assert np.array_equal(batches[0][0], r["ids"])
    img = r["img"]
    # host-once content beyond compile_all: embedding rows of later steps' tokens (the full table is resident in
    # the real image), and slot 1 with its loss weights (constant; only ids / mask / labels change per step)
    word = np.asarray(r["W"]["roberta.embeddings.word_embeddings.weight"], f32)
    have = set(int(x) for x in np.unique(r["ids"]))
    for ids, _, _ in batches[1:]:
        for t in np.unique(ids):
            if int(t) not in have:
                have.add(int(t))
                img.put(CP.REG["EMB_WORD"] + 4 * CP.D * int(t), word[t])
    if a.steps > 1:
        sl = np.zeros(CP.SLOT_BYTES // 4, U32)
        sl[c.si["w"] // 4:c.si["w"] // 4 + c.B] = np.ones(c.B, f32).view(U32)
        img.put(CP.REG["STEP_IN1"], sl)
    # every byte range the program touches -> DRAM model regions; write ranges -> dump / compare list
    items = CP.program(c, CP.EPOCHS * int(inp["n_batches"]))
    flat = CP.flatten(items, c, None, s0)
    rd_iv, wr_iv = [], []
    for name, f, _, _ in flat:
        if isa.OPS[name][1] == isa.UNIT_SEQ:
            continue
        R, Wr = CP.ranges(name, f, c)
        rd_iv += R
        wr_iv += Wr
    img_iv = [(ad, ad + 4 * arr.size) for ad, arr in img.reg]
    img_iv.append((CP.REG["PROG"], CP.REG["PROG"] + 4 * r["prog"].size + 256))
    # reads outside the image and the write ranges would read unwritten memory: left unmapped, so the DRAM model
    # flags them (SLVERR); ranges() over-approximates reads (e.g. EMB: the whole word table)
    regs = merge_regions(img_iv + wr_iv, 120)
    nwords = sum(hi - lo for lo, hi in regs) // 4
    memw = a.memw or 27                         # one build: 2^27 words = 512 MiB (the 12-layer image fits)
    wregs = merge_regions(wr_iv, 100000, align=4)
    ndump = sum(hi - lo for lo, hi in wregs) // 4
    print(f"DRAM model: {len(regs)} regions, {nwords * 4 / 2**20:.1f} MiB (MEMW {memw}); write ranges "
          f"{len(wregs)} = {ndump * 4 / 2**20:.2f} MiB", flush=True)
    # golden machine: same image
    m = interp.Machine(CP.IMG_END, r["csr"]["SEED"], s0, r["csr"]["SLOT_DELTA"], 0)
    m.load(img)
    pk = np.zeros(nwords, U32)                   # packed DRAM image = the golden machine's memory at load time
    o = 0
    for lo, hi in regs:
        pk[o:o + (hi - lo) // 4] = m.mem[lo // 4:hi // 4]
        o += (hi - lo) // 4
    freg, fimg, fdump = (os.path.join(bdir, n) for n in ("dram.reg", "dram.bin", "dump.reg"))
    cs = r["csr"]
    if not a.check_only:                         # never touch the files of a running simulation
        with open(freg, "w") as fr:
            fr.write("".join(f"{lo // 4:08x} {(hi - lo) // 4:08x}\n" for lo, hi in regs))
        pk.astype(">u4").tofile(fimg)
        with open(fdump, "w") as fd:
            fd.write("".join(f"{lo // 4:08x} {(hi - lo) // 4:08x}\n" for lo, hi in wregs))
        # host script
        seed = cs["SEED"]
        hs = [(1, 0x020, cs["PROG_BASE"]), (1, 0x024, cs["SLOT_DELTA"]), (1, 0x018, cs["STEP_T"]),
              (1, 0x060, seed & 0xFFFFFFFF), (1, 0x064, seed >> 32), (1, 0x008, 0x11)]
        for k in range(a.steps):
            hs.append((6, 0, 0))
            slot = k & 1
            if k > 0:                                # per-step inputs through the DRAM window
                base = CP.REG["STEP_IN"] if slot == 0 else CP.REG["STEP_IN1"]
                for off, words in slot_words(c, *batches[k]).items():
                    hs += [(3, (base + off) // 4 + i, int(w)) for i, w in enumerate(words)]
            hs += [(1, 0x010, 0x80000000 | slot), (4, 0, 0), (1, 0x014, 0), (7, 0, 0), (5, k + 1, 0)]
        hs += [(2, 0x018, 0), (2, 0x014, 0), (2, 0x070, 0)]
        fhost = os.path.join(bdir, "host.txt")
        with open(fhost, "w") as fh:
            fh.write("".join(f"{op:x} {x:x} {y:x}\n" for op, x, y in hs))
    print(f"prepared in {time.time() - t0:.0f} s", flush=True)
    # golden: interpreter, step by step (the slot-1 inputs written as the host writes them)
    gold = []
    for k in range(a.steps):
        slot = k & 1
        if k > 0:
            base = CP.REG["STEP_IN"] if slot == 0 else CP.REG["STEP_IN1"]
            for off, words in slot_words(c, *batches[k]).items():
                m.mem[(base + off) // 4:(base + off) // 4 + words.size] = words
            m.term[isa.T_STEP] = m.step_t
            m.term[isa.T_SLOT] = cs["SLOT_DELTA"] if slot else 0
            m.kcache = {}
        t1 = time.time()
        n = m.run(cs["PROG_BASE"])
        print(f"interp step {k + 1}: {n} descriptors ({time.time() - t1:.0f} s)", flush=True)
        gold.append((np.concatenate([m.mem[lo // 4:hi // 4] for lo, hi in wregs]), CC.results(c, m)))
    l1 = l1_chain(c, r, a.steps, batches) if a.l1 else None
    gfail = 0
    if l1:                                       # the two goldens agree at this shape (compiler-check at (B, L))
        for k in range(a.steps):
            gfail += cmp_tensors(f"step {k + 1} interp vs L1", gold[k][1], l1[k]) != 0
    # RTL
    if a.check_only:
        log, simt = open(f"{bdir}/sim.log").read().splitlines(), 0.0
    else:
        exe = build(memw, a.out, a.threads)
        if a.prep_only:
            print("prepared only")
            return
        log, simt = run_sim(exe, freg, fimg, fhost, fdump, bdir, a.maxcyc)
    nfail = gfail + (0 if any(ln.startswith("PASS tb_top") for ln in log) else 1)
    nfail += check_dumps(c, a, bdir, wregs, ndump, gold, l1)
    info = dict(tag=tag, sim_s=simt, regions=len(regs), memw=memw, log=[ln for ln in log if ln.startswith(("ACCT", "HOSTBYTES", "STEP"))])
    with open(f"{bdir}/result.json", "w") as f:
        json.dump(info, f, indent=1)
    print(f"E2E {tag} {'PASS' if nfail == 0 else 'FAIL'}")
    sys.exit(0 if nfail == 0 else 1)


def run_sim(exe, freg, fimg, fhost, fdump, bdir, maxcyc):
    t1 = time.time()
    run = sh(f"{exe} +reg={freg} +img={fimg} +host={fhost} +dump={fdump} +out={bdir}/dump +oplog={bdir}/oplog.txt "
             f"+maxcyc={maxcyc} > {bdir}/sim.log 2>&1")
    simt = time.time() - t1
    log = open(f"{bdir}/sim.log").read().splitlines()
    for ln in log:
        if ln.startswith(("STEP", "ACCT", "HOSTBYTES", "PASS", "FAIL", "CSR", "DRAMSTATS")):
            print(ln)
    print(f"simulation: {simt:.0f} s wall", flush=True)
    return log, simt


def check_dumps(c, a, bdir, wregs, ndump, gold, l1):
    nfail = 0
    for k in range(a.steps):
        p = f"{bdir}/dump_{k + 1}.hex"
        if not os.path.exists(p):
            print(f"FAIL no dump after step {k + 1}")
            nfail += 1
            continue
        d = read_hex(p, ndump)
        gw, gres = gold[k]
        bad = int((d != gw).sum())
        print(f"CHECK step {k + 1} ALLWRITES {ndump} words: {'PASS' if bad == 0 else f'FAIL {bad} mismatches'}")
        if bad:
            nfail += 1
            first = np.nonzero(d != gw)[0][:4]
            offs = np.cumsum([0] + [(hi - lo) // 4 for lo, hi in wregs])
            for i in first:
                j = int(np.searchsorted(offs, i, side="right") - 1)
                print(f"  word {wregs[j][0] + 4 * (i - offs[j]):#010x}: rtl {d[i]:08x} interp {gw[i]:08x}")
        mm = Mem()
        o2 = 0
        for lo, hi in wregs:
            mm.mem[lo // 4:hi // 4] = d[o2:o2 + (hi - lo) // 4]
            o2 += (hi - lo) // 4
        res = CC.results(c, mm)
        nfail += cmp_tensors(f"step {k + 1} RTL vs interp", res, gres) != 0
        if l1:
            nfail += cmp_tensors(f"step {k + 1} RTL vs L1", res, l1[k]) != 0
        print(f"step {k + 1} loss RTL {float(res['loss']):.9g}" + (f" L1 {float(l1[k]['loss']):.9g}" if l1 else ""))
    return nfail


if __name__ == "__main__":
    main()
