#!/usr/bin/env python3
"""P6c sequencer tests (`make test-ctrl`): tb_ct_seq (ct_seq + ct_dma, stub engines, random memory timing) on
  real_reduced  the compiled reduced program (1 layer, B 2, L 32: 87 descriptors), every unit payload vs the reference
  real_full     the compiled full step program (141 descriptors, LOOP 12 up / 11 down: 413 unit + fence ops)
  flags         synthetic: WAIT / SIG / CLR across units, FENCE, LOOP with layer terms, STEP_T and slot terms
  dma           DMA_COPY 1- / 2- / 3-level copies, unaligned lengths, a chunk-crossing row, memory checked
  badop         illegal opcode -> ERR.decode, halt, no IRQ
Expected values come from the ISA reference walk (sw/isa.py decode + the LOOP rule of docs/isa.md), never from
the RTL.  Checks: payload words of every start in program order, issue order, one op per unit, retire after
done + write idle (+ merge done), FENCE and WAIT/SIG ordering, CSR results (STEP_T, STEP_DONE, ERR, IRQ).
Prints `PASS ct_seq <n> checks` or `FAIL ...` per case and TEST-CTRL PASS / FAIL.
"""
import os
import random
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "sw"))
import isa  # noqa: E402

B = os.path.join(ROOT, "build", "test-ctrl")
UIDX = {isa.UNIT_DMA: 0, isa.UNIT_GEMM: 1, isa.UNIT_VEC: 2, isa.UNIT_LOWA: 3}
NPAY = {0: 9, 1: 36, 2: 25, 3: 10}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def build():
    os.makedirs(B, exist_ok=True)
    r = sh(f"verilator --binary -j 16 -O2 -Wno-fatal -Wno-lint -Wno-style --top-module tb_ct_seq -Mdir {B}/obj "
           f"{ROOT}/tb/ctrl/tb_ct_seq.sv {ROOT}/hw/ctrl/ct_seq.v {ROOT}/hw/ctrl/ct_dma.v {ROOT}/hw/common/cm_rstbuf.v > {B}/build.log 2>&1")
    if not os.path.exists(f"{B}/obj/Vtb_ct_seq"):
        raise RuntimeError("verilator build failed, see build/test-ctrl/build.log")


def walk(words, step_t, slot_delta=0, slot=0):
    """reference: program order list of ('U', unit, payload, hdr) / ('F', mask) / ('E',)"""
    term = [0] * 16
    term[isa.T_STEP], term[isa.T_SLOT] = step_t, (slot_delta if slot else 0)
    strd = [0] * 16
    pc, loop, out = 0, None, []
    for _ in range(100000):
        w0 = int(words[pc // 4])
        nsl = ((w0 >> 21) & 3) + 1
        try:
            h, name, f, pay = isa.decode_words(words[pc // 4:pc // 4 + 16 * nsl], term)
        except ValueError:
            out.append(("X",))
            return out
        nxt = pc + 64 * nsl
        if name == "LOOP":
            loop = dict(start=nxt, end=nxt + f["body"], left=f["count"], dir=f["dir"])
            term[isa.T_LIDX] = f["start"]
            term[1:13] = [0] * 12
        elif name == "SETCSR":
            if 1 <= f["csr"] <= 12:
                strd[f["csr"]] = f["value"]
        elif name == "FENCE":
            out.append(("F", f["mask"]))
        elif name == "END":
            out.append(("E",))
            return out
        elif name != "NOP":
            u = UIDX[h["unit"]]
            out.append(("U", u, pay[:NPAY[u]], h))
        pc = nxt
        if loop and pc == loop["end"]:
            loop["left"] -= 1
            if loop["left"] > 0:
                s = -1 if loop["dir"] else 1
                term[isa.T_LIDX] += s
                for k in range(1, 13):
                    term[k] = (term[k] + s * strd[k]) & 0xFFFFFFFF
                pc = loop["start"]
            else:
                loop = None
    raise RuntimeError("no END")


def write_hex(path, words, extra=()):
    with open(path, "w") as f:
        f.write("@0\n")
        f.write("\n".join(f"{int(w):08x}" for w in words) + "\n")
        for a, arr in extra:
            f.write(f"@{a:x}\n" + "\n".join(f"{int(w):08x}" for w in arr) + "\n")


def run_case(name, words, step_t=0, slot=0, slot_delta=0, extra=(), plus="", seed=1, maxcyc=3_000_000):
    hexp, logp = f"{B}/{name}.hex", f"{B}/{name}.log"
    write_hex(hexp, words, extra)
    r = sh(f"{B}/obj/Vtb_ct_seq +prog={hexp} +log={logp} +maxcyc={maxcyc} +seed={seed} +step_t={step_t} "
           f"+slot={slot} +slot_delta={slot_delta} {plus}")
    if "DONE ct_seq" not in r.stdout:
        return None, r.stdout[-500:]
    ev = {"S": [], "R": [], "D": [], "I": [], "M": [], "N": [], "C": {}, "W": {}}
    for ln in open(logp):
        t = ln.split()
        if t[0] == "C":
            ev["C"][t[1]] = t[2]
        elif t[0] == "W":
            ev["W"][int(t[1])] = int(t[2], 16)
        elif t[0] == "S":
            ev["S"].append((int(t[1]), int(t[2]), [int(x, 16) for x in t[3:]]))
        elif t[0] in "RDI":
            ev[t[0]].append((int(t[1]), int(t[2])))
        else:
            ev[t[0]].append(int(t[1]))
    return ev, ""


def check(ev, ref, step_t, expect_err=0):
    """returns (n checks, list of failures)"""
    fails, n = [], 0
    ops = [r for r in ref if r[0] == "U"]
    if ev["C"]["ID"] != "4c575441":
        fails.append("ID")
    if len(ev["S"]) != len(ops):
        fails.append(f"{len(ev['S'])} starts, reference {len(ops)}")
        return n, fails
    # per-unit retire / done / idle lists in order
    per = {u: [i for i, o in enumerate(ops) if o[1] == u] for u in range(4)}
    tS, tR, tD, tI = {}, {}, {}, {}
    for i, (u, c, pay) in enumerate(ev["S"]):
        n += 1
        if u != ops[i][1] or pay != ops[i][2]:
            fails.append(f"op {i}: start unit {u} payload differs from reference (unit {ops[i][1]})")
        tS[i] = c
    for key, store in (("R", tR), ("D", tD), ("I", tI)):
        cnt = {u: 0 for u in range(4)}
        for u, c in ev[key]:
            if cnt[u] < len(per[u]):
                store[per[u][cnt[u]]] = c
            cnt[u] += 1
    for i, o in enumerate(ops):
        n += 3
        if i not in tR:
            fails.append(f"op {i} never retired")
            continue
        if i > 0 and tS[i] < tS[i - 1]:
            fails.append(f"op {i} issued out of order")
        if not tR[i] > tD.get(i, 1 << 60):
            fails.append(f"op {i} retired before done")
        if o[1] != 0 and not tR[i] >= tI.get(i, 1 << 60):
            fails.append(f"op {i} retired before its writes drained")
    # unit serialisation, FENCE, WAIT / SIG
    k, last_on, fenced = 0, {}, []
    sig_of = {}
    for r in ref:
        if r[0] == "F":
            fenced.append((k, r[1]))
        elif r[0] == "U":
            i, u, h = k, r[1], r[3]
            if u in last_on:
                n += 1
                if not tS[i] > tR[last_on[u]]:
                    fails.append(f"op {i} started before the previous op of unit {u} retired")
            for (kk, m) in fenced:
                for j in range(kk):
                    if m & (1 << ops[j][1]):
                        n += 1
                        if not tS[i] > tR[j]:
                            fails.append(f"op {i} started before fenced op {j} retired")
            if h["wait"] != 0xF:
                n += 1
                j = sig_of.get(h["wait"])
                if j is None or not tS[i] > tR[j]:
                    fails.append(f"op {i} WAIT {h['wait']} not after its producer")
                if h["clr"]:
                    sig_of.pop(h["wait"], None)
            if h["sig"] != 0xF:
                sig_of[h["sig"]] = i
            last_on[u] = i
            k += 1
    fenced_end = [r for r in ref if r[0] == "E"]
    n += 4
    if expect_err:
        if ev["C"]["ERR"] != "1" or ev["C"]["IRQ"] != "0":
            fails.append(f"bad-opcode case: ERR {ev['C']['ERR']} IRQ {ev['C']['IRQ']}")
    else:
        if not fenced_end:
            fails.append("reference has no END")
        if int(ev["C"]["STEP_T"]) != step_t + 1 or ev["C"]["STEP_DONE"] != "1" or ev["C"]["IRQ"] != "1" \
                or ev["C"]["IRQ_AFTER_ACK"] != "0" or ev["C"]["ERR"] != "0" or int(ev["C"]["MKSTEP"]) != step_t + 1:
            fails.append(f"CSRs after END: {ev['C']}")
    return n, fails


# ------------------------------------------------------------------------------------------- programs
def prog(ops):
    return np.concatenate([isa.encode(*o[:2], **(o[2] if len(o) > 2 else {})) for o in ops]).astype(np.uint32)


def flags_prog(r):
    """WAIT / SIG chains across GEMM, VEC, LOWA, a FENCE, a 3-iteration LOOP with layer / slot / step terms."""
    ops = [("SETCSR", dict(csr=1, value=0x1000)), ("SETCSR", dict(csr=5, value=0x40))]
    ops.append(("GEMM", dict(m=1, n=2, k0=3, a_base0=0x100, c_base=0x200, m_en=1), dict(sig=3)))
    ops.append(("VEC", dict(vop=4, rows=7, a_base=0x300), dict(wait=3, clr=1, sig=4)))
    ops.append(("LOWA_SG", dict(op=0, a0=0x400, len=9), dict(wait=4, clr=1, sig=5)))
    ops.append(("VEC", dict(vop=1, a_base=0x500), dict(wait=5, clr=1)))
    ops.append(("FENCE", dict(mask=0xF)))
    ops.append(("LOWA_SCAL", dict(op=1, a0=0), dict(rel=dict(a0=isa.T_STEP))))
    body = [("GEMM", dict(a_base0=0x10000, b_base0=0x20000, layer=0, c_base=0x30000),
             dict(rel=dict(a_base0=1, b_base0=5, layer=isa.T_LIDX), sig=7)),
            ("VEC", dict(a_base=0x40000, layer=0, ids_base=0x50), dict(rel=dict(a_base=1, layer=isa.T_LIDX, ids_base=isa.T_SLOT),
                                                                    wait=7, clr=1)),
            ("LOWA_SG", dict(op=0, a6=0x60000), dict(rel=dict(a6=5)))]
    bw = prog(body)
    ops.append(("LOOP", dict(body=int(bw.size * 4), count=3, start=2, dir=1)))
    words = [prog(ops), bw]
    tail = [("DMA_COPY", dict(src=0x800, dst=0x900, n0=3), dict(sig=2)),
            ("VEC", dict(vop=10), dict(wait=2, clr=1)),
            ("GEMM", dict(m=5)), ("VEC", dict(vop=11)), ("LOWA_SG", dict(op=0)), ("END", {})]
    words.append(prog(tail))
    return np.concatenate(words)


def dma_prog():
    ops = [("DMA_COPY", dict(src=0x10000, dst=0x20000, n0=13)),
           ("DMA_COPY", dict(src=0x10000, dst=0x20100, n0=70, n1=3, s_s1=0x200, d_s1=0x120)),    # chunk-crossing rows
           ("DMA_COPY", dict(src=0x10004, dst=0x21000, n0=5, n1=2, s_s1=0x40, d_s1=0x20, n2=3, s_s2=0x400, d_s2=0x80)),
           ("FENCE", dict(mask=1)),
           ("DMA_COPY", dict(src=0x30000, dst=0x22000, n0=9, n1=4, s_s1=0, d_s1=0x24)),            # fill (stride 0)
           ("END", {})]
    return prog(ops)


def dma_expect(mem0):
    m = dict(mem0)
    def cp(src, dst, n0, n1=1, s_s1=0, d_s1=0, n2=1, s_s2=0, d_s2=0):
        for i2 in range(n2):
            for i1 in range(n1):
                s, d = src + i2 * s_s2 + i1 * s_s1, dst + i2 * d_s2 + i1 * d_s1
                for w in range(n0):
                    m[d // 4 + w] = m.get(s // 4 + w, 0)
    cp(0x10000, 0x20000, 13)
    cp(0x10000, 0x20100, 70, 3, 0x200, 0x120)
    cp(0x10004, 0x21000, 5, 2, 0x40, 0x20, 3, 0x400, 0x80)
    cp(0x30000, 0x22000, 9, 4, 0, 0x24)
    return m


def main():
    quick = "--quick" in sys.argv
    build()
    cases, nfail = [], 0
    for tag, nm in (("l1_b2_s32", "real_reduced"), ("l12_b32_s128", "real_full")):
        p = os.path.join(ROOT, "build", "compile", f"prog_{tag}.bin")
        if not os.path.exists(p):
            sh(f"cd {ROOT} && python3 sw/compile.py " + ("--layers 1 --batch 2 --seq 32" if tag.startswith("l1_") else ""))
        cases.append((nm, np.fromfile(p, np.uint32), dict(step_t=140, plus="+nowrite")))
    r = random.Random(5)
    cases.append(("flags", flags_prog(r), dict(step_t=7, slot=1, slot_delta=0x5000)))
    cases.append(("flags_s0", flags_prog(r), dict(step_t=0, slot=0, slot_delta=0x5000, seed=3)))
    src = {0x10000 // 4 + i: (0xA0000000 + i) for i in range(1024)}
    src.update({0x30000 // 4 + i: 0x5EED0000 + i for i in range(16)})
    cases.append(("dma", dma_prog(), dict(extra=[(0x10000 // 4, [src[0x10000 // 4 + i] for i in range(1024)]),
                                                  (0x30000 // 4, [src[0x30000 // 4 + i] for i in range(16)])],
                                           plus=f"+dlo={0x20000 // 4} +dn={0x3000 // 4}", dma=src)))
    bad = prog([("VEC", dict(vop=1)), ("NOP", {})]).copy()
    bad[32] = (int(bad[32]) & 0xFFFFFF00) | 0x7E                                    # illegal opcode on the 2nd descriptor
    cases.append(("badop", bad, dict(err=1)))
    if quick:
        cases = [c for c in cases if c[0] != "real_full"]
    for nm, words, kw in cases:
        st = kw.get("step_t", 0)
        ev, msg = run_case(nm, words, st, kw.get("slot", 0), kw.get("slot_delta", 0), kw.get("extra", ()),
                           kw.get("plus", ""), kw.get("seed", 1))
        if ev is None:
            print(f"FAIL ct_seq {nm}: no completion ({msg.strip()[-200:]})")
            nfail += 1
            continue
        ref = walk(words, st, kw.get("slot_delta", 0), kw.get("slot", 0))
        if kw.get("err"):
            ref = [x for x in ref if x[0] != "X"]
        n, fails = check(ev, ref, st, kw.get("err", 0))
        if "dma" in kw:
            exp = dma_expect(kw["dma"])
            for w, v in ev["W"].items():
                n += 1
                if exp.get(w, 0) != v:
                    fails.append(f"DMA word {w * 4:#x}: {v:08x} expected {exp.get(w, 0):08x}")
                    if len(fails) > 5:
                        break
        nfail += bool(fails)
        print(f"{'PASS' if not fails else 'FAIL'} ct_seq {nm}: {n} checks, {len(ev['S'])} unit ops, "
              f"{ev['C'].get('CYCLES')} cycles" + ("" if not fails else " :: " + "; ".join(fails[:4])))
    print(f"TEST-CTRL {'PASS' if nfail == 0 else 'FAIL'} ({len(cases) - nfail}/{len(cases)} cases)")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
