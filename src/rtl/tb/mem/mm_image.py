"""DRAM image for the P6a memory-system tests: a sparse set of word regions placed by the memory_map
layout (docs/calc/memmap.py `layout()`), written as the two files tb/mem/tb_dram.sv loads
(regions: "base_word nwords" per line; data: $readmemh of the packed words)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "docs", "calc"))
import memmap  # noqa: E402

REGION = {nm: base for nm, base, _, _, _ in memmap.layout(3)}
# Keep-bit stand-in for the mk channel (NOT part of the design: R2 / D1, the bits come from the on-chip
# Threefry generator in P6b).  Placed above the 1.947 GiB image so it can never alias a real region.
REGION["MASK_STANDIN"] = 0x8000_0000
PER_LAYER = (4 * 768 * 768 + 4 * 768 + 2 * 768 * 3072 + 3072 + 768 + 4 * 768) * 4
# byte offsets inside one layer body (memmap.py order: Wq,bq,Wk,bk,Wv,bv,Wo,bo,LN1,W1,b1,W2,b2,LN2)
_D, _F = 768, 3072
_o = 0
LAYER_OFF = {}
for _nm, _n in (("Wq", _D * _D), ("bq", _D), ("Wk", _D * _D), ("bk", _D), ("Wv", _D * _D), ("bv", _D),
                ("Wo", _D * _D), ("bo", _D), ("LN1", 2 * _D), ("W1", _F * _D), ("b1", _F), ("W2", _D * _F),
                ("b2", _D), ("LN2", 2 * _D)):
    LAYER_OFF[_nm] = _o
    _o += 4 * _n
assert _o == PER_LAYER


def layer_addr(layer, name):
    return REGION["LAYERS"] + layer * PER_LAYER + LAYER_OFF[name]


class Image:
    """word-addressed sparse memory; every byte address is 4-byte aligned."""

    def __init__(self):
        self.reg = {}                       # base word -> np.uint32 array

    def put(self, byte_addr, arr):
        assert byte_addr % 4 == 0
        a = np.ascontiguousarray(arr).view(np.uint32).ravel().copy()
        w = byte_addr // 4
        for b, d in self.reg.items():       # no overlaps allowed
            assert w + a.size <= b or b + d.size <= w, f"overlap at {byte_addr:#x}"
        self.reg[w] = a
        return byte_addr

    def zeros(self, byte_addr, nwords):
        return self.put(byte_addr, np.zeros(nwords, np.uint32))

    def find(self, w):
        for b, d in self.reg.items():
            if b <= w < b + d.size:
                return d, w - b
        raise KeyError(f"word {w:#x} unmapped")

    def rd(self, w):
        d, i = self.find(w)
        return int(d[i])

    def wr(self, w, v):
        d, i = self.find(w)
        d[i] = v

    def words(self):
        return sum(d.size for d in self.reg.values())

    def write(self, regfile, datfile):
        assert len(self.reg) <= 128, "tb_dram holds 128 regions"
        with open(regfile, "w") as fr, open(datfile, "w") as fd:
            for b in sorted(self.reg):
                d = self.reg[b]
                fr.write(f"{b:08x} {d.size:08x}\n")
                fd.write("\n".join(f"{int(x):08x}" for x in d) + "\n")
