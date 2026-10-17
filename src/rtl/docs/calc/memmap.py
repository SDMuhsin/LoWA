#!/usr/bin/env python3
"""LoWA-TA DRAM layout for roberta-base, B<=32, L<=128 (docs/memory_map.md is printed from this).
    python3 src/rtl/docs/calc/memmap.py [--C 2] [--steps 368160]
"""
import argparse

D, H, F, NL, VOCAB, NPOS = 768, 12, 3072, 12, 50265, 514
K_ATOM, S_TAP = 256, 25
B, L = 32, 128
N = B * L
FP = 4
ALIGN = 4096


def layout(C=3):
    """[(name, base, nbytes, note, rw)] in address order (also used by tb/mem/mm_image.py, P6a)."""
    regions = []

    def R(name, nbytes, note, rw):
        regions.append((name, nbytes, note, rw))

    # --- written once by the host
    R("PROG", 256 * 1024, "sequencer program (descriptors), see isa.md", "host W once")
    R("CONST", 4096, "run constants: gamma, h[25], lr0 x2, wd, betas, eps, 1/w, 1/(T-w), drop thr/scale, seeds", "host W once")
    R("EMB_WORD", VOCAB * D * FP, "word_embeddings [50265 x 768]", "host W once")
    R("EMB_POS", NPOS * D * FP, "position_embeddings [514 x 768]", "host W once")
    R("EMB_TT_LN", (D + 2 * D) * FP, "token_type[0] row, LN_emb gamma,beta", "host W once")
    per_layer = (4 * D * D + 4 * D + 2 * D * F + F + D + 4 * D) * FP
    R("LAYERS", NL * per_layer, "12 x {Wq,bq,Wk,bk,Wv,bv,Wo,bo,LN1 g/b,W1,b1,W2,b2,LN2 g/b} row-major [out x in]", "host W once")
    R("LOWA_CTR", 2 * NL * K_ATOM * 4, "24 x 256 centres (ci<<16 | cj), seed 777 order", "host W once")
    R("LOWA_HITS", 2 * NL * K_ATOM * S_TAP * 4, "24 x 6400 merge hit entries sorted by W block (dataflow.md 3)", "host W once")
    # --- trainable state (device resident; head initialised by host, coefs zero)
    n_head = D * D + D + D * C + C
    n_coef = 2 * NL * K_ATOM
    for nm in ("PARAM", "GRAD", "ADAM_M", "ADAM_V"):
        R(nm, n_coef * FP + n_head * FP, f"{nm.lower()}: 24x256 coefs then head (dense W,b, out_proj W,b; C<={C})", "device RW")
    R("OPT_SCAL", 4096, "step counter t, beta1^t, beta2^t, last loss, grad-norm", "device RW")
    # --- per-step input ring (host writes, double-buffered)
    step_in = N * 2 + N // 8 + B * 4 + 16
    R("STEP_IN", 2 * ALIGN * ((step_in + ALIGN - 1) // ALIGN), f"2 slots x {step_in} B: ids u16, mask bits, labels, descriptor", "host W per step")
    # --- activation stash (forward writes, backward reads)
    st_layer = (5 * N * D + N * F) * FP + (4 * N + 2 * B * H * L) * FP
    R("STASH_X0", N * D * FP, "layer-0 input (dropout(LN_emb)), Q/V LoWA X operand", "device RW")
    R("STASH", NL * st_layer, "12 x {z1, z2, Q, K, V [N x 768], u [N x 3072], mu/rstd x2 [N], softmax m, r [B*H*L]}", "device RW")
    # --- scratch (reused each layer)
    R("SCR_ACT", 4 * N * F * FP, "ping-pong activations: h, ctx, g=GELU(u) [N x 3072], dX streams", "device RW")
    R("SCR_ATT", 2 * 3 * H * L * L * FP, "attention S/P', dP', dS for one sequence (12 heads), x2 ping-pong", "device RW")
    R("SCR_CLS", 8 * B * D * FP, "head activations and grads", "device RW")
    R("LOSS_OUT", 64, "per-step loss / status (optional read by host)", "host R optional")

    out, addr = [], 0
    for nm, nb, note, rw in regions:
        out.append((nm, addr, nb, note, rw))
        addr += (nb + ALIGN - 1) // ALIGN * ALIGN
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=3, help="max num_labels the image reserves for")
    ap.add_argument("--steps", type=int, default=368160, help="max training steps (MNLI 30 ep)")
    a = ap.parse_args()
    C = a.C
    n_head = D * D + D + D * C + C
    n_coef = 2 * NL * K_ATOM
    lay = layout(C)
    print(f"{'region':10s} {'base':>12s} {'bytes':>15s}  rw               note")
    for nm, base, nb, note, rw in lay:
        print(f"{nm:10s} 0x{base:010x} {nb:15,d}  {rw:16s} {note}")
    addr = lay[-1][1] + (lay[-1][2] + ALIGN - 1) // ALIGN * ALIGN
    print(f"{'TOTAL':10s} {'':12s} {addr:15,d}  = {addr/2**30:.3f} GiB")
    print(f"trainable: coefs {n_coef:,} + head {n_head:,} = {n_coef+n_head:,}")


if __name__ == "__main__":
    main()
