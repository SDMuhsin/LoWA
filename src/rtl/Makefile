# LoWA-TA: one entry point.  model-check (P1), lint / test-fp / synth-fp (P2), ..., test-e2e (P7a); synth and
# pnr are stubs that FAIL on purpose, so nothing reads as passing before it is built.
SHELL := /bin/bash
PY   ?= $(abspath ../../env/bin/python)
# EDA tools (Yosys, Verilator, Icarus, OpenSTA, PDK): llmdocs/asic_flow_tutorial.md section 1
EDA  := source /workspace/GanAccelAsic/src/flow/env.sh >/dev/null 2>&1 &&
RUNS := model/runs
L0   := $(RUNS)/model_fp32.npz $(RUNS)/stepA_l0f32.npz $(RUNS)/stepA_l0f64.npz $(RUNS)/stepB_l0f32.npz $(RUNS)/stepB_l0f64.npz

.PHONY: model-check model-arith model-l0 model-l1 lint gen-check test-fp test-fp-quick test-unit \
        test-gemm test-gemm-quick test-vec test-vec-quick test-lowa test-lowa-quick test-mem test-mem-quick test-rng test-rng-quick synth-mem synth-rng synth-ctrl test-ctrl test-ctrl-quick compile compiler-check synth-fp synth-gemm synth-vec synth-lowa test-e2e test-e2e-full synth pnr clean-model clean \
        test-mem-t65 test-mem-gf180 test-mem-banks syn65-scout syn65-status syn65-results sync65-pull sync65-wip sync65-site sync65-results \
        syn65u-status syn65u-results syn65u-all syn65-exit p8d65 saif65-gm saif65-vu

## P1 gate: arithmetic proof, L0 dumps (GPU cuda:0, ~1.5 min, only if missing), L1 steps, L1 vs L0
model-check: model-arith model-l1
	cd model && $(PY) check_l1_l0.py runs A B

model-arith:
	$(MAKE) -s -C model/hwmodel/csrc
	cd model && $(PY) hwmodel/test_arith.py

$(L0) &: model/l0_ref.py model/rng.py
	CUDA_VISIBLE_DEVICES=0 $(PY) model/l0_ref.py --out $(RUNS)

model-l0: $(L0)

model-l1: $(L0)
	$(MAKE) -s -C model/hwmodel/csrc
	cd model && $(PY) rng.py && $(PY) -m hwmodel.step runs A && $(PY) -m hwmodel.step runs B

clean-model:
	rm -f $(RUNS)/*.npz
	$(MAKE) -s -C model/hwmodel/csrc clean

## P2: every hw/ file through Verilator -Wall, Yosys and Icarus (flow/lint.sh)
lint:
	$(EDA) flow/lint.sh

## generated RTL is up to date, and the SFU micro-programs reproduce L1 when run op by op
gen-check:
	@mkdir -p build/gen
	$(PY) flow/gen/sfu_ucode.py emit build/gen/fp_sfu_ucode.v
	cmp build/gen/fp_sfu_ucode.v hw/fp/fp_sfu_ucode.v || { echo "GEN-CHECK FAIL: hw/fp/fp_sfu_ucode.v is stale"; false; }
	$(PY) flow/gen/sfu_ucode.py check 200000
	$(PY) flow/gen/vu_ucode.py emit build/gen/vu_ucode.v
	cmp build/gen/vu_ucode.v hw/vec/vu_ucode.v || { echo "GEN-CHECK FAIL: hw/vec/vu_ucode.v is stale"; false; }
	OMP_NUM_THREADS=1 $(PY) flow/gen/vu_ucode.py check 1

## P2 gate: hw/fp bit-exact vs L1 (Verilator full sets + Icarus stage-count sweep), tb/fp/run_tests.py
test-fp: gen-check
	$(EDA) $(PY) tb/fp/run_tests.py

test-fp-quick:
	$(EDA) $(PY) tb/fp/run_tests.py --quick

## P3 gate: hw/gemm bit-exact vs L1 (PE unit, engine unit cases, real roberta-base GEMMs, 4x4 stress +
## Icarus smoke), tb/gemm/run_tests.py; cycles vs docs/calc/budget.py
test-gemm:
	$(EDA) $(PY) tb/gemm/run_tests.py

test-gemm-quick:
	$(EDA) $(PY) tb/gemm/run_tests.py --quick

## P4 gate: hw/vec bit-exact vs L1 (unit ops with directed corners + random, real roberta-base data from the
## L1 step replay, stress with memory stalls, Icarus smoke at V = 4), tb/vec/run_tests.py
test-vec: gen-check
	$(EDA) $(PY) tb/vec/run_tests.py

test-vec-quick:
	$(EDA) $(PY) tb/vec/run_tests.py --quick

## P5 gate: hw/lowa bit-exact vs L1 (merge on synthetic + real Wq tiles, sampled gradient, LOWA_SCAL, AdamW unit +
## real step-B coefficients / head, Icarus smoke), tb/lowa/run_tests.py
test-lowa:
	$(EDA) $(PY) tb/lowa/run_tests.py

test-lowa-quick:
	$(EDA) $(PY) tb/lowa/run_tests.py --quick

## P6a gate: hw/mem (AXI4 master + DMA channels) vs the tb/mem DRAM model (unit: bursts / 4 KiB / widths / outstanding /
## backpressure / ordering / channel patterns, python reference), and gm_top / vu_top / lw_top end to end on mm_top + tb_dram
## (L1 goldens), tb/mem/run_tests.py
test-mem:
	$(EDA) $(PY) tb/mem/run_tests.py

test-mem-quick:
	$(EDA) $(PY) tb/mem/run_tests.py --quick

## P6b gate: hw/rng (on-chip Threefry dropout-mask generator) bit-exact vs model/rng.py: KAT + 1M random blocks,
## real step-B masks on the mk channel, generator-sourced real GEMM / vector cases vs L1, tb/rng/run_tests.py
test-rng:
	$(EDA) $(PY) tb/rng/run_tests.py

test-rng-quick:
	$(EDA) $(PY) tb/rng/run_tests.py --quick

## P8a: the same memory tests on the tsmc65 latch-register-file backend (behavioural clock gate) and on the gf180 macros.
## The latch RF simulates 60-75x slower than the generic array, so test-mem-t65 defaults to the --quick set (all unit cases +
## the non-real integration cases); T65_MEM_ARGS= runs the real cases too (real_ffn1_256 alone: about 10 h per DRAM setting).
T65_MEM_ARGS ?= --quick
test-mem-t65:
	$(EDA) $(PY) tb/mem/run_tests.py --tech tsmc65 $(T65_MEM_ARGS)

test-mem-gf180:
	$(EDA) $(PY) tb/mem/run_tests.py --tech gf180

## P8a: mm_sram_bank port contract on all three backends (Icarus + Verilator, tb/mem/tb_mm_sram_bank.sv)
test-mem-banks:
	$(EDA) $(PY) tb/mem/run_banks.py

test-unit: test-fp test-gemm test-vec test-lowa test-mem test-rng test-ctrl

## P2: area / STA table of every fp unit, then gate-level simulation of the production netlists
synth-fp:
	$(EDA) $(PY) flow/synth/run_fp.py
	$(EDA) $(PY) tb/fp/run_tests.py --gls

## P3: area / STA of the PE (stage sweep), the 4x4 / 8x8 / 16x16 arrays and the 16x16 engine
synth-gemm:
	$(EDA) $(PY) flow/synth/run_gemm.py

## P4: area / STA of one lane and the 16-lane vector unit (flow/synth/synth_vec.sh)
synth-vec:
	$(EDA) $(PY) flow/synth/run_vec.py

## P5: area / STA of the LoWA unit (flow/synth/synth_vec.sh with wrap_lw_top)
synth-lowa:
	$(EDA) $(PY) flow/synth/run_lowa.py

## P6a: area / STA of the memory system (mm_top, full-chip channel set) through flow/synth/synth_vec.sh
synth-mem:
	$(EDA) $(PY) flow/synth/run_mem.py

## P6b: area / STA of the dropout-mask generator (rn_mkgen, vector-rate and GEMM-rate configurations)
synth-rng:
	$(EDA) $(PY) flow/synth/run_rng.py

## P6c: compiler (model config -> program + DRAM image under build/compile) and its golden test (the program run
## through the interpreter = sequencer decode + L1 per-op entry points, bit-exact vs the L1 step)
compile:
	$(PY) sw/compile.py
	$(PY) sw/compile.py --layers 1 --batch 2 --seq 32

compiler-check:
	$(PY) sw/check_compiler.py

## P6c: sequencer (hw/ctrl) unit tests + the reduced program on sequencer + engines + memory system + DRAM model
test-ctrl:
	$(EDA) $(PY) tb/ctrl/run_tests.py

test-ctrl-quick:
	$(EDA) $(PY) tb/ctrl/run_tests.py --quick

synth-ctrl:
	$(EDA) $(PY) flow/synth/run_ctrl.py

clean:
	rm -rf build

## P7a: the chip top (hw/top) end to end on the DRAM model, driven through its AXI4-Lite CSR slave.
## test-e2e: the reduced program (1 layer, B 2, L 32), two steps (step 2 host writes ids / mask / labels to slot 1),
##   every written word + the 24 result tensors bit-exact vs the interpreter (quick, ~minutes)
## test-e2e-full: the 12-layer roberta-base program at B 2, L 32 (only batch x seq reduced), two steps, bit-exact vs
##   the interpreter and vs two chained L1 steps (long: hours)
test-e2e:
	$(EDA) $(PY) tb/top/run_e2e.py --layers 1 --batch 2 --seq 32 --steps 2 --maxcyc 30000000

test-e2e-full:
	$(EDA) $(PY) tb/top/run_e2e.py --layers 12 --batch 2 --seq 32 --steps 2 --l1 --maxcyc 200000000

## P8a: TSMC-65 synthesis on the CMC server (flow/tsmc65; server and tree from the git-ignored site/site.sh).
## Code goes up by git (push, then make sync65-pull) or, before a push, make sync65-wip.  Jobs run detached, one DC at a time:
##   make syn65-<block> [PERIODS="5.0 4.0 6.67"]   one block (flow/tsmc65/common/blocks.tcl) at each period
##   make syn65-scout                              the P8a scout: fma pe sfu bank768 lane gm4 at PERIODS
##   make syn65-status / syn65-results             progress markers / one summary line per run (numbers only)
T65     := flow/tsmc65/sync.sh
PERIODS ?= 5.0 4.0 6.67
SCOUT   ?= fma pe sfu bank768 lane gm4
t65_jobs = $(foreach b,$(1),$(foreach p,$(PERIODS),$(b):$(p)))
t65_launch = $(T65) run 'mkdir -p build/tsmc65; nohup flow/tsmc65/syn/run_scout.sh $(call t65_jobs,$(1)) > build/tsmc65/scout_$(2).log 2>&1 < /dev/null & echo launched $(2)'

sync65-pull:
	$(T65) pull

sync65-wip:
	$(T65) wip

sync65-site:
	$(T65) site

syn65-scout:
	$(call t65_launch,$(SCOUT),scout)

syn65-status:
	$(T65) run 'tail -n 3 build/tsmc65/scout_*.log; ps -u $$USER -o pid,etime,args | grep [d]c_shell | grep -v tcsh | cut -c1-60 || true'

syn65-results:
	$(T65) run 'flow/tsmc65/tools/summarize.sh'

syn65-%:
	$(call t65_launch,$*,$*)

## P8b: hierarchical DC of the full16 units at 4.0 ns (flow/tsmc65/common/hier.txt; children first, finished blocks reused):
##   make syn65u-<unit|block> [T65_PERIOD=4.0] [FORCE=1|all]   unit = fp gemm vec lowa mem rng ctrl, or any block of hier.txt
##        [T65_FP=1 T65_TAG=_fp]                  P8c2: floorplanned run of the named block (common/floorplan.txt), own output dir
##   make syn65u-all [UNITS=...]                   every unit in one sequential batch (log build/tsmc65/hier_all.log)
##   make syn65u-status / syn65u-results         progress markers / one summary line per block (numbers only)
##   make sync65-results                         fetch the summary text files to flow/tsmc65/results/
sync65-results:
	$(T65) results

syn65u-status:
	$(T65) run 'tail -n 2 build/tsmc65/hier_*.log; ps -u $$USER -o pid,etime,rss,args | grep [d]c_shell | grep -v tcsh | cut -c1-70 || true'

syn65u-results:
	$(T65) run 'flow/tsmc65/tools/summarize_hier.sh'

UNITS ?= fp ctrl rng mem lowa vec gemm
syn65u-all:
	$(T65) run 'mkdir -p build/tsmc65; T65_PERIOD=$(or $(T65_PERIOD),4.0) FORCE=$(or $(FORCE),0) nohup flow/tsmc65/syn/run_hier.sh $(UNITS) > build/tsmc65/hier_all.log 2>&1 < /dev/null & echo launched all'

syn65u-%:
	$(T65) run 'mkdir -p build/tsmc65; T65_PERIOD=$(or $(T65_PERIOD),4.0) FORCE=$(or $(FORCE),0) T65_FP=$(or $(T65_FP),0) T65_TAG=$(T65_TAG) T65_KID_TAG=$(T65_KID_TAG) nohup flow/tsmc65/syn/run_hier.sh $* > build/tsmc65/hier_$*$(T65_TAG).log 2>&1 < /dev/null & echo launched $*$(T65_TAG)'

## P8d: exit rule, then the chip at the closed clock (flow/tsmc65/run_p8d.sh; one job at a time, log build/tsmc65/p8d.log):
##   make syn65-exit                                vu16 at 5.0 (5.25 if it misses), gm16 at the closed period
##   make p8d65 P8D_PERIOD=5.0 [P8D_STAGES="stale chip fm pt"]   chip DC, Formality, PrimeTime ss/tt/ff + PrimePower
##   make saif65-gm / saif65-vu P8D_PERIOD=5.0 [SAIF_CYC=20000]      RTL SAIF (VCS) of real_dgelu / real_ln_fwd
##   make p8d65 P8D_PERIOD=5.0 P8D_STAGES=ptsaif                  PrimePower of gm16 / vu16 with those SAIFs
P8D_PERIOD ?= 5.0
P8D_STAGES ?= stale chip fm pt
SAIF_CYC ?= 20000
syn65-exit:
	$(T65) run 'mkdir -p build/tsmc65; nohup flow/tsmc65/syn/run_exit.sh > build/tsmc65/hier_exit.log 2>&1 < /dev/null & echo launched exit'

p8d65:
	$(T65) run 'mkdir -p build/tsmc65; T65_SAIF_GM=$$PWD/build/tsmc65/saif/gm_real_dgelu/dut_g.saif T65_SAIF_VU=$$PWD/build/tsmc65/saif/vu_real_ln_fwd/dut_g.saif nohup flow/tsmc65/run_p8d.sh $(P8D_PERIOD) $(P8D_STAGES) > build/tsmc65/p8d.log 2>&1 < /dev/null & echo launched p8d'

saif65-gm:
	$(T65) run 'nohup flow/tsmc65/sim/run_saif.sh gm build/tsmc65/vec/gm real_dgelu $(SAIF_CYC) $(P8D_PERIOD) > build/tsmc65/saif_gm.log 2>&1 < /dev/null & echo launched saif gm'

saif65-vu:
	$(T65) run 'nohup flow/tsmc65/sim/run_saif.sh vu build/tsmc65/vec/vu real_ln_fwd $(SAIF_CYC) $(P8D_PERIOD) > build/tsmc65/saif_vu.log 2>&1 < /dev/null & echo launched saif vu'

synth pnr:
	@echo "make $@: not implemented yet (see PLAN.md phases P2-P8)"; false
