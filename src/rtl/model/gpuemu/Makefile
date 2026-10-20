# GPU port of the L1 hwmodel (P9d).  -fmad=false: no contraction the source did not ask for;
# -ftz=true: the chip's DAZ/FTZ (proven against L1 by check_gpu.py, not assumed).
NVCC ?= /usr/local/cuda/bin/nvcc
liblwta_gpu.so: lwta_gpu.cu
	$(NVCC) -O3 -arch=sm_86 -ftz=true -fmad=false -prec-div=true -prec-sqrt=true -Xcompiler -fPIC -shared -o $@ $< -lcudart
clean:
	rm -f liblwta_gpu.so
