# L1 hwmodel core.  -ffp-contract=off: no FMA the source did not ask for; -fno-fast-math: no
# reassociation; the source uses intrinsics for every rounded op.
CFLAGS = -O2 -std=gnu11 -fPIC -fopenmp -mavx512f -mfma -ffp-contract=off -fno-fast-math -Wall -Wno-unknown-pragmas
liblwta.so: lwta.c
	gcc $(CFLAGS) -shared -o $@ $< -lm
clean:
	rm -f liblwta.so
