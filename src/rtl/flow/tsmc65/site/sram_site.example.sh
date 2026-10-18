# P8e site settings for flow/tsmc65/sram (compiled SRAM / RF macros).  Copy to sram_site.sh (git-ignored).
T65_MC_KIT=<MEMORY_COMPILER_DIR>          # directory holding the memory compilers
T65_MC_JRE=<JAVA8_JRE_DIR>                # a Java 8 runtime (jre dir) of the host
T65_MC_SCRATCH=<LOCAL_SCRATCH_DIR>        # local (non-NFS) scratch for the staged compilers and runs
T65_MC_OUT=<MACRO_VIEW_DIR>               # where the views stay on the server
T65_MC_CORNERS=<C_SS>,<C_TT>,<C_FF_COLD>,<C_FF_HOT>    # exactly four characterisation corners, ss first
T65_MC_EMA_IDX=0                          # margin-adjust setting the summary reports (the site wrapper ties it)
T65_MC_LC_SETUP=<LIBRARY_COMPILER_SETUP_CSH>
# sweep candidates  words:bits:compiler:mux
T65_MC_CANDIDATES="<W>:<B>:<COMPILER>:<MUX> ..."
# chosen macro per bank shape  words:bits:compiler:mux:instance
T65_MC_SHAPES="256:32:<COMPILER>:<MUX>:<INST256> 512:32:<COMPILER>:<MUX>:<INST512> 768:32:<COMPILER>:<MUX>:<INST768>"
