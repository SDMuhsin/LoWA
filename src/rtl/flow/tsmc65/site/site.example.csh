# Site tool environment template (tcsh). Copy to site.csh (git-ignored) and fill in.
# Sourced by common/env.csh: Synopsys DC (+ PrimeTime / PrimePower / Formality for P8c) and T65_KIT.
source <TOOL_SETUP_SCRIPT_DC> >& /dev/null
source <TOOL_SETUP_SCRIPT_PT> >& /dev/null
source <TOOL_SETUP_SCRIPT_FM> >& /dev/null
setenv T65_KIT <STDCELL_KIT_ROOT>
