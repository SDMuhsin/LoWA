# Site configuration (process kit, tools, server)

The scripts under `flow/tsmc65` contain no kit names, kit paths or library cell names. Everything
specific to the kit, the tool installation and the server is read from git-ignored files in this
directory; only the templates (`*.example.*`) and this file are tracked.

| Template (tracked) | Copy to (ignored) | Read by | Holds |
|---|---|---|---|
| `site.example.tcl` | `site.tcl` | `common/site.tcl`, `sdc/gen_sdc.tcl` | library dir, per-corner .db and operating conditions, Milkyway techfile / reference lib and RC files (topographical mode), reference cells (NAND2 for gate equivalents, ICG name pattern, input / clock drivers, load pin), optional dont-use list |
| `site.example.csh` | `site.csh` | `common/env.csh` | tool setup scripts, `T65_KIT` (kit root) |
| `site.example.sh` | `site.sh` | `sync.sh`, Makefile `syn65-*` | remote runner command, ssh host alias, server checkout and work trees |
| `mm_icg_site.example.v` | `mm_icg_site.v` | `common/rtl.tcl` (DC, Formality) | `mm_icg` mapped to the library integrated clock-gating cell; replaces `hw/mem/tech/mm_icg.v` |

Setup on a new site (locally, then `flow/tsmc65/sync.sh site` copies the real files to the server):

    cd src/rtl/flow/tsmc65/site
    for f in tcl csh sh; do cp site.example.$f site.$f; done
    cp mm_icg_site.example.v mm_icg_site.v

and replace every `<PLACEHOLDER>`. Simulations never read these files: they use the behavioural
clock gate `hw/mem/tech/mm_icg.v`.
