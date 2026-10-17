#!/usr/bin/env bash
# Server link of the TSMC-65 flow (settings in the git-ignored site/site.sh, template site/site.example.sh).
# Code reaches the server through git: commit and push locally, then `sync.sh pull` fast-forwards the server
# checkout.  Before a push, `sync.sh wip` copies this tree's hw/ and flow/ to a separate work tree on the server
# (T65_WIP; set T65_TREE=wip to run there).  Only scripts and RTL go up; only summary text comes back.
#   sync.sh pull            git pull --ff-only in the server checkout (T65_CHECKOUT)
#   sync.sh wip             rsync src/rtl/{hw,flow} to the server work tree (T65_WIP)
#   sync.sh site            rsync the real site files to the selected tree (T65_TREE)
#   sync.sh results         fetch build/tsmc65/summary/*.txt of the selected tree to flow/tsmc65/results/
#   sync.sh run '<cmd>'     run a command in src/rtl of the selected tree
#   sync.sh tree            print the selected remote src/rtl path
set -eu
T65=$(cd "$(dirname "$0")" && pwd)
RTL=$(cd "$T65/../.." && pwd)
[ -f "$T65/site/site.sh" ] || { echo "missing $T65/site/site.sh (see site/README.md)"; exit 1; }
. "$T65/site/site.sh"
case "${T65_TREE:-checkout}" in
    wip) TREE=$T65_WIP/src/rtl ;;
    *)   TREE=$T65_CHECKOUT/src/rtl ;;
esac
RS="rsync -rltp -e ssh"
up() { "$T65_RUN" -n true; }        # open / reuse the runner's master connection for rsync
case "${1:-}" in
pull)
    "$T65_RUN" -n "cd ~/$T65_CHECKOUT || exit 1; git pull -q --ff-only && git log -1 --format='%h %s'" ;;
wip)
    up; "$T65_RUN" -n "mkdir -p ~/$T65_WIP/src/rtl"
    $RS --delete --exclude __pycache__ --exclude 'flow/tsmc65/results' "$RTL/hw" "$RTL/flow" "$T65_HOST:$T65_WIP/src/rtl/"
    echo "wip tree updated: $T65_WIP/src/rtl" ;;
site)
    up; $RS --exclude '*.example.*' --exclude README.md "$T65/site/" "$T65_HOST:$TREE/flow/tsmc65/site/"
    echo "site config copied to $TREE" ;;
results)
    up; mkdir -p "$T65/results"
    $RS "$T65_HOST:$TREE/build/tsmc65/summary/" "$T65/results/"
    ls "$T65/results" ;;
run)
    "$T65_RUN" -n "cd ~/$TREE || exit 1; $2" ;;
tree)
    echo "$TREE" ;;
*)  echo "usage: $0 pull|wip|site|results|run '<cmd>'|tree"; exit 1 ;;
esac
