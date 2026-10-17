# Site settings template for flow/tsmc65/sync.sh and the Makefile syn65-* targets. Copy to site.sh (git-ignored).
T65_RUN=<REMOTE_RUNNER>               # command that runs one shell command on the server, e.g. an ssh wrapper
T65_HOST=<SSH_HOST_ALIAS>             # ssh host alias for rsync (the runner's master connection is reused)
T65_CHECKOUT=<CLONE_DIR>              # repository clone on the server, relative to the remote home (git pull)
T65_WIP=<WORK_DIR>                    # pre-push work tree on the server (rsync of hw/ + flow/), relative to home
T65_TREE=checkout                     # which src/rtl the syn65-* targets run in: checkout | wip
