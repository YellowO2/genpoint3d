#!/bin/bash
# Run a command on the NSCC login node, through the NTU jump host.
#
#   scripts/nscc.sh qstat -u '$USER'
#   scripts/nscc.sh 'tail -20 ~/scratch/run9_costvol.live.log'
#   scripts/nscc.sh pull          # git pull in the repo
#   scripts/nscc.sh logs          # copy every run's log.json into runs/
#
# Hosts and usernames come from ~/.ssh/config (the `nscc` alias), so nothing
# identifying lives in this repo. Run `scripts/nscc_setup.sh` once to create it.
#
# With an ssh key installed this never prompts. Without one it asks for the two
# passwords in YOUR terminal -- do not put them in a file, and never in a repo.

set -euo pipefail
REPO='$HOME/scratch/genpoint3d'

case "${1:-}" in
  pull) exec ssh nscc "cd $REPO && git pull" ;;
  logs)
    # Every log is named log.json, so a plain scp would overwrite them all into
    # one file. Stream them as a tar instead and name each after its run dir.
    dest=$(cd "$(dirname "$0")/../runs" && pwd)
    ssh nscc "cd $REPO/outputs && tar cf - */log.json" | tar xf - -C "$dest"
    for d in "$dest"/*/; do
      [ -f "$d/log.json" ] || continue
      mv -f "$d/log.json" "$dest/$(basename "$d").json"
      rmdir "$d"
    done
    ls -la "$dest"/*.json
    exit 0 ;;
  "")
    echo "usage: $0 <command> | pull | logs" >&2; exit 2 ;;
esac

exec ssh nscc "$@"
