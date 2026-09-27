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
REPO=~/scratch/genpoint3d

case "${1:-}" in
  pull) exec ssh nscc "cd $REPO && git pull" ;;
  logs)
    # -O: some jump-host setups reject sftp, which scp now uses by default.
    exec scp -O "nscc:$REPO/outputs/*/log.json" "$(dirname "$0")/../runs/" ;;
  "")
    echo "usage: $0 <command> | pull | logs" >&2; exit 2 ;;
esac

exec ssh nscc "$@"
