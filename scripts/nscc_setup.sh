#!/bin/bash
# One-time: create an ssh key and a `nscc` alias, so nothing ever prompts again
# and no password is stored anywhere.
#
#   scripts/nscc_setup.sh <ntu-user>@<jump-host> <nscc-user>@<nscc-host>
#
# You type each password ONCE, into your own terminal, during ssh-copy-id.
# After this, `ssh nscc` works directly and `scripts/nscc.sh` needs no input.

set -euo pipefail
JUMP=${1:?usage: $0 <ntu-user>@<jump-host> <nscc-user>@<nscc-host>}
DEST=${2:?usage: $0 <ntu-user>@<jump-host> <nscc-user>@<nscc-host>}
KEY=~/.ssh/nscc

[ -f "$KEY" ] || ssh-keygen -t ed25519 -f "$KEY" -N "" -C "genpoint3d"

# Idempotent: skip if the alias is already there, so re-running is harmless.
if ! grep -q "^Host nscc$" ~/.ssh/config 2>/dev/null; then
  cat >> ~/.ssh/config <<CFG

Host nscc-jump
    HostName ${JUMP#*@}
    User ${JUMP%@*}
    IdentityFile $KEY

Host nscc
    HostName ${DEST#*@}
    User ${DEST%@*}
    ProxyJump nscc-jump
    IdentityFile $KEY
    ServerAliveInterval 60
CFG
  chmod 600 ~/.ssh/config
  echo "added nscc + nscc-jump to ~/.ssh/config"
fi

# The jump host first, since the second hop is tunnelled through it.
ssh-copy-id -i "$KEY.pub" nscc-jump
ssh-copy-id -i "$KEY.pub" nscc

echo
ssh nscc 'echo "connected as $USER on $(hostname), no password needed"'
