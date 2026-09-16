#!/bin/bash
# Download clips, then submit the preprocessing job when the download ends.
#
#   nohup scripts/overnight.sh > ~/scratch/dl.log 2>&1 &
#
# Deliberately no `set -e`: if the download dies partway we still want to
# preprocess whatever clips completed. preprocess.py skips incomplete clips
# and resumes, so a partial download costs nothing.

CLIPS=${1:-3500}
ROOT=$HOME/scratch

cd $ROOT/genpoint3d
module load python/3.11.7-gcc11
source $ROOT/venvs/fyp/bin/activate

echo "=== DOWNLOAD START $(date) -- $CLIPS clips ==="
python scripts/download_kubric.py --out $ROOT/kubric --clips $CLIPS

echo "=== DOWNLOAD DONE $(date) -- submitting preprocess ==="
qsub -q normal -P personal-yhuang01 scripts/preprocess.pbs
