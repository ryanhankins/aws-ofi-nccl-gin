#!/bin/bash
# Run the NCCL EP CI matrix on 2 nodes x 4 GPUs (8 ranks) inside an existing 2-node allocation.
# usage: run_ep_matrix.sh [skip-list]   e.g. run_ep_matrix.sh "40 62 63 64 65 66"
# Needs: manifest.txt next to this script (see README, "Generate the EP step list").
source "$(dirname "$0")/env.site.example.sh"
D=$(cd "$(dirname "$0")" && pwd); LOG=$D/logs; mkdir -p $LOG; : > $D/results.txt
SKIP=" ${1:-} "
n=$(wc -l < $D/manifest.txt)
for i in $(seq 1 $n); do
  [[ "$SKIP" == *" $i "* ]] && { echo "$i skipped"; continue; }
  cmd=$(sed -n "${i}p" $D/manifest.txt); envs=""
  env_pattern="^NCCL_EP_[A-Z_0-9]+=[^ ]* "
  while [[ "$cmd" =~ $env_pattern ]]; do
    envs="$envs ${cmd%% *}"
    cmd="${cmd#* }"
  done
  cmd="${cmd/@@/$NCCL_EP_BUILDDIR/test/nccl_ep}"
  env $envs TMO=420 bash -c "source $D/env.site.example.sh; epsrun 2 4 $cmd" > $LOG/step$i.log 2>&1; rc=$?
  echo "$i rc=$rc $envs ${cmd#$NCCL_EP_BUILDDIR/test/nccl_ep/}" \
      | cut -c1-170 | tee -a $D/results.txt
done
passed=$(grep -c ' rc=0 ' "$D/results.txt")
failed=$(grep -vc ' rc=0 \|skipped' "$D/results.txt")
echo "passed: $passed / $n (failures: $failed)"
