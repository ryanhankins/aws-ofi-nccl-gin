#!/bin/bash
# Run the EP CI steps that are only valid inside ONE NVLink domain (single node, 4 ranks, no GIN):
# LL zero-copy, HT expert-major pull-push, MXFP8 combine. usage: run_ep_single.sh [steps...]
# default steps: 40 62 63 64 65 66. Run inside a 1-node allocation (or from salloc).
source "$(dirname "$0")/env.site.example.sh"
unset NCCL_NET_PLUGIN
D=$(cd "$(dirname "$0")" && pwd); mkdir -p $D/logs1
for i in ${@:-40 62 63 64 65 66}; do
  cmd=$(sed -n "${i}p" $D/manifest.txt); envs=""
  env_pattern="^NCCL_EP_[A-Z_0-9]+=[^ ]* "
  while [[ "$cmd" =~ $env_pattern ]]; do
    envs="$envs ${cmd%% *}"
    cmd="${cmd#* }"
  done
  cmd="${cmd/@@/$NCCL_EP_BUILDDIR/test/nccl_ep}"
  env $envs srun -N1 -n4 --gpus-per-node=4 --gpu-bind=none \
      --cpu-bind=none --mem=0 --export=ALL timeout 420 $cmd \
      </dev/null > $D/logs1/step$i.log 2>&1; rc=$?
  echo "$i rc=$rc $envs ${cmd#$NCCL_EP_BUILDDIR/test/nccl_ep/}" | cut -c1-170
done
