#!/bin/bash
# stub: print the command (with any NCCL_EP_* env the CI exported) instead of running it
while [[ "$1" == -* ]]; do case "$1" in -p|-t|-n|-N) shift;; esac; shift; done
bin="$1"; shift
pre=""
for v in $(compgen -e | grep '^NCCL_EP_' | \
    grep -v -e '^NCCL_EP_HOME$' -e '^NCCL_EP_BENCH_HT$'); do
  pre="$pre$v=${!v} "
done
echo "${pre}@@/$(basename "$bin") $*"
