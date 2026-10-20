# NCCL GIN, NCCL EP and M2N on HPE Slingshot (CXI)

How to build and run NCCL's GPU-Initiated Networking (GIN), **NCCL EP** and **M2N**
(`nccl-extensions`) on a Slingshot-11 system, using the `aws-ofi-nccl` plugin
(branch `jeff-cxi-gin-port`) with an **upstream, unpatched libfabric**.

Validated on: 4× GH200 (aarch64) nodes, one Slingshot CXI NIC per GPU (`cxi0`–`cxi3`),
Slurm, CUDA 12.9.1, NCCL 2.32.3. Everything below was run on 2 nodes (8 GPUs).

## 1. How it fits together

```
NCCL 2.32.3  ──loads──▶  libnccl-net-ofi.so (aws-ofi-nccl, branch jeff-cxi-gin-port)
   │  GIN proxy (NCCL_GIN_TYPE=2)              │ RMA plugin API
   │                                           ▼
nccl_ep / nccl_m2n (built against this NCCL)   libfabric (upstream, CXI provider, fi_writedata)
```

* GIN runs in NCCL's **proxy mode** (`NCCL_GIN_TYPE=2`): a CPU proxy thread drives the
  plugin's RMA calls (`fi_writedata`, `fi_read`, `fi_send`).
* Inter-node traffic uses GIN; intra-node traffic uses NVLink/LSA. A single-node run
  **never uses GIN** (see §8 to verify GIN is active).
* No libfabric patches are needed. You need a libfabric that has CXI `fi_writedata`
  support (upstream `ofiwg/libfabric` PR #11684) and you must set
  `FI_CXI_ENABLE_WRITEDATA=1`. The system libfabric on the test machine (2.3.1) lacks
  it, so it is built privately (§4). The plugin has an `RPATH` to this libfabric.

## 2. Prerequisites and site-specific placeholders

* Build **on a compute node** (here: GH200/aarch64, the login node is x86). Use Slurm for
  everything, e.g.:
  ```bash
  srun -A <ACCOUNT> -p <PARTITION> -N1 --gpus-per-node=4 --exclusive -t 30 --pty bash -l
  ```
  (`<ACCOUNT>`, `<PARTITION>` are placeholders — the validated runs used a `gpu` partition.)
* Modules (EasyBuild names on the test system; adapt): GCC 14.3, CUDA 12.9.1,
  GDRCopy 2.5 (≥ 2.5 required by GIN), MPICH 4.3.2 (CUDA-aware), hwloc 2.12.1.
* System libcxi/libcxi-devel (for the CXI provider) and a working `srun`.
* Sources used:

| Component | Source | Version used |
|---|---|---|
| NCCL | `github.com/NVIDIA/nccl` | `v2.32.3-1-6-g3e46c089` (master) |
| libfabric | `github.com/ofiwg/libfabric` | main @ `058a099` (2.7.0a1;<br>CXI `fi_writedata`) |
| aws-ofi-nccl | this repo | branch `jeff-cxi-gin-port` @ `a870c8a` |
| nccl-extensions (EP + M2N) | `github.com/NVIDIA/nccl-extensions` |<br>`6ac183f` + §6.3 test fix |
| nccl-tests (optional, GIN microbenchmarks) | `github.com/NVIDIA/nccl-tests` | `afd59ab` |

The generic `env.sh` uses `ROOT` as the install/build root. Adapt the
site-specific `env.site.example.sh` (modules, provider settings, and Slurm
options) for your system; it sources `env.sh`:

```bash
cd <this directory>; source ./env.site.example.sh # after adapting it for your site
```

## 3. Build NCCL

NCCL must include GIN support; EP and M2N were built and run against this exact build
(2.32.3), **not** the older NCCL pinned as a submodule in `nccl-extensions`.

```bash
source ./env.site.example.sh
cd nccl
make -j32 src.build CUDA_HOME=$CUDA_HOME BUILDDIR=$ROOT/nccl-build \
     NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"      # sm_90 = GH200; adjust for your GPU
```

Optional: nccl-tests (contains the GIN device-API microbenchmarks used as smoke tests):

```bash
cd nccl-tests
make -j16 MPI=1 MPI_HOME=$MPI_HOME NCCL_HOME=$ROOT/nccl-build CUDA_HOME=$CUDA_HOME \
     NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" BUILDDIR=$ROOT/tests-build
# GIN tests: $ROOT/tests-build/device_api/gin/{throughput,latency}/*
# alltoall GIN kernel: alltoall_perf -R 2 -D 3
```

## 4. Build libfabric (CXI, with `fi_writedata`)

```bash
git clone https://github.com/ofiwg/libfabric && cd libfabric && ./autogen.sh
mkdir -p $ROOT/lf-build && cd $ROOT/lf-build
<path-to>/libfabric/configure --prefix=$ROOT/lf-install --libdir=$ROOT/lf-install/lib64 \
    --enable-only --enable-cxi=/usr --enable-tcp --enable-shm --enable-rxm \
    --with-cuda=$CUDA_HOME --enable-cuda-dlopen --enable-dmabuf_peer_mem
make -j32 && make install
```

Check: `grep -r enable_writedata <libfabric>/prov/cxi/src/cxip_dom.c` finds the
`fi_writedata` support. Without it GIN bootstrap fails with
`fi_writedata failed; RC: -38, Error: Function not implemented`.

## 5. Build the aws-ofi-nccl GIN plugin for Slingshot

```bash
git clone <your fork> aws-ofi-nccl && cd aws-ofi-nccl && git checkout jeff-cxi-gin-port
./autogen.sh
mkdir -p $ROOT/ofi-build && cd $ROOT/ofi-build
<path-to>/aws-ofi-nccl/configure --prefix=$ROOT/ofi-install \
    --with-libfabric=$ROOT/lf-install --with-cuda=$CUDA_HOME \
    --with-gdrcopy=$EBROOTGDRCOPY --with-hwloc=$EBROOTHWLOC \
    --disable-tests --disable-picky-compiler --disable-werror
make -j32 && make install
ldd $ROOT/ofi-install/lib/libnccl-net-ofi.so | grep fabric   # must show $ROOT/lf-install
```

(Use `--enable-tests` instead of `--disable-tests` and run `make -C tests/unit check`
to run the plugin's unit tests; this integration build passed all 21 unit tests.)

What the CXI support changes relative to upstream (all gated on `FI_MR_ENDPOINT`
providers, i.e. EFA code paths are unchanged):

* MRs are registered per rail, bound once to the matching GIN endpoint, and enabled.
* Remote-write completions use `fi_writedata` (CXI rejects `FI_REMOTE_CQ_DATA` in
  `fi_writemsg`). Since endpoint-MR CXI completions may lack a usable source address,
  the 64-bit metadata and ACK control messages carry the sender rank; receivers with a
  valid CQ source address continue to use `fi_cq_readfrom`/`FI_SOURCE`.
* Sender windows use receiver-advertised grants derived from configured RX CQ capacity,
  its reserve margin, the live communicator count, and the worst-case peer fan-in.
  CXI additionally applies a conservative software target-EQ budget of 384 events:
  `max(1, floor(384 / (ranks - 1)) - 1)` operations per peer. This is a planning cap,
  not a provider-reported EQ capacity or a universal hardware guarantee; provider CQ
  depth realized at open is not available through the standard post-open query API.
* `FI_MORE` doorbell batching around `fi_writedata` (via `FI_SETOPSFLAG`) is retained
  for large writes. A failed/deferred post drops `FI_MORE` before retry.
* Endpoint-MR teardown cancels and drains posted receives before releasing RX-buffer
  MRs; if bounded draining cannot retire them, endpoints are closed before MR teardown
  to prevent DMA into freed buffers.
* Resource registry ownership remains shared/weak-pointer based; communicator release
  does not directly delete a potentially shared resources object.
* Communicator IDs are 10-bit wire fields. Successfully published IDs are not recycled
  when a communicator closes because the current wire protocol has no generation field and
  close does not prove all late messages have drained. Therefore, a shared endpoint resource
  lifetime supports at most 1024 cumulative GIN communicator creations (not just 1024
  simultaneous communicators); further setup fails explicitly. This inherited capacity limit
  requires owner acceptance or a protocol-level redesign before production use with repeated
  communicator churn. Failed setup before wire publication rolls its ID back safely.

## 6. Build NCCL EP and M2N

### 6.1 Build

`nccl-extensions` normally builds its pinned NCCL submodule; here the build is pointed at
the NCCL from §3 instead. googletest is needed for the unit tests.

```bash
git clone https://github.com/NVIDIA/nccl-extensions && cd nccl-extensions
git submodule update --init third_party/googletest        # NOT third_party/nccl
source $OLDPWD/env.site.example.sh
export NCCL_HOME=$ROOT/nccl-build BUILDDIR=$ROOT/ext-build
G90="-gencode=arch=compute_90,code=sm_90"
make -C nccl_ep -j64 MPI=1 NVCC_GENCODE="$G90" \
    # builds libnccl_ep, ep_test, and ep_bench
make -C nccl_ep/tests -j64 NVCC_GENCODE="$G90"                        # 25 gtest unit binaries
make -C nccl_m2n -j64 NVCC_GENCODE="$G90" MPI_HOME=$MPI_HOME          # libnccl_m2n
make -C nccl_m2n -j64 bench NVCC_GENCODE="$G90" MPI_HOME=$MPI_HOME    # reshard_*bench
make -C nccl_m2n -j64 tests NVCC_GENCODE="$G90" MPI_HOME=$MPI_HOME    # basic_api_test_{local,mpi}
```

Outputs: `$BUILDDIR/lib`, `$BUILDDIR/bin` (M2N tests/benches), `$BUILDDIR/test/nccl_ep`
(`ep_test`, `ep_bench`, unit tests).

### 6.2 Python bindings

Not built or tested here (they need an aarch64 venv with nccl4py/cuda-core/Cython/mpi4py,
and pin `nvidia-nccl 2.30.7`).

### 6.3 One test fix for M2N (non-GIN bug)

`PipeGraphPathTest::initComm` uses the communicator before asynchronous init has finished.
With `NCCL_COMM_BLOCKING=0` this fails with *"Attempt to use communicator before the
previous operation returned ncclSuccess"* — **with or without GIN** (reproduced with
`NCCL_NET=Socket`). Apply the fix before building the M2N tests:

```bash
git am <this directory>/m2n_pipegraph_nonblocking_fix.patch   # or: git apply
```

Without it, 6 `NCCL_COMM_BLOCKING=0` cases (3 per MPI profile) fail.

## 7. Runtime environment

`env.site.example.sh` contains the example system's provider, module, loader,
and Slurm settings; `env.sh` sets common plugin and NCCL paths. Adapt the
site-specific values before sourcing it. **Required** for GIN over CXI:

| Variable | Value | Why |
|---|---|---|
| `NCCL_NET_PLUGIN` | `ofi` | load `libnccl-net-ofi.so` (found via `LD_LIBRARY_PATH`) |
| `NCCL_GIN_TYPE` | `2` | NCCL proxy GIN over the plugin's RMA backend |
| `OFI_NCCL_PROTOCOL` | `RDMA` | the GIN code path lives in the RDMA protocol |
| `FI_PROVIDER` | `cxi` | Site/provider-specific; set in the site example. |
| `FI_CXI_ENABLE_WRITEDATA` | `1` | CXI `fi_writedata`; set in the site example. |
| `LD_PRELOAD` | site-specific | Use the site's real NVML library if needed. |
| `LD_LIBRARY_PATH` | NCCL/plugin/libfabric/ext-build and `/usr/lib64` | Site example removes CUDA stubs; `env.sh` prepends build paths. |
|  |  | Plugin `RPATH` pins libfabric. |
| Site Slurm options | varies | The example launcher uses flags validated on one system. |

NCCL EP additionally JIT-compiles kernels at run time and needs
`NCCL_EP_JIT_SOURCE_DIR=$BUILDDIR/include/nccl_ep`,
`NCCL_EP_JIT_BUILD_INCLUDE_DIR=$NCCL_HOME/include`,
`NCCL_EP_JIT_CUDA_INCLUDE_DIR=$CUDA_HOME/include` (and `nvcc` on `PATH`).

Used in every validated run, but **not** individually proven necessary:
`FI_MR_CACHE_MONITOR=userfaultfd`, `FI_CXI_DISABLE_HOST_REGISTER=1`,
`FI_CXI_DEFAULT_CQ_SIZE=131072`, `FI_CXI_DEFAULT_TX_SIZE=2048`, `NCCL_CROSS_NIC=1`,
`NCCL_NET_GDR_LEVEL=PHB`, `UCX_NET_DEVICES=nsc-eth`, `UCX_TLS=tcp,self,sm`.

Other notes:

* GIN needs **≥ 2 nodes** (the CXI VNI/service is only available to multi-node jobs). For
  single-node tests do `unset NCCL_NET_PLUGIN`; otherwise the plugin logs "ofi domain
  creation failed" and falls back harmlessly.
* Use `srun --mem=0` for steps that allocate a lot of host memory (M2N reduced-bucket
  cases were OOM-killed by Slurm's step memory limit otherwise).
* Debugging: `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV`.

## 8. Run the tests

Allocate 2 nodes first:

```bash
salloc -A <ACCOUNT> -p <PARTITION> -N2 --gpus-per-node=4 --exclusive -t 30
source ./env.site.example.sh
```

### 8.1 Verify GIN is actually in use

Compare per-NIC TX packets before/after a 2-node run (all 8 NICs must increase):

```bash
snap() {
  for h in $(scontrol show hostnames $SLURM_NODELIST); do
    srun -N1 -n1 -w "$h" --overlap bash -c \
      'for n in 0 1 2 3; do \
        echo -n "$(hostname):cxi$n "; \
        cat /sys/class/cxi/cxi$n/device/telemetry/hni_pkts_sent_by_tc_* | \
          cut -d@ -f1 | paste -sd+ | bc; \
      done'
  done
}
snap > before.txt
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET epsrun 2 4 $EP_BENCH --algorithm high-throughput \
   --layout fl --tokens 4096 --hidden 7168 --top-k 8 --experts 256 \
   --validate 2>&1 | tee run.log | \
   grep -E "GIN/Plugin|Global validation"
snap > after.txt; paste before.txt after.txt | awk '{printf "%s delta=%d\n",$1,$4-$2}'
```

Expect the GIN plugin load message and a skip message indicating that the
external proxy plugin is skipped and NCCL GIN proxy uses the RMA backend
Libfabric. Packet counts should be in the millions per NIC (measured: HT
≈ 7.7 M, LL ≈ 0.9 M, M2N ≈ 0.3 M per NIC).
With `NCCL_GIN_TYPE=0` the same EP run fails and the NICs see ~no traffic.

### 8.2 GIN microbenchmarks (nccl-tests, optional smoke test)

```bash
B=$ROOT/tests-build/device_api/gin
srun --network=disable_rdzv_get --mpi=pmi2 -N2 --ntasks-per-node=1 \
     --gpus-per-task=1 --gpu-bind=closest \
     $B/throughput/ginPutBW_perf -b 8 -e 1M -f 8 --gin_op put_signal
srun ... -N2 --ntasks-per-node=4 ... \
     $ROOT/tests-build/alltoall_perf -b 8 -e 8M -f 2 -R 2 -D 3 -g 1  # 8 ranks
```

### 8.3 NCCL EP — unit tests (single node, 4 GPUs; **does not exercise GIN**)

```bash
srun -N1 -n1 --mem=0 --cpu-bind=none --export=ALL bash -c \
  "unset NCCL_NET_PLUGIN; NCCL_HOME=$NCCL_HOME NCCL_EP_BUILDDIR=$NCCL_EP_BUILDDIR \
   bash <nccl-extensions>/nccl_ep/tests/run_tests.sh 4"
```

Result: 81 suites passed, none skipped.

### 8.4 NCCL EP — CI matrix across 2 nodes (**exercises GIN**)

The repo's `nccl_ep/ci/run_nccl_ep_ci_tests.sh` runs each step on **one** node, which
never touches GIN. `manifest.txt` (74 steps) is the same step list; regenerate it
with the CI script and a stub `srun` that records the command line instead of running it:

```bash
mkdir -p fake/test/nccl_ep stubbin
touch fake/test/nccl_ep/{ep_test,ep_bench}
chmod +x fake/test/nccl_ep/*
cp srun_stub.sh stubbin/srun; export PATH=$PWD/stubbin:$PATH
NP=4 NCCL_HOME=/x NCCL_EP_HOME=$PWD/fake NCCL_EP_BENCH_HT=1 \
   bash <nccl-extensions>/nccl_ep/ci/run_nccl_ep_ci_tests.sh 2>/dev/null | grep '^@@' > manifest.txt
```

Then run on 8 ranks (2 nodes × 4). Steps **40, 62–66** (LL zero-copy, HT expert-major
pull-push, MXFP8 combine ×3) are single-NVLink-domain-only by design and are not valid on
2 nodes — run them on one node (4 ranks, `unset NCCL_NET_PLUGIN`):

```bash
# 68 steps, 2 nodes (default CPU binding)
bash run_ep_matrix.sh "40 62 63 64 65 66"
EP_BIND="-c 72 --cpu-bind=cores" bash run_ep_matrix.sh "40 62 63 64 65 66"   # same, tuned binding
# The 6 single-node-only steps, 4 ranks
bash run_ep_single.sh
```

Pass criterion: every `--validate` step prints `Global validation: Dispatch=PASSED,
Combine=PASSED`; `ep_test` prints success on all ranks. Result: **68/68** (2 nodes, both
binding modes) + 6/6 single-node = 74/74; a full single-node rerun of all steps: 69/69.
The matrix takes roughly 30 min (≈20–28 s per step); split it over several jobs if your time
limit is short (the runner skips what you list; keep a `results.txt` between jobs).

### 8.5 M2N

Single node (**does not exercise GIN**), 4 GPUs: 74 pass, 307 skipped (they need > 4 GPUs):

```bash
srun -N1 -n1 --mem=0 --export=ALL bash -c \
     "unset NCCL_NET_PLUGIN; $BUILDDIR/bin/basic_api_test_local -N 4"
```

MPI profiles across 2 nodes (**exercise GIN**) use the repo's
`nccl_m2n/ci/run_nccl_m2n_mpi_tests.sh`, with these changes (patch provided): `--mpi=pmi2
--network=disable_rdzv_get`, `--mem=0`, a 1700 s per-case timeout, the 8-rank
"all algorithms" case split into 4 gtest shards (it exceeds a single 10-minute step), and
resume markers so a long run can span several jobs:

```bash
patch -o m2n_mpi.sh <nccl-extensions>/nccl_m2n/ci/run_nccl_m2n_mpi_tests.sh m2n_mpi_slingshot.patch
export NCCL_HOME=$BUILDDIR M2N_MARK_DIR=$PWD/m2n_marks \
       M2N_BUDGET=1400  # NCCL_HOME contains bin/basic_api_test_mpi
for prof in "four_rank 2" "eight_rank 4"; do set -- $prof
  NCCL_M2N_MPI_TEST_PROFILE=$1 NCCL_M2N_MPI_NNODES=2 NCCL_M2N_MPI_NTASKS_PER_NODE=$2 bash m2n_mpi.sh
done
# re-run (same M2N_MARK_DIR) until it prints "SUMMARY: FAILS=0" and no case is "[budget] deferring"
```

Result: all cases pass for both profiles (default matrix of 282, cross-dimension, PIPE LSA
fanout reuse, PACK split/group, `reshard_bench` "VALIDATION PASSED"), including the three
`NCCL_COMM_BLOCKING=0` cases per profile once §6.3 is applied. A CTA-count sweep
(`NCCL_RESHARD_NUM_CTAS` ∈ {1,4,5,6,8,9,16,32}, `basic_api_test_local -N 4 --filter
tiny_contribution`, single node) passes at all values.

## 9. Performance notes

* **Always bind cores:** `srun -c 72 --cpu-bind=cores` (instead of `--cpu-bind=none`)
  improves NCCL EP by 15–22 % and GIN small-message rate by ~70 %. e.g. EP HT dispatch,
  4096 tokens bf16: 11.0–11.7 ms → 9.5–9.8 ms; LL 4096 dispatch 14.4–15.2 → 11.8 ms.
* No measurable effect (keep defaults): `NCCL_GIN_NCONNECTIONS`, `NCCL_EP_TOKENS_PER_CHUNK`,
  EP dispatch/combine SM counts, `NCCL_GIN_PROXY_QUEUE_SIZE`, `NCCL_GIN_PROXY_POLL_BATCH`,
  `FI_CXI_RX_MATCH_MODE=hybrid`, rendezvous variables.
* `NCCL_GIN_PROXY_NTHREADS` > 1 is **worse** (e.g. HT combine 3.4 → 11 ms).
* Reference numbers (8 ranks, 2 nodes, tuned binding): HT 4096 dispatch ≈ 9.5–9.9 ms /
  combine ≈ 3.4 ms; LL128 dispatch ≈ 540 µs / combine ≈ 700 µs; LL4096 dispatch ≈ 11.8 ms /
  combine ≈ 16.4 ms. HT dispatch reaches about 56 % of NIC line rate.

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `fi_writedata failed; RC: -38` | Build CXI `fi_writedata` support (§4); enable it in CXI. |
| `ofi domain creation failed` (single node) | no VNI on 1-node jobs; `unset NCCL_NET_PLUGIN` |
| NVML errors / wrong GPU info | Remove CUDA stubs; preload `/usr/lib64/libnvidia-ml.so.1`. |
| Writes fail or hang | CXI EQ overflow; use the plugin's per-peer GIN grant cap. |
| Hang/abort at communicator destroy | teardown fixes are in this branch (§5) |
| M2N "previous operation" error | Test race, not GIN; apply §6.3. |
| M2N case killed (OOM) | add `--mem=0` to `srun` |
| `LD_LIBRARY_PATH` does not change libfabric | Plugin `RPATH` pins it; use `LD_PRELOAD`. |

## 11. Not covered

* **EFA** was not tested on hardware (none available). The CXI changes are gated on
  `FI_MR_ENDPOINT`, so the EFA code path is unchanged by construction.
* Python bindings and `ep_test.py` (§6.2).
* M2N single-node cases that need > 4 GPUs are skipped by the test binary; the 8-rank
  MPI profile covers those scenarios across 2 nodes.
* Scale beyond 2 nodes (8 GPUs).

## Files in this directory

| File | Purpose |
|---|---|
| `env.sh` | generic plugin/NCCL paths and runtime setup |
| `env.site.example.sh` | example module, provider, loader, and `epsrun` settings |
| `run_ep_matrix.sh`, `manifest.txt` | runs the 74-step EP CI matrix on 8 ranks (2 nodes) |
| `run_ep_single.sh` | runs the 6 single-node-only EP steps on 4 ranks |
| `srun_stub.sh` | stub `srun` used to regenerate `manifest.txt` from the repo's CI script |
| `m2n_mpi_slingshot.patch` | adapts the M2N MPI CI runner (pmi2, `--mem=0`, shards, resume) |
| `m2n_pipegraph_nonblocking_fix.patch` | fixes the M2N test init race (§6.3) |
