# Source after setting ROOT and loading the required compiler, CUDA, MPI and
# provider libraries for the local site. See env.site.example.sh.
: "${ROOT:?Set ROOT to your installation workspace}"
: "${CUDA_HOME:?Set CUDA_HOME to your CUDA toolkit}"

export NCCL_HOME=${NCCL_HOME:-$ROOT/nccl-build}
export NCCL_EP_BUILDDIR=${NCCL_EP_BUILDDIR:-$ROOT/ext-build}
export NCCL_EP_HOME=${NCCL_EP_HOME:-$NCCL_EP_BUILDDIR}
export OFI_PREFIX=${OFI_PREFIX:-$ROOT/ofi-install}
export LF_PREFIX=${LF_PREFIX:-$ROOT/lf-install}
export LD_LIBRARY_PATH=$NCCL_EP_HOME/lib:$NCCL_HOME/lib:$OFI_PREFIX/lib:\
$LF_PREFIX/lib64:$LF_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

export NCCL_NET_PLUGIN=ofi
export NCCL_GIN_TYPE=2
export OFI_NCCL_PROTOCOL=RDMA
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# NCCL EP JIT-compiles kernels and needs headers from matching installations.
export NCCL_EP_JIT_SOURCE_DIR=$NCCL_EP_BUILDDIR/include/nccl_ep
export NCCL_EP_JIT_BUILD_INCLUDE_DIR=$NCCL_HOME/include
export NCCL_EP_JIT_CUDA_INCLUDE_DIR=$CUDA_HOME/include
export PATH=$CUDA_HOME/bin:$PATH
export EP_TEST=$NCCL_EP_BUILDDIR/test/nccl_ep/ep_test
export EP_BENCH=$NCCL_EP_BUILDDIR/test/nccl_ep/ep_bench
