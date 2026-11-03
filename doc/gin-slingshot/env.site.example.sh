# Example site profile for the validated GH200/Slingshot environment.
# Adapt module names, paths, provider options and Slurm flags for your system.
export ROOT=${ROOT:-$HOME/NCCL/work}
module load GPU/buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 CUDA/12.9.1 GDRCopy/2.5 \
            MPICH/4.3.2-CUDA-12.9.1 hwloc/2.12.1 >/dev/null 2>&1
export CUDA_HOME=$EBROOTCUDA MPI_HOME=$EBROOTMPICH
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v stubs | paste -sd:)
export LD_LIBRARY_PATH=/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD=/usr/lib64/libnvidia-ml.so.1

export FI_PROVIDER=cxi
export FI_CXI_ENABLE_WRITEDATA=1
export FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=2048
export NCCL_CROSS_NIC=1 NCCL_NET_GDR_LEVEL=PHB
export UCX_NET_DEVICES=nsc-eth UCX_TLS=tcp,self,sm

site_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$site_dir/env.sh"

# Site-specific multi-node launcher: epsrun <nodes> <tasks-per-node> <cmd...>.
epsrun() {
	local n=$1 t=$2
	shift 2
	srun --network=disable_rdzv_get --mpi=pmi2 -N"$n" \
	     --ntasks-per-node="$t" --gpus-per-node=4 --gpu-bind=none \
	     ${EP_BIND:---cpu-bind=none} --export=ALL timeout "${TMO:-600}" "$@" </dev/null
}
