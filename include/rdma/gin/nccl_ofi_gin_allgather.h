/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_GIN_ALLGATHER_H
#define NCCL_OFI_RDMA_GIN_ALLGATHER_H

#include <memory>
#include <functional>

#include "nccl_ofi.h"
#include "nccl_ofi_gin_base.h"

/**
 * Closes a bootstrap ring comm. The two halves of the ring are owned through
 * this from the moment connect() establishes them, so a setup failure at any
 * point releases them exactly once.
 */
struct nccl_ofi_gin_ring_comm_closer {
	void operator()(nccl_net_ofi_send_comm *comm) const;
	void operator()(nccl_net_ofi_recv_comm *comm) const;
};

using nccl_ofi_gin_ring_send_comm =
	std::unique_ptr<nccl_net_ofi_send_comm, nccl_ofi_gin_ring_comm_closer>;
using nccl_ofi_gin_ring_recv_comm =
	std::unique_ptr<nccl_net_ofi_recv_comm, nccl_ofi_gin_ring_comm_closer>;

class nccl_ofi_gin_deferred_progress_error {
private:
	int error = 0;

public:
	template <typename ProgressFn>
	void run(ProgressFn &&progress)
	{
		int ret = progress();
		if (error == 0 && ret != 0) {
			error = ret;
		}
	}

	int result(int operation_error) const
	{
		return error != 0 ? error : operation_error;
	}

	bool failed() const
	{
		return error != 0;
	}
};

template <typename ProgressFn, typename RequestsActiveFn, typename PollFn>
int nccl_ofi_gin_drain_all_gather_requests(
	nccl_ofi_gin_deferred_progress_error &progress_error, ProgressFn &&progress,
	RequestsActiveFn &&requests_active, PollFn &&poll, nccl_ofi_gin_abort_token abort = {})
{
	while (requests_active()) {
		if (OFI_UNLIKELY(abort.requested())) {
			return -ECANCELED;
		}
		if (progress) {
			progress_error.run(progress);
		}

		int ret = poll();
		if (ret != 0 && !progress_error.failed()) {
			return ret;
		}
	}

	return 0;
}

/**
 * Communicator to perform a ring-based allgather.
 */
class nccl_ofi_gin_allgather_comm {
private:
	/* Deregisters one of all_gather()'s buffer registrations on every exit.
	 * deregMr() needs the comm the buffer was registered on as well as the
	 * handle, and a unique_ptr deleter only receives the handle, so it
	 * carries the comm. */
	struct mr_deregistrar {
		void *comm;
		void operator()(nccl_net_ofi_mr_handle_t *mhandle) const;
	};

	using mr_ptr = std::unique_ptr<nccl_net_ofi_mr_handle_t, mr_deregistrar>;

	nccl_ofi_gin_ring_send_comm s_comm;
	nccl_ofi_gin_ring_recv_comm r_comm;
	size_t rank;
	size_t nranks;
	nccl_ofi_gin_abort_token abort;

public:
	nccl_ofi_gin_allgather_comm(nccl_ofi_gin_ring_send_comm s_comm_arg,
				    nccl_ofi_gin_ring_recv_comm r_comm_arg, size_t rank_arg,
				    size_t nranks_arg, nccl_ofi_gin_abort_token abort_arg = {})
	    : s_comm(std::move(s_comm_arg)), r_comm(std::move(r_comm_arg)), rank(rank_arg),
	      nranks(nranks_arg), abort(abort_arg)
	{
	}

	/**
	 * Perform the all_gather operation
	 *
	 * @param data: a buffer of size (size * nranks) bytes
	 * @param size: the size of each data element in bytes
	 * @param progress: optional progress callback; errors are deferred until
	 *                  posted requests complete and the buffer MRs are deregistered
	 *
	 * @return: 0 on success, -ECANCELED if the communicator aborted while
	 *          waiting, -errno on failure
	 */
	int all_gather(void *data, size_t size, std::function<int()> progress = {});

	nccl_ofi_gin_abort_token get_abort_token() const
	{
		return abort;
	}
};

#endif /* NCCL_OFI_RDMA_GIN_ALLGATHER_H */
