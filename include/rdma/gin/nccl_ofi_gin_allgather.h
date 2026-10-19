/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_GIN_ALLGATHER_H
#define NCCL_OFI_RDMA_GIN_ALLGATHER_H

#include <functional>

#include "nccl_ofi.h"

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
	RequestsActiveFn &&requests_active, PollFn &&poll)
{
	while (requests_active()) {
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
	nccl_net_ofi_send_comm *s_comm;
	nccl_net_ofi_recv_comm *r_comm;
	size_t rank;
	size_t nranks;

public:
	nccl_ofi_gin_allgather_comm(nccl_net_ofi_send_comm *s_comm_arg,
				    nccl_net_ofi_recv_comm *r_comm_arg, size_t rank_arg,
				    size_t nranks_arg)
	    : s_comm(s_comm_arg), r_comm(r_comm_arg), rank(rank_arg), nranks(nranks_arg)
	{
	}

	~nccl_ofi_gin_allgather_comm();

	/**
	 * Perform the all_gather operation
	 *
	 * @param data: a buffer of size (size * nranks) bytes
	 * @param size: the size of each data element in bytes
	 * @param progress: optional progress callback; errors are deferred until
	 *                  posted requests complete and the buffer MRs are deregistered
	 *
	 * @return: 0 on success, -errno on failure (including deferred progress errors)
	 */
	int all_gather(void *data, size_t size, std::function<int()> progress = {});
};

#endif /* NCCL_OFI_RDMA_GIN_ALLGATHER_H */
