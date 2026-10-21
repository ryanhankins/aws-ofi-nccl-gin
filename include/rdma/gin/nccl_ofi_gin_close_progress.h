/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_GIN_CLOSE_PROGRESS_H_
#define NCCL_OFI_RDMA_GIN_CLOSE_PROGRESS_H_

#include <atomic>
#include <mutex>
#include <thread>

/**
 * Stop and join close progress before entering the close collective.
 *
 * Joining establishes that progress_error is terminal before it is inspected;
 * the collective is never entered if progress failed.
 */
template <typename AllGatherFn>
int nccl_ofi_gin_stop_progress_before_close_barrier(std::atomic<bool> &stop_progress,
						    std::thread &progress_thread,
						    std::mutex &progress_error_mutex,
						    int &progress_error,
						    int &close_progress_error,
						    AllGatherFn &&all_gather)
{
	stop_progress.store(true, std::memory_order_release);
	progress_thread.join();

	{
		std::lock_guard<std::mutex> lock(progress_error_mutex);
		close_progress_error = progress_error;
	}

	if (close_progress_error != 0) {
		return close_progress_error;
	}

	return all_gather();
}

#endif /* NCCL_OFI_RDMA_GIN_CLOSE_PROGRESS_H_ */
