/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <stdio.h>

#include "unit_test.h"
#include "rdma/gin/nccl_ofi_gin_allgather.h"
#include "rdma/gin/nccl_ofi_gin_close_progress.h"

int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	unit_test_init();

	std::atomic<bool> stop_progress{false};
	std::mutex progress_error_mutex;
	std::condition_variable worker_started_cv;
	std::mutex worker_started_mutex;
	bool worker_started = false;
	int progress_error = 0;
	int close_progress_error = 0;
	bool all_gather_called = false;

	std::thread progress_thread([&]() {
		{
			std::lock_guard<std::mutex> lock(worker_started_mutex);
			worker_started = true;
		}
		worker_started_cv.notify_one();

		while (!stop_progress.load(std::memory_order_acquire)) {
			std::this_thread::yield();
		}

		/* Model an error published in the last worker iteration after the
		 * caller's initial successful-progress check but before barrier entry. */
		std::lock_guard<std::mutex> lock(progress_error_mutex);
		progress_error = -5;
	});
	{
		std::unique_lock<std::mutex> lock(worker_started_mutex);
		worker_started_cv.wait(lock, [&]() { return worker_started; });
	}

	int ret = nccl_ofi_gin_stop_progress_before_close_barrier(
		stop_progress, progress_thread, progress_error_mutex, progress_error,
		close_progress_error, [&]() {
			all_gather_called = true;
			return 0;
		});

	assert(ret == -5);
	assert(close_progress_error == -5);
	assert(!all_gather_called);

	nccl_ofi_gin_deferred_progress_error deferred_error;
	bool request_live = true;
	bool memory_registered = true;
	int progress_calls = 0;
	int request_polls = 0;
	std::function<int()> failing_progress = [&]() {
		progress_calls++;
		assert(request_live);
		assert(memory_registered);
		return -5;
	};
	int drain_ret = nccl_ofi_gin_drain_all_gather_requests(
		deferred_error, failing_progress, [&]() { return request_live; }, [&]() {
		assert(request_live);
		assert(memory_registered);
		if (++request_polls == 2) {
			request_live = false;
		}
		return 0;
	});
	assert(progress_calls == 2);
	assert(drain_ret == 0);
	assert(!request_live);
	assert(memory_registered);

	/* Registered storage remains live until the drain returns, then the
	 * caller can deregister it and report the saved progress failure. */
	memory_registered = false;
	assert(deferred_error.result(0) == -5);

	printf("GIN close progress barrier ordering tests passed\n");
	return 0;
}
