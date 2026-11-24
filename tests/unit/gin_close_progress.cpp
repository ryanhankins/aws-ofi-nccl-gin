/*
 * Copyright (c) 2026, Hewlett Packard Enterprise Development LP.
 */

#include "config.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <functional>

#include "nccl_ofi_assert.h"
#include "unit_test.h"
#include "rdma/gin/nccl_ofi_gin_allgather.h"

int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	unit_test_init();

	nccl_ofi_gin_deferred_progress_error deferred_error;
	bool request_live = true;
	bool memory_registered = true;
	int progress_calls = 0;
	int request_polls = 0;
	std::function<int()> failing_progress = [&]() {
		progress_calls++;
		assert_always(request_live);
		assert_always(memory_registered);
		return -5;
	};
	int drain_ret = nccl_ofi_gin_drain_all_gather_requests(
		deferred_error,
		failing_progress,
		[&]() { return request_live; },
		[&]() {
			assert_always(request_live);
			assert_always(memory_registered);
			if (++request_polls == 2) {
				request_live = false;
			}
			return 0;
		},
		nccl_ofi_gin_abort_token());
	assert_always(progress_calls == 2);
	assert_always(drain_ret == 0);
	assert_always(!request_live);
	assert_always(memory_registered);

	/* Registered storage remains live until the drain returns, then the
	 * caller can deregister it and report the saved progress failure. */
	memory_registered = false;
	assert_always(deferred_error.result(0) == -5);

	uint32_t abort_flag = 1;
	request_live = true;
	request_polls = 0;
	auto poll_until_aborted = [&]() {
		request_polls++;
		std::atomic_ref<uint32_t>(abort_flag).store(1, std::memory_order_release);
		return 0;
	};
	drain_ret = nccl_ofi_gin_drain_all_gather_requests(
		deferred_error, failing_progress, [&]() { return request_live; },
		poll_until_aborted, nccl_ofi_gin_abort_token(&abort_flag));
	assert_always(drain_ret == -ECANCELED);
	assert_always(request_polls == 0);

	abort_flag = 0;
	memory_registered = true;
	drain_ret = nccl_ofi_gin_drain_all_gather_requests(
		deferred_error, failing_progress, [&]() { return request_live; },
		poll_until_aborted, nccl_ofi_gin_abort_token(&abort_flag));
	assert_always(drain_ret == -ECANCELED);
	assert_always(request_polls == 1);
	assert_always(request_live);

	printf("GIN close progress request draining tests passed\n");
	return 0;
}
