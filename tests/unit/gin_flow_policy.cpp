/*
 * Copyright (c) 2026, Hewlett Packard Enterprise Development LP.
 */

#include "config.h"

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <mutex>

#include "nccl_ofi_assert.h"
#include "unit_test.h"
#include "rdma/gin/nccl_ofi_gin_flow_posted_recv.h"
#include "rdma/gin/nccl_ofi_gin_flow_writedata_routed.h"
#include "rdma/gin/nccl_ofi_gin_resources.h"

/* Progress hook state. The policies only reach the communicator through
 * gin_flow_context::progress(), so a fake hook can simulate ACK arrival
 * or inject an abort/error while a sender waits for window space. */
struct fake_progress_state {
	nccl_ofi_gin_peer_rank_info *peer = nullptr;
	std::mutex *ep_lock = nullptr;
	int calls = 0;
	int inject_ret = 0;
	uint32_t advance_tail = 0;
	bool lock_released = false;
};

static fake_progress_state progress_state;

static int fake_progress(nccl_ofi_rdma_gin_put_comm &)
{
	progress_state.calls++;
	/* Other communicators must be able to use the endpoint while TX waits. */
	if (progress_state.ep_lock->try_lock()) {
		progress_state.lock_released = true;
		progress_state.ep_lock->unlock();
	}
	if (progress_state.inject_ret != 0) {
		return progress_state.inject_ret;
	}
	progress_state.peer->tx_tail =
		(progress_state.peer->tx_tail + progress_state.advance_tail) & GIN_RX_CONSUMED_MASK;
	return 0;
}

/* Stand-in object for the communicator and resources references. The fake
 * progress hook ignores its argument, and prepare_tx()/process_ack() do not
 * touch the resources, so neither is ever accessed through these. */
alignas(std::max_align_t) static unsigned char placeholder[1];

static gin_flow_context make_ctx(uint32_t refresh_window)
{
	gin_flow_context ctx;
	ctx.conservative_refresh_window = refresh_window;
	ctx.progress_comm = reinterpret_cast<nccl_ofi_rdma_gin_put_comm *>(placeholder);
	ctx.progress_fn = fake_progress;
	return ctx;
}

static void reset_progress(nccl_ofi_gin_peer_rank_info &peer, std::mutex &lock)
{
	progress_state = fake_progress_state();
	progress_state.peer = &peer;
	progress_state.ep_lock = &lock;
}

static void test_posted_recv()
{
	gin_flow_posted_recv policy;
	std::mutex lock;
	gin_flow_context ctx = make_ctx(GIN_IMM_SEQ_MASK + 1);
	const uint32_t seq_window = GIN_IMM_SEQ_MASK + 1;

	/* Below the window: no wait, no ACK request. */
	{
		nccl_ofi_gin_peer_rank_info peer = {};
		policy.init_peer(peer, 4);
		reset_progress(peer, lock);
		std::unique_lock<std::mutex> ep_lock(lock);
		gin_tx_flow_decision decision;
		assert_always(policy.prepare_tx(peer, decision, ctx, ep_lock) == 0);
		assert_always(progress_state.calls == 0);
		assert_always(!decision.request_ack);
	}

	/* Full window: progress until an ACK frees a slot, dropping ep_lock
	 * while progressing and holding it again on return. */
	{
		nccl_ofi_gin_peer_rank_info peer = {};
		policy.init_peer(peer, 4);
		peer.tx_head = seq_window;
		reset_progress(peer, lock);
		progress_state.advance_tail = 1;
		std::unique_lock<std::mutex> ep_lock(lock);
		gin_tx_flow_decision decision;
		assert_always(policy.prepare_tx(peer, decision, ctx, ep_lock) == 0);
		assert_always(progress_state.calls == 1);
		assert_always(progress_state.lock_released);
		assert_always(ep_lock.owns_lock());
	}

	/* Abort while waiting for window space: the error propagates and the
	 * caller still owns ep_lock. */
	{
		nccl_ofi_gin_peer_rank_info peer = {};
		policy.init_peer(peer, 4);
		peer.tx_head = seq_window;
		reset_progress(peer, lock);
		progress_state.inject_ret = -ECANCELED;
		std::unique_lock<std::mutex> ep_lock(lock);
		gin_tx_flow_decision decision;
		assert_always(policy.prepare_tx(peer, decision, ctx, ep_lock) == -ECANCELED);
		assert_always(progress_state.calls == 1);
		assert_always(ep_lock.owns_lock());
		assert_always(peer.tx_tail == 0);
	}

	/* A covering ACK clears the pending request; an older one does not. */
	{
		nccl_ofi_gin_peer_rank_info peer = {};
		peer.has_pending_ack_request = true;
		peer.last_ack_requested_seq = 100;
		gin_ack_msg ack = {};
		ack.rx_consumed = 50;
		policy.process_ack(peer, ack, false);
		assert_always(peer.has_pending_ack_request);
		ack.rx_consumed = 101;
		policy.process_ack(peer, ack, true);
		assert_always(!peer.has_pending_ack_request);
	}
}

static void test_writedata_routed()
{
	const uint32_t nranks = 2;
	gin_flow_writedata_routed policy(*reinterpret_cast<nccl_ofi_gin_resources *>(placeholder),
					 nranks);
	std::mutex lock;
	gin_flow_context ctx = make_ctx(4);

	/* Full grant with no ACK request in flight: request one instead of
	 * waiting, so the receiver can open the window. */
	{
		nccl_ofi_gin_peer_rank_info peer = {};
		peer.current_window_limit = 4;
		peer.tx_head = 4;
		reset_progress(peer, lock);
		std::unique_lock<std::mutex> ep_lock(lock);
		gin_tx_flow_decision decision;
		assert_always(policy.prepare_tx(peer, decision, ctx, ep_lock) == 0);
		assert_always(decision.request_ack);
		assert_always(progress_state.calls == 0);
	}

	/* Full grant with an ACK request pending: wait for the ACK. */
	{
		nccl_ofi_gin_peer_rank_info peer = {};
		peer.current_window_limit = 4;
		peer.tx_head = 4;
		peer.has_pending_ack_request = true;
		reset_progress(peer, lock);
		progress_state.advance_tail = 2;
		std::unique_lock<std::mutex> ep_lock(lock);
		gin_tx_flow_decision decision;
		assert_always(policy.prepare_tx(peer, decision, ctx, ep_lock) == 0);
		assert_always(progress_state.calls == 1);
		assert_always(progress_state.lock_released);
		assert_always(ep_lock.owns_lock());
	}

	/* Abort while waiting for the ACK. */
	{
		nccl_ofi_gin_peer_rank_info peer = {};
		peer.current_window_limit = 4;
		peer.tx_head = 4;
		peer.has_pending_ack_request = true;
		reset_progress(peer, lock);
		progress_state.inject_ret = -ECANCELED;
		std::unique_lock<std::mutex> ep_lock(lock);
		gin_tx_flow_decision decision;
		assert_always(policy.prepare_tx(peer, decision, ctx, ep_lock) == -ECANCELED);
		assert_always(ep_lock.owns_lock());
	}

	/* Grants: a reordered older ACK cannot widen the window, a newer one
	 * can, and every grant is clamped to the CXI event-queue limit. */
	{
		nccl_ofi_gin_peer_rank_info peer = {};
		peer.current_window_limit = 10;
		gin_ack_msg ack = {};
		ack.granted_window = 100;
		policy.process_ack(peer, ack, false);
		assert_always(peer.current_window_limit == 10);
		policy.process_ack(peer, ack, true);
		assert_always(peer.current_window_limit == 100);
		ack.granted_window = GIN_ACK_GRANT_MAX;
		policy.process_ack(peer, ack, true);
		assert_always(peer.current_window_limit == gin_cxi_eq_get_window_limit(nranks));
		ack.granted_window = 5;
		policy.process_ack(peer, ack, false);
		assert_always(peer.current_window_limit == 5);
	}
}

static void test_recv_drain()
{
	int polls = 0;
	int outstanding = 3;
	auto has_outstanding = [&]() { return outstanding > 0; };

	/* Cancellation completes after a few polls. */
	assert_always(gin_recv_reqs_drain(10, has_outstanding, [&]() {
			      polls++;
			      outstanding--;
			      return 1;
		      }) == 0);
	assert_always(polls == 3);

	/* Nothing outstanding: no polling. */
	polls = 0;
	assert_always(gin_recv_reqs_drain(10, has_outstanding, [&]() {
			      polls++;
			      return 0;
		      }) == 0);
	assert_always(polls == 0);

	/* Cancellation never completes: bounded, then -ETIMEDOUT. */
	outstanding = 1;
	assert_always(gin_recv_reqs_drain(50, has_outstanding, [&]() {
			      polls++;
			      return 0;
		      }) == -ETIMEDOUT);
	assert_always(polls == 50);

	/* CQ error while draining is returned immediately. */
	polls = 0;
	assert_always(gin_recv_reqs_drain(50, has_outstanding, [&]() {
			      polls++;
			      return -EIO;
		      }) == -EIO);
	assert_always(polls == 1);

	/* Last allowed poll retires the final receive. */
	polls = 0;
	assert_always(gin_recv_reqs_drain(1, has_outstanding, [&]() {
			      polls++;
			      outstanding = 0;
			      return 1;
		      }) == 0);
	assert_always(polls == 1);
}

int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	unit_test_init();

	test_posted_recv();
	test_writedata_routed();
	test_recv_drain();

	printf("PASS: GIN flow policy and receive drain behavior\n");
	return 0;
}
