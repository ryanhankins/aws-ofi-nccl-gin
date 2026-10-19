/*
 * SPDX-FileCopyrightText:  Copyright Hewlett Packard Enterprise Development LP
 */

#include "config.h"

#include <algorithm>
#include <thread>

#include "rdma/gin/nccl_ofi_gin_flow_writedata_routed.h"
#include "rdma/gin/nccl_ofi_gin.h"
#include "nccl_ofi_assert.h"

void gin_flow_writedata_routed::init_peer(nccl_ofi_gin_peer_rank_info &peer, uint32_t peer_nranks)
{
	assert_always(peer_nranks == nranks);
	peer.current_window_limit = resources.get_shared_cq_window_limit(nranks);
}

int gin_flow_writedata_routed::prepare_tx(nccl_ofi_gin_peer_rank_info &peer,
			                  gin_tx_flow_decision &decision, gin_flow_context &ctx)
{
	uint32_t outstanding = gin_cursor_delta(peer.tx_head, peer.tx_tail);
	uint32_t current_window_limit = peer.current_window_limit;
	while (OFI_UNLIKELY(outstanding >= current_window_limit)) {
		if (OFI_UNLIKELY(!peer.has_pending_ack_request)) {
			decision.request_ack = true;
			peer.consecutive_puts_without_ack = 0;
			return 0;
		}
		int ret = ctx.progress();
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}
		std::this_thread::yield();
		outstanding = gin_cursor_delta(peer.tx_head, peer.tx_tail);
		current_window_limit = peer.current_window_limit;
	}

	const uint32_t conservative_refresh_window = ctx.conservative_refresh_window;
	const uint32_t ack_req_interval = gin_ack_request_get_interval(current_window_limit);
	const uint32_t ack_req_threshold =
		gin_ack_request_get_threshold(current_window_limit,
			std::min<uint32_t>(current_window_limit, conservative_refresh_window),
			ack_req_interval);
	if (OFI_UNLIKELY(outstanding >= ack_req_threshold)) {
		if (OFI_UNLIKELY(peer.consecutive_puts_without_ack++ >= ack_req_interval)) {
			decision.request_ack = true;
			peer.consecutive_puts_without_ack = 0;
		}
	}

	return 0;
}

void gin_flow_writedata_routed::prepare_ack(
	const nccl_ofi_gin_peer_rank_info &peer,
	uint32_t rx_consumed, gin_ack_msg_t &ack)
{
	gin_fill_ack_header(peer, rx_consumed, ack);
	ack.granted_window = resources.get_shared_cq_window_limit(nranks);
}

void gin_flow_writedata_routed::process_ack(
	nccl_ofi_gin_peer_rank_info &peer, const gin_ack_msg_t &ack)
{
	gin_apply_granted_window(peer, ack.granted_window);
	gin_complete_pending_ack_request(peer, ack.rx_consumed);
}
