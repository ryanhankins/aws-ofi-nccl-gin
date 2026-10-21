/*
 * SPDX-FileCopyrightText:  Copyright Hewlett Packard Enterprise Development LP
 */

#include "config.h"

#include <thread>

#include "rdma/gin/nccl_ofi_gin_flow_posted_recv.h"
#include "rdma/gin/nccl_ofi_gin.h"

void gin_flow_posted_recv::init_peer(nccl_ofi_gin_peer_rank_info &peer, uint32_t nranks)
{
	(void)nranks;
	peer.current_window_limit = GIN_IMM_SEQ_MASK + 1;
}

int gin_flow_posted_recv::prepare_tx(
	nccl_ofi_gin_peer_rank_info &peer,
	gin_tx_flow_decision &decision,
	gin_flow_context &ctx)
{
	uint32_t outstanding = gin_cursor_delta(peer.tx_head, peer.tx_tail);
	while (OFI_UNLIKELY(outstanding >= (uint32_t)(GIN_IMM_SEQ_MASK + 1))) {
		int ret = ctx.progress();
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}
		std::this_thread::yield();
		outstanding = gin_cursor_delta(peer.tx_head, peer.tx_tail);
	}

	if (OFI_UNLIKELY(outstanding >= GIN_ACK_REQ_THRESHOLD)) {
		if (OFI_UNLIKELY(peer.consecutive_puts_without_ack++ >= GIN_ACK_INTERVAL)) {
			decision.request_ack = true;
			peer.consecutive_puts_without_ack = 0;
		}
	}

	return 0;
}

void gin_flow_posted_recv::prepare_ack(
	const nccl_ofi_gin_peer_rank_info &peer,
	uint32_t rx_consumed, gin_ack_msg &ack)
{
	gin_fill_ack_header(peer, rx_consumed, ack);
}

void gin_flow_posted_recv::process_ack(
	nccl_ofi_gin_peer_rank_info &peer, const gin_ack_msg &ack)
{
	gin_complete_pending_ack_request(peer, ack.rx_consumed);
}
