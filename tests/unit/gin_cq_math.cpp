/*
 * SPDX-FileCopyrightText:  Copyright Hewlett Packard Enterprise Development LP
 */

#include "config.h"

#include <stdlib.h>
#include <stdio.h>

#include "unit_test.h"
#include "rdma/gin/nccl_ofi_gin_types.h"

struct cq_case {
	uint32_t nranks;
	size_t cq_size;
	size_t ref_cnt;
	bool expect_escape_hatch_safe;
};

struct ack_case {
	uint32_t current_window;
	uint32_t refresh_window;
};

struct grant_case {
	uint32_t start_limit;
	uint32_t start_hysteresis;
	uint32_t granted;
	uint32_t expect_limit;
	uint32_t expect_hysteresis;
};

int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	setenv("OFI_NCCL_GIN_RX_CQ_RESERVE_MARGIN", "17", 0);
	unit_test_init();
	const size_t configured_reserve = ofi_nccl_gin_rx_cq_reserve_margin();
	if (gin_rx_cq_get_reserve_entries(3) !=
	    gin_rx_cq_get_reserve_entries(3, configured_reserve) ||
	    gin_target_cq_get_size(4, 16) !=
		    gin_target_cq_get_size(4, 16, configured_reserve)) {
		printf("CQ reserve defaults do not use the configured parameter\n");
		return 1;
	}

	/* Source ranks are carried in endpoint-MR wire/CQ data. Validate the
	 * full value before narrowing it to uint32_t or using it as an index. */
	const struct {
		uint64_t peer_rank;
		uint32_t nranks;
		bool expected;
	} peer_rank_cases[] = {
		{0, 1, true},
		{3, 4, true},
		{4, 4, false},
		{UINT32_MAX, 4, false},
		{UINT64_MAX, 4, false},
		{0, 0, false},
	};
	for (const auto &tc : peer_rank_cases) {
		if (gin_peer_rank_is_valid(tc.peer_rank, tc.nranks) != tc.expected) {
			printf("Peer-rank bounds check failed: rank=%llu nranks=%u\n",
			       static_cast<unsigned long long>(tc.peer_rank), tc.nranks);
			return 1;
		}
	}
	const uint64_t max_wire_ranks = 1ULL << GIN_MSG_SRC_RANK_BITS;
	if (gin_rank_count_is_valid(max_wire_ranks, true) != true ||
	    gin_rank_count_is_valid(max_wire_ranks + 1, true) ||
	    !gin_rank_count_is_valid(max_wire_ranks + 1, false) ||
	    gin_rank_count_is_valid(0, true) ||
	    gin_required_cq_data_size(true) != sizeof(uint64_t) ||
	    gin_required_cq_data_size(false) != sizeof(uint32_t)) {
		printf("Endpoint-MR rank or CQ data-width requirement failed\n");
		return 1;
	}

	/* If we start requesting ACKs at threshold and then request another one
	 * every interval ops, threshold + interval + the one-op escape hatch must
	 * still fit under the refresh window. Small windows clamp threshold to 1. */
	const ack_case ack_cases[] = {
		{1, 1},
		{4, 4},
		{33, 33},
		{64, 64},
		{96, 48},
		{256, 256},
	};
	for (const auto &tc : ack_cases) {
		const uint32_t interval = gin_ack_request_get_interval(tc.current_window);
		const uint32_t threshold = gin_ack_request_get_threshold(tc.current_window,
							    tc.refresh_window,
							    interval);
		if (tc.refresh_window > interval + 1) {
			if (threshold + interval + 1 > tc.refresh_window) {
				printf("ACK guard failed: current=%u refresh=%u "
			       "threshold=%u interval=%u\n",
				       tc.current_window, tc.refresh_window, threshold, interval);
				return 1;
			}
		} else if (threshold != 1) {
			printf("ACK threshold should clamp to 1: current=%u "
			       "refresh=%u threshold=%u interval=%u\n",
			       tc.current_window, tc.refresh_window, threshold, interval);
			return 1;
		}
	}

	const struct {
		uint32_t nranks;
		uint32_t expected;
	} eq_cases[] = {
		{1, GIN_IMM_SEQ_MASK + 1},
		{2, GIN_CXI_EQ_EVENT_BUDGET - 1},
		{8, GIN_CXI_EQ_EVENT_BUDGET / 7 - 1},
		{GIN_CXI_EQ_EVENT_BUDGET + 1, 1},
		{1024, 1},
	};
	for (const auto &tc : eq_cases) {
		if (gin_cxi_eq_get_window_limit(tc.nranks) != tc.expected) {
			printf("CXI EQ limit failed: nranks=%u limit=%u expected=%u\n",
			       tc.nranks, gin_cxi_eq_get_window_limit(tc.nranks), tc.expected);
			return 1;
		}
	}

	const cq_case cq_cases[] = {
		{2, configured_reserve > 0 ? configured_reserve - 1 : 0, 1, false},
		{2, configured_reserve, 1, false},
		{2, configured_reserve + 1, 1, false},
		{2, gin_rx_cq_get_reserve_entries(1) + GIN_RX_CQ_ENTRIES_PER_PUT_SIGNAL, 1, true},
		{2, gin_rx_cq_get_reserve_entries(1) +
		 3 * GIN_RX_CQ_ENTRIES_PER_PUT_SIGNAL, 2, true},
		{8, gin_rx_cq_get_reserve_entries(7) +
		 static_cast<size_t>(7) * GIN_RX_CQ_ENTRIES_PER_PUT_SIGNAL, 1, true},
		{8, gin_target_cq_get_size(8, GIN_TARGET_GRANTED_WINDOW), 2, true},
		{8, gin_target_cq_get_size(8, GIN_TARGET_GRANTED_WINDOW), 4, true},
		{4096, GIN_AUTO_CQ_SIZE_MAX, 8, true},
	};
	for (const auto &tc : cq_cases) {
		const uint32_t peers = gin_fanin_get_worst_case_peers(tc.nranks);
		const uint32_t window = gin_worst_case_window_get_limit_with_reserve(
			tc.cq_size, tc.nranks, configured_reserve, tc.ref_cnt);
		const size_t reserve_margin = configured_reserve;
		const size_t cq_budget = tc.ref_cnt * static_cast<size_t>(peers) *
			GIN_RX_CQ_ENTRIES_PER_PUT_SIGNAL * (window + 1) + reserve_margin;
		if (tc.expect_escape_hatch_safe) {
			if (cq_budget > tc.cq_size) {
				printf("Shared CQ safety failed: nranks=%u cq=%zu ref_cnt=%zu "
				       "peers=%u window=%u budget=%zu\n",
				       tc.nranks, tc.cq_size, tc.ref_cnt, peers, window, cq_budget);
				return 1;
			}
		} else if (window != 1) {
			printf("Tiny CQ should clamp to window=1: nranks=%u cq=%zu "
			       "ref_cnt=%zu window=%u\n",
			       tc.nranks, tc.cq_size, tc.ref_cnt, window);
			return 1;
		}
	}

	/* gin_apply_granted_window(): a receiver-advertised grant applied to a
	 * peer's sender window. Zero leaves the window untouched; growth and
	 * equal grants preserve the ACK hysteresis counter; only a strict
	 * shrink resets it so the ack_req gate can re-arm under the smaller
	 * window; an over-max grant clamps to gin_granted_window_get_max_limit(). */
	const uint32_t max_grant = gin_granted_window_get_max_limit();
	const grant_case grant_cases[] = {
		/* zero grant: window and hysteresis both untouched */
		{64, 7, 0, 64, 7},
		/* growth: window rises, hysteresis preserved */
		{64, 7, 128, 128, 7},
		/* shrink: window falls, hysteresis resets */
		{128, 7, 64, 64, 0},
		/* equal: no shrink, so no reset */
		{64, 7, 64, 64, 7},
		/* over-max grant clamps down; from a smaller window that is growth */
		{64, 7, GIN_ACK_GRANT_MAX, max_grant, 7},
		/* clamp floor: grant of 1 is the smallest legal window */
		{64, 7, 1, 1, 0},
	};
	for (const auto &tc : grant_cases) {
		nccl_ofi_gin_peer_rank_info peer = {};
		peer.current_window_limit = tc.start_limit;
		peer.consecutive_puts_without_ack = tc.start_hysteresis;
		gin_apply_granted_window(peer, tc.granted);
		if (peer.current_window_limit != tc.expect_limit ||
		    peer.consecutive_puts_without_ack != tc.expect_hysteresis) {
			printf("Grant apply failed: start=%u granted=%u -> limit=%u hyst=%u "
			       "(expected limit=%u hyst=%u)\n",
			       tc.start_limit, tc.granted, peer.current_window_limit,
			       peer.consecutive_puts_without_ack, tc.expect_limit,
			       tc.expect_hysteresis);
			return 1;
		}
	}

	/* gin_complete_pending_ack_request(): a backward/duplicate rx_consumed
	 * (delta in the upper half of the cursor range, i.e. an older or
	 * reordered ACK arriving after a newer one) must not clear a still-
	 * pending ack request. */
	{
		nccl_ofi_gin_peer_rank_info peer = {};
		peer.has_pending_ack_request = true;
		peer.last_ack_requested_seq = 100;
		/* rx_consumed behind last_ack_requested_seq by a small amount wraps
		 * to a large forward delta (near GIN_RX_CONSUMED_MASK), landing in
		 * the upper half -> treated as backward/stale. */
		const uint32_t backward_rx_consumed = 50;
		gin_complete_pending_ack_request(peer, backward_rx_consumed);
		if (!peer.has_pending_ack_request) {
			printf("Backward ACK incorrectly cleared has_pending_ack_request: "
			       "last_ack_requested_seq=%u rx_consumed=%u\n",
			       peer.last_ack_requested_seq, backward_rx_consumed);
			return 1;
		}
	}

	{
		nccl_ofi_gin_peer_rank_info peer = {};
		peer.has_pending_ack_request = true;
		peer.last_ack_requested_seq = 100;
		/* rx_consumed exactly equal to last_ack_requested_seq means the request
		 * op has not been retired yet. The ACK doesn't cover it. */
		const uint32_t exact_rx_consumed = 100;
		gin_complete_pending_ack_request(peer, exact_rx_consumed);
		if (!peer.has_pending_ack_request) {
			printf("Exact (non-covering) ACK incorrectly cleared "
			       "has_pending_ack_request: "
			       "last_ack_requested_seq=%u rx_consumed=%u\n",
			       peer.last_ack_requested_seq, exact_rx_consumed);
			return 1;
		}
	}
	printf("PASS: gin CQ arithmetic\n");
	return 0;
}
