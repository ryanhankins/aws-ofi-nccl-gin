/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_GIN_TYPES_H
#define NCCL_OFI_RDMA_GIN_TYPES_H

#include "rdma/fabric.h"

#include "nccl_ofi_assert.h"
#include "nccl_ofi_param.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>

/**
 * Forward-declarations of GIN types
 */
class nccl_ofi_gin_mr_handle_t;
class nccl_ofi_rdma_gin_put_comm;
class nccl_ofi_gin_resources;
struct nccl_ofi_gin_ep_rail_t;
struct nccl_ofi_gin_peer_rank_info;
struct gin_ack_msg_t;

struct gin_tx_flow_decision {
	bool request_ack = false;
};

/**
 * Cached bundle of comm-level state a gin_flow_policy needs
 */
struct gin_flow_context {
	uint32_t conservative_refresh_window = 0;
	/* Drains this endpoint's CQ + gdrcopy done-queue under the ep lock.
	 * Bound to a free function that only touches the comm's already-public
	 * interface (get_ep_lock()/get_resources()/drain_gdrcopy_done_queue()),
	 * so handing this callable to the policy grants "make progress" without
	 * granting lock or internals access. Returns 0 or -errno. */
	nccl_ofi_rdma_gin_put_comm *progress_comm = nullptr;
	int (*progress_fn)(nccl_ofi_rdma_gin_put_comm &) = nullptr;

	int progress() const
	{
		assert_always(progress_fn != nullptr);
		return progress_fn(*progress_comm);
	}
};

/**
 * Provider-specific flow-control policy, owned by the comm (one instance per
 * comm, selected once at construction from the endpoint's negotiated mode).
 * comm's header.
 */
class gin_flow_policy {
public:
	virtual ~gin_flow_policy() = default;

	virtual void init_peer(nccl_ofi_gin_peer_rank_info &peer, uint32_t nranks) = 0;

	virtual int prepare_tx(nccl_ofi_gin_peer_rank_info &peer,
			      gin_tx_flow_decision &decision,
			      gin_flow_context &ctx) = 0;

	virtual void prepare_ack(const nccl_ofi_gin_peer_rank_info &peer,
				uint32_t rx_consumed, gin_ack_msg_t &ack) = 0;

	virtual void process_ack(nccl_ofi_gin_peer_rank_info &peer,
			 const gin_ack_msg_t &ack) = 0;
};

/**
 * Message types for GIN sends (metadata and ACK messages).
 */
enum gin_msg_type_t : uint8_t {
	GIN_MSG_TYPE_METADATA = 0,
	GIN_MSG_TYPE_ACK = 1,
	GIN_MSG_TYPE_MAX = GIN_MSG_TYPE_ACK,
};

/**
 * Represents metadata associated with a put-signal request. This is sent from
 * the put-signal initiator to the target.
 */

/**
 * Constants
 */
#define MAX_NUM_RAILS (4)

/**
 * Immediate data format (32 bits) for RDMA write-with-immediate signals.
 *
 * ACKs are no longer sent via immediate data — they use fi_send with
 * gin_ack_msg_t.  The immediate data is used only for data signals:
 *
 *  31      29  28  27      24 23             11 10            1  0
 *  [unused:3] [ar] [seg_cnt ] [  msg_seq_num  ] [   comm_id   ] [0]
 *
 * Bit 0 is reserved (always 0 for data signals).
 *
 * comm_id is 10 bits (1024 values). At most NCCL_GIN_MAX_CONTEXTS (4)
 * gin_comms share an endpoint, so at most 4 comm_id values are in use
 * per endpoint at any time. 10 bits provides ample headroom.
 */

/* Common fields (same position in both formats) */
#define GIN_IMM_TYPE_BITS      1
#define GIN_IMM_COMM_BITS      10
#define GIN_IMM_SEQ_BITS       13  /* can expand up to 15 without further
                                      changes to uint16_t seq vars or the
                                      32-bit immediate data layout */
#define GIN_IMM_COMM_SHIFT     GIN_IMM_TYPE_BITS
#define GIN_IMM_SEQ_SHIFT      (GIN_IMM_COMM_SHIFT + GIN_IMM_COMM_BITS)
#define GIN_IMM_SEQ_MASK       ((1 << GIN_IMM_SEQ_BITS) - 1)
#define GIN_IMM_COMM_MASK      ((1 << GIN_IMM_COMM_BITS) - 1)

#define GIN_IMM_GET_COMM_ID(data)  (((data) >> GIN_IMM_COMM_SHIFT) & GIN_IMM_COMM_MASK)
#define GIN_IMM_GET_SEQ_NUM(data)  (((data) >> GIN_IMM_SEQ_SHIFT) & GIN_IMM_SEQ_MASK)

/* Non-ACK fields above the common fields */
#define GIN_IMM_SEG_CNT_BITS   4
#define GIN_IMM_ACK_REQ_BITS   1
#define GIN_IMM_SEG_CNT_SHIFT  (GIN_IMM_SEQ_SHIFT + GIN_IMM_SEQ_BITS)
#define GIN_IMM_ACK_REQ_SHIFT  (GIN_IMM_SEG_CNT_SHIFT + GIN_IMM_SEG_CNT_BITS)
#define GIN_IMM_SEG_CNT_MASK   ((1 << GIN_IMM_SEG_CNT_BITS) - 1)

#define GIN_IMM_GET_SEG_CNT(data)       (((data) >> GIN_IMM_SEG_CNT_SHIFT) & GIN_IMM_SEG_CNT_MASK)
#define GIN_IMM_GET_ACK_REQUESTED(data) (((data) >> GIN_IMM_ACK_REQ_SHIFT) & 1)
#define GIN_IMM_SEG_DATA(comm_id, seq, nseg, ack_req)                                               \
	(((ack_req) << GIN_IMM_ACK_REQ_SHIFT) | ((nseg) << GIN_IMM_SEG_CNT_SHIFT) |               \
	 ((seq) << GIN_IMM_SEQ_SHIFT) | ((comm_id) << GIN_IMM_COMM_SHIFT))

/* msg_type in message headers (2 values currently. Reserving one more bit for now) */
#define GIN_MSG_TYPE_BITS      2
static_assert(((1 << GIN_MSG_TYPE_BITS) - 1) >= GIN_MSG_TYPE_MAX,
	      "GIN_MSG_TYPE_BITS must be wide enough to hold all message types");

/* Receiver's "rx_consumed" cursor: the count of in-order delivered ops
   modulo a wrap-safe space. The wire width is exactly twice the seq
   window so the standard half-window forward-delta comparison
   distinguishes "newer" from "older" ACKs without ambiguity. */
#define GIN_RX_CONSUMED_BITS  (GIN_IMM_SEQ_BITS + 1)
#define GIN_RX_CONSUMED_MASK  (((uint64_t)1 << GIN_RX_CONSUMED_BITS) - 1)
#define GIN_RX_CONSUMED_HALF  ((uint32_t)((GIN_RX_CONSUMED_MASK + 1) >> 1))
static_assert((1ULL << GIN_RX_CONSUMED_BITS) == 2ULL * (GIN_IMM_SEQ_MASK + 1),
	      "GIN_RX_CONSUMED_BITS must give exactly 2x the seq window for "
	      "wrap-safe forward-delta comparison between TX/RX cursors");

/* Once the sender's outstanding window is at GIN_ACK_REQ_THRESHOLD, only
   request a standalone ACK every GIN_ACK_INTERVAL ops. Without this
   throttle, every iputSignal above the threshold would request an ACK,
   producing an ACK storm rather than the slow drip the receiver needs
   to drain the bitset. */
#define GIN_ACK_INTERVAL  64

/* Worst-case receiver CQ footprint of one logical iputSignal op: one
   CQ entry for the metadata fi_send and one for the data write-immediate.
   Flow-control grants must budget for both to avoid many-to-one CQ overrun. */
#define GIN_RX_CQ_ENTRIES_PER_PUT_SIGNAL 2
/* Auto-sized GIN CQs aim to preserve a modest triple-digit worst-case
   granted window at scale. 128 is large enough to avoid near-serial
   progress at O(1 / nranks) fan-in, without immediately forcing multi-
   million-entry CQs on 1K-rank jobs. */
#define GIN_TARGET_GRANTED_WINDOW       128
/* Cap only the automatic CQ growth. Explicit OFI_NCCL_CQ_SIZE remains a
   floor/override and may still exceed this if an operator requests it. */
#define GIN_AUTO_CQ_SIZE_MAX            (1U << 20)

/* Max consecutive iputSignal ops coalesced onto one pinned rail before the pin
 * rotates to the next rail (so no single rail is pinned indefinitely). The
 * boundary op drops FI_MORE to flush the rail, then the pin advances. */
#define GIN_REQS_PER_DOORBELL 16
#define GIN_CXI_PAIR_DOORBELL_MIN (64 * 1024)
static_assert(GIN_ACK_INTERVAL <= (GIN_IMM_SEQ_MASK >> 1),
	      "GIN_ACK_INTERVAL must be much smaller than the seq window so "
	      "ack-request hysteresis does not delay drain past wrap");

/* Maximum number of GIN comms */
#define NCCL_GIN_MAX_COMMS    (1 << GIN_IMM_COMM_BITS)

/* Reserved bit computation for Metadata message header */
#define GIN_MSG_HEADER_RESERVED_BITS (64 - GIN_MSG_TYPE_BITS - GIN_IMM_COMM_BITS \
                                      - GIN_IMM_SEQ_BITS - GIN_IMM_SEG_CNT_BITS \
                                      - 1 /* ack_req */)

/**
 * Metadata message header (64 bits).
 *
 * Both nccl_net_ofi_gin_msg_header_t and gin_ack_msg_t share a common
 * prefix in bits [0:25]: msg_type(2) + comm_id(10) + rx_consumed(14) =
 * 26 bits.  This allows dispatch and ACK processing to read from the
 * same offsets regardless of message type.
 */
struct nccl_net_ofi_gin_msg_header_t {
	/* Message type identifier */
	uint64_t msg_type        : GIN_MSG_TYPE_BITS;
	/* Comm identifier on the receiver side */
	uint64_t remote_comm_id  : GIN_IMM_COMM_BITS;
	/* Message sequence number and count */
	uint64_t seq_num         : GIN_IMM_SEQ_BITS;
	uint64_t seq_seg_cnt     : GIN_IMM_SEG_CNT_BITS;
	/* Receiver should send back a standalone ACK soon. Mirrors the
	 * GIN_IMM_ACK_REQ bit carried in the data-write immediate; the sender
	 * sets both to the same value, the receiver OR's them across the two
	 * submission paths. */
	uint64_t ack_req         : 1;
	/* Reserved for future use */
	uint64_t reserved        : GIN_MSG_HEADER_RESERVED_BITS;
};
static_assert(sizeof(nccl_net_ofi_gin_msg_header_t) <= 8,
	      "nccl_net_ofi_gin_msg_header_t fields must fit in 64 bits");

/**
 * Metadata message (32 bytes for inline send).
 */
struct nccl_net_ofi_gin_signal_metadata_msg_t {
	nccl_net_ofi_gin_msg_header_t header;

	/* Signal information (if applicable) */
	uint64_t signal_base_address;
	uint64_t signal_offset;
	uint64_t signal_value;
};
static_assert(sizeof(nccl_net_ofi_gin_signal_metadata_msg_t) == 32,
	     "nccl_net_ofi_gin_signal_metadata_msg_t must be exactly 32 bytes for inline send");

#define GIN_ACK_MSG_GRANTED_WINDOW_BITS 14
#define GIN_ACK_GRANT_MAX ((1U << GIN_ACK_MSG_GRANTED_WINDOW_BITS) - 1)
#define GIN_ACK_MSG_RESERVED_BITS (64 - GIN_MSG_TYPE_BITS - GIN_IMM_COMM_BITS - GIN_RX_CONSUMED_BITS - GIN_ACK_MSG_GRANTED_WINDOW_BITS)

/**
 * Standalone ACK message sent via fi_send from receiver to sender (8 bytes).
 *
 * Shares the common prefix layout with nccl_net_ofi_gin_msg_header_t
 * so msg_type-based dispatch works at the same bit position.
 */
struct gin_ack_msg_t {
	/* Message type identifier */
	uint64_t msg_type        : GIN_MSG_TYPE_BITS;
	/* Comm identifier on the original sender side */
	uint64_t ack_comm_id     : GIN_IMM_COMM_BITS;
	/* Receiver's rx_consumed cursor (wraps at 2x seq window) */
	uint64_t rx_consumed     : GIN_RX_CONSUMED_BITS;
	/* Receiver-granted sender outstanding window */
	uint64_t granted_window  : GIN_ACK_MSG_GRANTED_WINDOW_BITS;
	/* Reserved for future use */
	uint64_t reserved        : GIN_ACK_MSG_RESERVED_BITS;
};
static_assert(sizeof(gin_ack_msg_t) <= 8,
	      "gin_ack_msg_t must be exactly 8 bytes for inline send");

/* Above layout assume little-endian byte order, as well as following static
   assert checks. All message structs must have msg_type at the lowest bits
   so that type-based dispatch can read the first bits without knowing the
   concrete type. */
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
	      "GIN message struct layout assumes little-endian byte order");
static_assert(offsetof(nccl_net_ofi_gin_signal_metadata_msg_t, header) == 0,
	      "header must be at offset 0 for type-based dispatch");
/* Verify msg_type is at the LSB of each struct. Clang does not yet support
   constexpr bit_cast on bitfield structs, so only check on GCC. */
#if !defined(__clang__) && defined(__GNUC__) && (__GNUC__ >= 9)
static_assert(
	__builtin_bit_cast(uint64_t,
		nccl_net_ofi_gin_msg_header_t{GIN_MSG_TYPE_MAX, 0, 0, 0, 0, 0})
		== (uint64_t)GIN_MSG_TYPE_MAX,
	"msg_type must be at LSB of nccl_net_ofi_gin_msg_header_t");
static_assert(
	__builtin_bit_cast(uint64_t,
		gin_ack_msg_t{GIN_MSG_TYPE_MAX, 0, 0, 0, 0})
		== (uint64_t)GIN_MSG_TYPE_MAX,
	"msg_type must be at LSB of gin_ack_msg_t");
#endif

/* Base half-window ACK-request threshold. Runtime code clamps this
   against the receiver-granted window so grant shrink can never leave the
   sender blocked below a stale fixed gate. */
#define GIN_ACK_REQ_THRESHOLD ((GIN_IMM_SEQ_MASK + 1) / 2)

static_assert(GIN_TARGET_GRANTED_WINDOW <= GIN_ACK_GRANT_MAX,
	      "GIN auto-sized target window must fit in the ACK grant field");

/**
 * Represents per-peer-rank data associated with a collective communicator.
 *
 * The collective communicator stores a vector of these structures, of size
 * nranks.
 *
 * Producer/consumer flow-control model
 * ------------------------------------
 * Each peer pair maintains four monotonically-advancing cursors of width
 * GIN_RX_CONSUMED_BITS (= GIN_IMM_SEQ_BITS + 1, so exactly 2x the seq
 * window). Cursors wrap at GIN_RX_CONSUMED_MASK + 1 and are stored in a
 * uint32_t backing for cheap arithmetic.
 *
 *   tx_head     - sender: ops produced (advances on every iputSignal)
 *   tx_tail     - sender's local view of "ops the receiver has consumed"
 *                 (advances when a standalone ACK arrives)
 *   rx_consumed - receiver: ops delivered upstream in seq order
 *
 * Wrap-safe forward delta is computed as
 *     ((b - a) & GIN_RX_CONSUMED_MASK)
 * which is in [0, 2x seq window). A delta in the lower half is "forward",
 * in the upper half is "backward" (older ACK or duplicate).
 *
 * Invariants:
 *   outstanding   = (tx_head - tx_tail) & GIN_RX_CONSUMED_MASK
 *   tx may send   iff outstanding stays below the provider-selected
 *                 sender window (full sequence window for posted_recv,
 *                 receiver grant for writedata_routed)
 *   ack-request   follows the provider-selected threshold policy
 *
 * The wire seq number is the low GIN_IMM_SEQ_BITS of tx_head.
 * The wire `rx_consumed` field is the entire receiver-side cursor
 * (since cursor width matches wire width).
 */
struct nccl_ofi_gin_peer_rank_info {
	/* Remote comm id */
	uint32_t comm_id;

	/* Rail addresses */
	fi_addr_t address[MAX_NUM_RAILS];

	/* Sender-side cursors. tx_head is the next ring-position we'll
	   produce; tx_tail is the latest position the receiver has
	   acknowledged consuming. tx_head's low GIN_IMM_SEQ_BITS go on
	   the wire as msg_seq_num. */
	uint32_t tx_head = 0;
	uint32_t tx_tail = 0;

	/* Receiver-side cursor. rx_consumed advances every time we retire
	   an op in seq order (signal delivered upstream). */
	uint32_t rx_consumed = 0;

	/* Hysteresis counter for the sender-side ack-request gate. Once
	   outstanding crosses GIN_ACK_REQ_THRESHOLD this counts ops; only
	   every GIN_ACK_INTERVAL'th op flips ack_req on the wire. Without
	   this, every op above threshold would request an ACK and we'd
	   flood the receiver with standalone ACK packets. */
	uint32_t consecutive_puts_without_ack = 0;

	/* Sequence-number tracking for close-time drain.
	   last_ack_requested_seq records tx_head at the time the most
	   recent is_ack_requested was set on the wire.
	   closeColl waits until a standalone ACK covers this sequence. */
	uint32_t last_ack_requested_seq = 0;
	bool has_pending_ack_request = false;
	uint32_t current_window_limit = 1;

	/**
	 * Next-to-be-delivered sequence number, stored at target, from
	 * (initiator) peer rank. Mirrors the low GIN_IMM_SEQ_BITS of
	 * rx_consumed; kept as a separate field to match how it is read
	 * (compared against incoming msg_seq_num bitfield).
	 */
	uint16_t next_delivered_signal_seq_num = 0;
};

/* Helpers for the cursor ring. Cursors are uint32_t in memory but we
   never let them advance past GIN_RX_CONSUMED_MASK, so subtraction is
   performed under the same mask. */
static inline uint32_t gin_cursor_inc(uint32_t c)
{
	return (c + 1) & GIN_RX_CONSUMED_MASK;
}

static inline uint32_t gin_cursor_delta(uint32_t lhs, uint32_t rhs)
{
	return (lhs - rhs) & GIN_RX_CONSUMED_MASK;
}

static inline void gin_complete_pending_ack_request(
	nccl_ofi_gin_peer_rank_info &peer, uint32_t rx_consumed)
{
	if (peer.has_pending_ack_request) {
		const uint32_t delta = gin_cursor_delta(rx_consumed,
					       peer.last_ack_requested_seq);
		if (delta != 0 && delta <= GIN_RX_CONSUMED_HALF) {
			peer.has_pending_ack_request = false;
		}
	}
}

static inline void gin_fill_ack_header(const nccl_ofi_gin_peer_rank_info &peer,
			       uint32_t rx_consumed, gin_ack_msg_t &ack)
{
	ack = {};
	ack.msg_type = GIN_MSG_TYPE_ACK;
	ack.ack_comm_id = static_cast<uint16_t>(peer.comm_id);
	ack.rx_consumed = rx_consumed & GIN_RX_CONSUMED_MASK;
}

struct gin_cq_sizing_diagnostics {
	size_t configured_size;
	bool was_auto_sized;
	bool was_auto_capped;
	uint32_t effective_window;
};

static inline uint32_t gin_fanin_get_worst_case_peers(uint32_t nranks)
{
	return std::max<uint32_t>(1, nranks - 1);
}

static inline uint32_t gin_granted_window_get_max_limit()
{
	return std::min<uint32_t>(GIN_IMM_SEQ_MASK + 1, GIN_ACK_GRANT_MAX);
}

/* Apply a receiver-advertised grant to a peer's sender window. A zero grant
   means an older/skewed peer did not advertise one; leave the window alone.
   Shrinking resets the ACK hysteresis counter so the next op can re-arm the
   ack_req gate under the smaller window. */
static inline void gin_apply_granted_window(nccl_ofi_gin_peer_rank_info &peer,
			uint32_t granted_window)
{
	if (granted_window == 0) {
		return;
	}
	const uint32_t old_limit = peer.current_window_limit;
	const uint32_t new_limit =
		std::clamp<uint32_t>(granted_window, 1, gin_granted_window_get_max_limit());
	peer.current_window_limit = new_limit;
	if (new_limit < old_limit) {
		peer.consecutive_puts_without_ack = 0;
	}
}

static inline size_t gin_rx_cq_get_reserve_entries(
	uint32_t peers, size_t reserve_margin = ofi_nccl_gin_rx_cq_reserve_margin())
{
	return reserve_margin +
		static_cast<size_t>(peers) * GIN_RX_CQ_ENTRIES_PER_PUT_SIGNAL;
}

static inline size_t gin_target_cq_get_size(
	uint32_t nranks, uint32_t target_window,
	size_t reserve_margin = ofi_nccl_gin_rx_cq_reserve_margin())
{
	const uint32_t peers = gin_fanin_get_worst_case_peers(nranks);
	return static_cast<size_t>(peers) * GIN_RX_CQ_ENTRIES_PER_PUT_SIGNAL * target_window +
		gin_rx_cq_get_reserve_entries(peers, reserve_margin);
}

static inline uint32_t gin_worst_case_window_get_limit_with_reserve(
	size_t cq_size, uint32_t nranks, size_t reserve_margin, size_t ref_cnt = 1)
{
	/* Grants must fit the sum of all concurrently valid peer windows, not
	 * just the latest ACK's snapshot of active fan-in. Use the actual CQ
	 * size configured when this endpoint/context was created: auto-sizing may
	 * have raised it above the raw OFI_NCCL_CQ_SIZE floor to preserve a more
	 * useful worst-case sender window at scale.
	 *
	 * Increase OFI_NCCL_CQ_SIZE if large communicators need
	 * a wider steady-state grant than auto-sizing (or its cap) provides.
	 *
	 * ref_cnt counts live communicators sharing the same endpoint/CQ. Each
	 * communicator gets its own one-op ACK-request escape hatch, so the shared
	 * budget must divide the entire (window + overshoot) allowance across all
	 * live communicators, not just the steady-state window alone. */
	const uint32_t peers = gin_fanin_get_worst_case_peers(nranks);
	const size_t live_comms = std::max<size_t>(1, ref_cnt);
	const size_t per_comm_budget =
		(cq_size > reserve_margin)
			? (cq_size - reserve_margin) /
				  (live_comms * peers * GIN_RX_CQ_ENTRIES_PER_PUT_SIGNAL)
			: 0;
	const size_t worst_case_window = (per_comm_budget > 0) ? (per_comm_budget - 1) : 0;
	return std::clamp<uint32_t>(static_cast<uint32_t>(worst_case_window), 1,
				     gin_granted_window_get_max_limit());
}



static inline uint32_t gin_ack_request_get_interval(uint32_t current_window_limit)
{
	return std::min<uint32_t>(GIN_ACK_INTERVAL,
		std::max<uint32_t>(1, current_window_limit / 4));
}

static inline uint32_t gin_ack_request_get_threshold(uint32_t current_window_limit,
				       uint32_t refresh_window_limit,
				       uint32_t ack_req_interval)
{
	/* Window shrink must never outrun the ACK-request gate: clamp the
	   half-window policy against both the current receiver grant and the
	   conservative pre-ACK refresh budget so hysteresis still leaves at
	   least one op to carry ack_req before either limit fills. Keep the
	   safety threshold tied to the caller's scaled ACK interval so both
	   helpers encode the same pacing assumption. */
	uint32_t latest_safe_threshold = 1;
	if (refresh_window_limit > (ack_req_interval + 1)) {
		latest_safe_threshold = refresh_window_limit - ack_req_interval - 1;
	}
	return std::max<uint32_t>(1,
		std::min<uint32_t>(current_window_limit / 2, latest_safe_threshold));
}

static inline gin_cq_sizing_diagnostics gin_compute_cq_sizing_diagnostics(
	size_t cq_size_floor, uint32_t nranks,
	size_t reserve_margin = ofi_nccl_gin_rx_cq_reserve_margin())
{
	const size_t target_cq_size =
		gin_target_cq_get_size(nranks, GIN_TARGET_GRANTED_WINDOW, reserve_margin);
	const size_t auto_sized_cq = std::min<size_t>(target_cq_size, GIN_AUTO_CQ_SIZE_MAX);
	const size_t configured_size = std::max<size_t>(cq_size_floor, auto_sized_cq);
	return {configured_size,
		configured_size > cq_size_floor,
		target_cq_size > GIN_AUTO_CQ_SIZE_MAX && auto_sized_cq > cq_size_floor,
		gin_worst_case_window_get_limit_with_reserve(
			configured_size, nranks, reserve_margin)};
}

#endif
