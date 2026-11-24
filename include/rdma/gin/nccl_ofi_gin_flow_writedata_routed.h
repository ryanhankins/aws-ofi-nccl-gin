/*
 * Copyright (c) 2026, Hewlett Packard Enterprise Development LP.
 */

#ifndef NCCL_OFI_RDMA_GIN_FLOW_WRITEDATA_ROUTED_H_
#define NCCL_OFI_RDMA_GIN_FLOW_WRITEDATA_ROUTED_H_

#include "rdma/gin/nccl_ofi_gin_resources.h"
#include "rdma/gin/nccl_ofi_gin_types.h"

class gin_flow_writedata_routed : public gin_flow_policy {
public:
	gin_flow_writedata_routed(nccl_ofi_gin_resources &resources_arg, uint32_t nranks_arg)
		: resources(resources_arg), nranks(nranks_arg)
	{
	}

	/* Called with the shared endpoint's ep_lock held. */
	void init_peer(nccl_ofi_gin_peer_rank_info &peer, uint32_t peer_nranks) override
		REQUIRES(resources.get_ep_lock());

	int prepare_tx(nccl_ofi_gin_peer_rank_info &peer, gin_tx_flow_decision &decision,
		       gin_flow_context &ctx, std::unique_lock<std::mutex> &ep_lock) override;

	void prepare_ack(const nccl_ofi_gin_peer_rank_info &peer,
			 uint32_t rx_consumed,
			 gin_ack_msg &ack) override REQUIRES(resources.get_ep_lock());

	void process_ack(nccl_ofi_gin_peer_rank_info &peer,
			 const gin_ack_msg &ack,
			 bool is_newest_ack) override;

private:
	nccl_ofi_gin_resources &resources;
	const uint32_t nranks;
};

#endif
