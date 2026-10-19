/*
 * SPDX-FileCopyrightText:  Copyright Hewlett Packard Enterprise Development LP
 */

#ifndef NCCL_OFI_RDMA_GIN_FLOW_WRITEDATA_ROUTED_H
#define NCCL_OFI_RDMA_GIN_FLOW_WRITEDATA_ROUTED_H

#include "rdma/gin/nccl_ofi_gin_types.h"

class gin_flow_writedata_routed : public gin_flow_policy {
public:
	gin_flow_writedata_routed(nccl_ofi_gin_resources &resources_arg, uint32_t nranks_arg)
		: resources(resources_arg), nranks(nranks_arg)
	{
	}

	void init_peer(nccl_ofi_gin_peer_rank_info &peer, uint32_t peer_nranks) override;

	int prepare_tx(nccl_ofi_gin_peer_rank_info &peer, gin_tx_flow_decision &decision,
		       gin_flow_context &ctx) override;

	void prepare_ack(const nccl_ofi_gin_peer_rank_info &peer, uint32_t rx_consumed,
			 gin_ack_msg_t &ack) override;

	void process_ack(nccl_ofi_gin_peer_rank_info &peer, const gin_ack_msg_t &ack) override;

private:
	nccl_ofi_gin_resources &resources;
	const uint32_t nranks;
};

#endif
