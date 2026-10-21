/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_COMM_ID_POOL_H_
#define NCCL_OFI_GIN_COMM_ID_POOL_H_

#include "nccl_ofi_idpool.h"

/*
 * Successfully published GIN communicator IDs remain allocated for the
 * lifetime of the shared resources: the wire protocol has no generation
 * field, and close does not prove that every late message has drained.
 */
class nccl_ofi_gin_comm_id_pool {
public:
	explicit nccl_ofi_gin_comm_id_pool(size_t capacity) : ids(capacity) {}

	nccl_ofi_gin_comm_id_pool(const nccl_ofi_gin_comm_id_pool &) = delete;
	nccl_ofi_gin_comm_id_pool &operator=(const nccl_ofi_gin_comm_id_pool &) = delete;

	size_t allocate()
	{
		return ids.allocate_id();
	}

	/* Only IDs from failed setup before wire publication may be rolled back. */
	void rollback_unpublished(size_t id)
	{
		ids.free_id(id);
	}

private:
	nccl_ofi_idpool_t ids;
};

#endif
