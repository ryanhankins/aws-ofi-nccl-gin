/*
 * Copyright (c) 2026, Hewlett Packard Enterprise Development LP.
 */

#include "config.h"

#include <cstdio>

#include "nccl_ofi_assert.h"
#include "unit_test.h"
#include "nccl_ofi.h"
#include "rdma/gin/nccl_ofi_gin_comm_id_pool.h"
#include "rdma/gin/nccl_ofi_gin_types.h"

int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	unit_test_init();

	nccl_ofi_gin_comm_id_pool lifetime_pool(NCCL_GIN_MAX_COMMS);
	for (size_t expected = 0; expected < NCCL_GIN_MAX_COMMS; ++expected) {
		assert_always(lifetime_pool.allocate() == expected);
	}
	assert_always(lifetime_pool.allocate() == FI_KEY_NOTAVAIL);

	nccl_ofi_gin_comm_id_pool setup_pool(NCCL_GIN_MAX_COMMS);
	const size_t failed_setup_id = setup_pool.allocate();
	setup_pool.rollback_unpublished(failed_setup_id);
	assert_always(setup_pool.allocate() == failed_setup_id);

	printf("GIN communicator ID lifecycle tests passed\n");
	return 0;
}
