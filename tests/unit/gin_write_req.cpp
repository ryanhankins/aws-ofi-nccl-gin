/*
 * SPDX-FileCopyrightText:  Copyright Hewlett Packard Enterprise Development LP
 */

#include "config.h"

#include <array>

#include "unit_test.h"
#include "nccl_ofi_assert.h"
#include "rdma/gin/nccl_ofi_gin_reqs.h"

static nccl_net_ofi_gin_write_req_t::args expected;
static void *post_context;
static ssize_t post_result;
static unsigned int writemsg_calls;
static unsigned int writedata_calls;

static void check_context(void *context)
{
	assert_always(context != nullptr);
	if (post_context) {
		assert_always(context == post_context);
	}
	post_context = context;
}

static ssize_t mock_writemsg(struct fid_ep *ep, const struct fi_msg_rma *msg, uint64_t flags)
{
	++writemsg_calls;
	assert_always(ep == expected.ep);
	assert_always(msg->iov_count == 1 && msg->rma_iov_count == 1);
	assert_always(msg->msg_iov[0].iov_base == expected.src);
	assert_always(msg->msg_iov[0].iov_len == expected.size);
	assert_always(msg->desc[0] == expected.desc);
	assert_always(msg->addr == expected.remote_addr);
	assert_always(msg->rma_iov[0].addr == expected.dest);
	assert_always(msg->rma_iov[0].len == expected.size);
	assert_always(msg->rma_iov[0].key == expected.key);
	assert_always(msg->data == expected.imm_data);
	assert_always(flags == expected.flags);
	check_context(msg->context);
	return post_result;
}

static ssize_t mock_writedata(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			     uint64_t data, fi_addr_t remote_addr, uint64_t dest, uint64_t key,
			     void *context)
{
	++writedata_calls;
	assert_always(ep == expected.ep);
	assert_always(buf == expected.src && len == expected.size);
	assert_always(desc == expected.desc && data == expected.imm_data);
	assert_always(remote_addr == expected.remote_addr);
	assert_always(dest == expected.dest && key == expected.key);
	check_context(context);
	return post_result;
}

int main()
{
	unit_test_init();
	struct fi_ops_rma ops = {};
	ops.writemsg = mock_writemsg;
	ops.writedata = mock_writedata;
	struct fid_ep ep = {};
	ep.rma = &ops;
	std::array<char, 16> buffer = {};
	int descriptor;
	expected = {&ep, buffer.data(), buffer.size(), &descriptor, UINT64_C(0x123456789abcdef0),
		    7, 128, 42, nullptr, FI_REMOTE_CQ_DATA};

	const bool original_endpoint_mr = endpoint_mr;
	for (bool use_writedata : {false, true}) {
		nccl_net_ofi_gin_write_req_t writemsg_req(expected);
		nccl_ofi_gin_writedata_req writedata_req(expected);
		nccl_net_ofi_gin_op_req_t &req = use_writedata
			? static_cast<nccl_net_ofi_gin_op_req_t &>(writedata_req)
			: static_cast<nccl_net_ofi_gin_op_req_t &>(writemsg_req);
		writemsg_calls = writedata_calls = 0;
		post_context = nullptr;
		/* Posting and retrying must use the request type, not the global. */
		for (ssize_t result : {-FI_EAGAIN, 0, -FI_EIO}) {
			endpoint_mr = !use_writedata;
			post_result = result;
			assert_always(req.post() == result);
		}
		assert_always(writemsg_calls == (use_writedata ? 0U : 3U));
		assert_always(writedata_calls == (use_writedata ? 3U : 0U));
	}
	endpoint_mr = original_endpoint_mr;
	return 0;
}
