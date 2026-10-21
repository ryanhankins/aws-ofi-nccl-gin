/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "rdma/gin/nccl_ofi_gin_resources.h"
#include "rdma/gin/nccl_ofi_gin.h"
#include "rdma/gin/nccl_ofi_gin_reqs.h"
#include "rdma/gin/nccl_ofi_gin_gdaki.h"
#include "rdma/gin/nccl_ofi_gin_flow_posted_recv.h"
#include "rdma/gin/nccl_ofi_gin_flow_writedata_routed.h"

#include "nccl_ofi_assert.h"
#include <cinttypes>
#include <cstdlib>
#include <chrono>

#include <thread>
#if HAVE_CUDA
#include "nccl_ofi_cuda.h"
#endif
#include "nccl_ofi_dmabuf.h"
#include "nccl_ofi_ofiutils.h"
#include "nccl_ofi_mr.h"
#include "nccl_ofi_param.h"
#include "nccl_ofi.h"
#include "nccl_ofi_log.h"
#include "rdma/gin/nccl_ofi_gin.h"

std::shared_ptr<nccl_ofi_gin_resources>
nccl_ofi_gin_resources_registry::get(const std::shared_ptr<nccl_net_ofi_ep_t> &ep,
								   uint32_t nranks)
{
	std::lock_guard<std::mutex> guard(lock);

	auto it = by_ep.find(ep.get());
	if (it != by_ep.end()) {
		if (auto existing = it->second.resource.lock()) {
			existing->maybe_warn_cq_sizing_mismatch(nranks);
			return existing;
		}
	}

	auto resources = std::make_shared<nccl_ofi_gin_resources>(ep, nranks);
	by_ep[ep.get()] = {resources.get(), resources};
	return resources;
}

void nccl_ofi_gin_resources_registry::release(nccl_net_ofi_ep_t *ep,
						      nccl_ofi_gin_resources *resource)
{
	std::lock_guard<std::mutex> guard(lock);
	gin_registry_erase_if_same_resource(by_ep, ep, resource);
}

nccl_ofi_gin_resources_registry &gin_resources_registry()
{
	static nccl_ofi_gin_resources_registry registry;
	return registry;
}

static inline struct fi_info *get_gin_info(struct fi_info *info);

nccl_ofi_rdma_gin_ep_t::nccl_ofi_rdma_gin_ep_t(nccl_net_ofi_domain_t &domain_arg,
				       uint32_t cq_sizing_nranks_arg)
	: domain(domain_arg),
	  configured_rx_cq_size(0),
	  cq_sizing_nranks(cq_sizing_nranks_arg)
{
	/*
	 * Determine once, at construction, whether this provider negotiated
	 * FI_RX_CQ_DATA (e.g. EFA) or not (e.g. CXI), and select the matching
	 * CQ completion/error implementations. See the comment on
	 * process_completions_impl in nccl_ofi_gin_resources.h for why this
	 * differs by provider.
	 */
	ofi_info_ptr gin_info(get_gin_info(domain.get_device()->get_ofi_info(0)));
	uses_posted_recv_flow = gin_info->mode & FI_RX_CQ_DATA;
	if (uses_posted_recv_flow) {
		process_completions_impl =
			&nccl_ofi_rdma_gin_ep_t::gin_process_completions_posted_recv;
	} else {
		process_completions_impl =
			&nccl_ofi_rdma_gin_ep_t::gin_process_completions_writedata_routed;
	}

	bool cq_was_auto_sized = false;
	bool cq_was_auto_capped = false;
	uint32_t cq_effective_window = 0;
	const size_t cq_size_floor = ofi_nccl_cq_size();
	this->rx_cq_reserve_margin = ofi_nccl_gin_rx_cq_reserve_margin();
	if (uses_posted_recv_flow) {
		/* Posted-recv providers do not route windowing through CQ grants, so
		 * preserve the plain OFI_NCCL_CQ_SIZE floor rather than applying the
		 * CXI-only auto-sizing heuristic. */
		configured_rx_cq_size = cq_size_floor;
	} else {
		const auto cq_diagnostics =
			gin_compute_cq_sizing_diagnostics(cq_size_floor, cq_sizing_nranks,
						 rx_cq_reserve_margin);
		configured_rx_cq_size = cq_diagnostics.configured_size;
		cq_was_auto_sized = cq_diagnostics.was_auto_sized;
		cq_was_auto_capped = cq_diagnostics.was_auto_capped;
		cq_effective_window = cq_diagnostics.effective_window;
	}

	const size_t requested_rx_cq_size = configured_rx_cq_size;
	/* The CQ-safety invariant below depends on the provider honoring this
	 * requested fi_cq_attr.size. libfabric offers no standard post-open query
	 * to verify the realized depth, so we size grants from the request itself. */
	this->num_rails = domain.get_ofi_num_rails();
	this->cq_process_max_iter = ofi_nccl_gin_cq_process_max_iter();
	assert_always(this->num_rails > 0);
	rails.reserve(this->num_rails);

	/*
	 * Create rails. Rail 0 reuses the fi_info fetched above (this is the
	 * same fi_getinfo() result rail 0 would otherwise re-derive), so only
	 * rails 1..N-1 need their own get_gin_info() call. This turns what
	 * was an N+1 fi_getinfo() call sequence into N.
	 */
	rails.emplace_back(uint16_t{0}, domain, requested_rx_cq_size, std::move(gin_info));
	for (uint16_t r = 1; r < this->num_rails; r++) {
		rails.emplace_back(r, domain, requested_rx_cq_size);
	}
	if (cq_was_auto_sized) {
		NCCL_OFI_INFO(
			NCCL_NET, "GIN auto-sized RX CQ from %zu to %zu entries "
			"for nranks=%u (target window=%u)",
			      cq_size_floor, requested_rx_cq_size, cq_sizing_nranks,
			      GIN_TARGET_GRANTED_WINDOW);
	}
	if (cq_was_auto_capped) {
		NCCL_OFI_WARN("GIN auto-sized RX CQ capped at %zu entries for nranks=%u; "
			      "worst-case window=%u remains below target %u",
			      configured_rx_cq_size, cq_sizing_nranks,
			      cq_effective_window,
			      GIN_TARGET_GRANTED_WINDOW);
	}
	this->scheduler = new nccl_net_ofi_threshold_scheduler(this->num_rails);
}

nccl_ofi_rdma_gin_ep_t::~nccl_ofi_rdma_gin_ep_t()
{
	if (scheduler) {
		delete scheduler;
		scheduler = nullptr;
	}
}

std::unique_ptr<gin_flow_policy> nccl_ofi_rdma_gin_ep_t::make_flow_policy(
	nccl_ofi_gin_resources &resources, uint32_t nranks) const
{
	if (uses_posted_recv_flow) {
		return std::make_unique<gin_flow_posted_recv>();
	}
	return std::make_unique<gin_flow_writedata_routed>(resources, nranks);
}

int nccl_ofi_rdma_gin_ep_t::dispatch_via_op_context(struct fi_cq_data_entry *cq_entry,
						    fi_addr_t src_addr, uint16_t rail_id)
{
	void *op_ctx = cq_entry->op_context;

	if (OFI_UNLIKELY(op_ctx == NULL)) {
		NCCL_OFI_WARN("Invalid request context provided");
		return -EINVAL;
	}

	nccl_net_ofi_gin_context *ctx =
		cpp_container_of(static_cast<struct fi_context2 *>(op_ctx),
				 &nccl_net_ofi_gin_context::ofi_ctx);

	int ret = ctx->handle_cq_entry(reinterpret_cast<struct fi_cq_entry *>(cq_entry), src_addr,
					rail_id);
	if (ret != 0) {
		NCCL_OFI_WARN("Context progress failed: %d", ret);
		return ret;
	}

	return 0;
}

int nccl_ofi_rdma_gin_ep_t::gin_process_completions_writedata_routed(
	struct fi_cq_data_entry *cq_entry, fi_addr_t *src_addrs,
	uint64_t num_cqes, uint16_t rail_id)
{
	int ret = 0;

	for (uint64_t comp_idx = 0; comp_idx < num_cqes; comp_idx++) {
		void *op_ctx = cq_entry[comp_idx].op_context;

		if (op_ctx == NULL && endpoint_mr && remote_write_cb &&
		    (cq_entry[comp_idx].flags & FI_REMOTE_WRITE)) {
			ret = remote_write_cb(&cq_entry[comp_idx], src_addrs[comp_idx], rail_id);
			if (OFI_UNLIKELY(ret != 0)) {
				NCCL_OFI_WARN("Remote write completion failed: %d", ret);
				return ret;
			}
			continue;
		}

		if (OFI_UNLIKELY(op_ctx == NULL)) {
			NCCL_OFI_WARN("Invalid request context provided");
			return -EINVAL;
		}

		ret = dispatch_via_op_context(&cq_entry[comp_idx], src_addrs[comp_idx], rail_id);
		if (ret != 0) {
			return ret;
		}
	}

	return 0;
}

int nccl_ofi_rdma_gin_ep_t::gin_process_completions_posted_recv(
	struct fi_cq_data_entry *cq_entry, fi_addr_t *src_addrs,
	uint64_t num_cqes, uint16_t rail_id)
{
	int ret = 0;

	for (uint64_t comp_idx = 0; comp_idx < num_cqes; comp_idx++) {
		/* FI_RX_CQ_DATA guarantees writedata completions are always
		 * delivered against a posted fi_recv(); handle_cq_entry() on that
		 * request classifies FI_REMOTE_WRITE itself. */
		ret = dispatch_via_op_context(&cq_entry[comp_idx], src_addrs[comp_idx], rail_id);
		if (ret != 0) {
			return ret;
		}
	}

	return 0;
}

int nccl_ofi_rdma_gin_ep_t::gin_process_error_entry(struct fi_cq_err_entry *err_entry, struct fid_cq *cq,
					       uint16_t rail_id)
{
	void *op_ctx = err_entry->op_context;
	if (OFI_UNLIKELY(op_ctx == NULL)) {
		NCCL_OFI_WARN("Invalid request context provided");
		return -EINVAL;
	}

	nccl_net_ofi_gin_context *ctx = cpp_container_of(static_cast<struct fi_context2 *>(op_ctx),
							 &nccl_net_ofi_gin_context::ofi_ctx);

	return ctx->handle_error_entry(cq, err_entry,
					 rail_id);
}

int nccl_ofi_rdma_gin_ep_t::gin_process_cq_rail(uint16_t rail_id)
{
	assert(rail_id < num_rails);

	struct fi_cq_data_entry cqe_buffers[cq_read_count];
	fi_addr_t src_addrs[cq_read_count];
	auto *rail_cq = this->get_rail(rail_id).rail_cq.get();
	ssize_t rc = 0;
	size_t iter = 0;
	int ret = 0;
	int total_processed = 0;

	do {
		++iter;
		/* Receive completions for the given rail */
		rc = fi_cq_readfrom(rail_cq, cqe_buffers, cq_read_count, src_addrs);
		if (rc > 0) {
			ret = gin_process_completions(cqe_buffers, src_addrs, rc, rail_id);
			if (OFI_UNLIKELY(ret != 0))
				goto exit;
			total_processed += rc;
		} else if (OFI_UNLIKELY(rc == -FI_EAVAIL)) {
			/*
			 * On call to fi_cq_readerr, Libfabric requires some members of
			 * err_entry to be zero-initialized or point to valid data.  For
			 * simplicity, just zero out the whole struct.
			 */
			struct fi_cq_err_entry err_entry = {};

			ret = fi_cq_readerr(rail_cq, &err_entry, 0);
			if (OFI_UNLIKELY(ret == -FI_EAGAIN)) {
				/*
				 * Error not available yet.
				 * fi_cq_readfrom will keep returning -FI_EAVAIL so just bail
				 * out and try again later.
				 */
				ret = 0;
				break;
			} else if (OFI_UNLIKELY(ret < 0)) {
				NCCL_OFI_WARN(
					"Unable to read from fi_cq_readerr. RC: %d. Error: %s", ret,
					fi_strerror(-ret));
				goto exit;
			}

			if (OFI_UNLIKELY((err_entry.flags & FI_REMOTE_WRITE) &&
					err_entry.op_context == nullptr)) {
				/* Target-side writedata errors are keyed by the bound MR,
				 * not by a posted fi_recv() context (providers without
				 * FI_RX_CQ_DATA); providers with FI_RX_CQ_DATA always
				 * deliver a valid op_context, so this never fires there. */
				NCCL_OFI_WARN("GIN remote-write CQ error on rail %hu. data=%" PRIu64
					      " src_addr=%" PRIu64 " err=%d prov_errno=%d (%s)",
					      rail_id, err_entry.data, err_entry.src_addr,
					      err_entry.err, err_entry.prov_errno,
					      fi_cq_strerror(rail_cq, err_entry.prov_errno,
							     err_entry.err_data, NULL, 0));
				/* Drop the completion and continue; a single transient remote-write
				 * failure shouldn't inherently crash the entire shared endpoint. */
				total_processed += 1;
				continue;
			}

			/* Every other completion (sends, acks, metadata, flush reads, ...)
			 * always carries a valid op_context, independent of FI_RX_CQ_DATA. */
			ret = gin_process_error_entry(&err_entry, rail_cq, rail_id);
			if (ret != 0) {
				goto exit;
			}
			/* Error completions are completions too */
			total_processed += 1;
		} else if (rc == -FI_EAGAIN) {
			/* No completions to process */
			break;
		} else {
			NCCL_OFI_WARN(
				"Unable to retrieve completion queue entries. RC: %zd, ERROR: %s",
				rc, fi_strerror(-rc));
			ret = -EINVAL;
			goto exit;
		}
	} while (iter < cq_process_max_iter);

	ret = total_processed;
exit:
	return ret;
}

int nccl_ofi_rdma_gin_ep_t::process_cq()
{
	int ret = 0;
	int total_processed = 0;

	/* Process completion queues for all rails */
	for (uint16_t rail_id = 0; rail_id < num_rails; ++rail_id) {
		ret = gin_process_cq_rail(rail_id);
		if (OFI_UNLIKELY(ret < 0)) {
			NCCL_OFI_WARN("Failed to process CQ for rail %u: %d", rail_id, ret);
			return ret;
		}
		total_processed += ret;
	}

	return total_processed;
}

void nccl_ofi_rdma_gin_ep_t::close_ofi_eps()
{
	rails.clear();
}

/**
 * Set mr attrs. This function closely resembles the same one in RDMA
 */
static int set_mr_req_attr(uint64_t mr_key, nccl_ofi_mr_ckey_ref ckey, uint64_t *flags, int type,
			   struct fi_mr_attr *mr_attr)
{
	int ret = 0;

	/* Basic put-signal access + read access for iget */
	mr_attr->access = FI_WRITE | FI_REMOTE_WRITE | FI_READ | FI_REMOTE_READ;
	nccl_ofi_mr_ckey_fill_mr_attrs(ckey, mr_attr, flags);

	switch (type) {
	case NCCL_PTR_HOST:
		mr_attr->iface = FI_HMEM_SYSTEM;
		break;
#if HAVE_CUDA
	case NCCL_PTR_CUDA:
		mr_attr->iface = FI_HMEM_CUDA;

		/* Get CUDA device ID */
		ret = nccl_net_ofi_get_gpu_device_for_addr((void *)nccl_ofi_mr_ckey_baseaddr(ckey),
							   &mr_attr->device.cuda);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Failed call to nccl_net_ofi_get_cuda_device_for_addr: %d",
				      ret);
			return ret;
		}
		break;
#endif

	default:
		NCCL_OFI_WARN("Invalid type: %d", type);
		return -EINVAL;
	}

	mr_attr->requested_key = mr_key;

	return ret;
}

int nccl_ofi_rdma_gin_ep_t::reg_mr(nccl_ofi_mr_ckey_ref ckey, int type,
			      nccl_ofi_gin_mr_handle_t **mhandle)
{
	int ret = 0;

	*mhandle = NULL;

	nccl_ofi_idpool_t *key_pool = this->domain.mr_rkey_pool;
	uint64_t mr_key = 0;

	if (key_pool->get_size() != 0) {
		mr_key = key_pool->allocate_id();
		if (OFI_UNLIKELY(mr_key == FI_KEY_NOTAVAIL)) {
			NCCL_OFI_WARN("MR key allocation failed");
			return -ENOMEM;
		}
	}

	struct fi_mr_attr mr_attr = {};
	uint64_t regattr_flags = 0;
	auto ret_handle =
		std::make_unique<nccl_ofi_gin_mr_handle_t>(this->domain, num_rails, mr_key);

	ret = set_mr_req_attr(ret_handle->mr_key, ckey, &regattr_flags, type, &mr_attr);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not set registration request attributes, dev: %d",
			      domain.get_device()->dev_id);
		return ret;
	}

	/* Register memory on each rail */
	for (uint16_t rail_id = 0; rail_id != num_rails; ++rail_id) {
		auto mr_result = nccl_ofi_ofiutils_mr_regattr(domain.get_ofi_domain(rail_id),
							      &mr_attr, regattr_flags);
		if (OFI_UNLIKELY(mr_result.is_failure())) {
			NCCL_OFI_WARN("Could not register memory on rail %u with flag %lu", rail_id,
				      regattr_flags);
			return mr_result.error_code;
		}
		ret_handle->set_mr(rail_id, mr_result.resource);

		if (endpoint_mr) {
			auto *ofi_ep = get_rail(rail_id).ofi_ep.get();
			ret = nccl_ofi_ofiutils_mr_bind_and_enable(ret_handle->get_mr(rail_id),
								   ofi_ep,
								   rail_id,
								   "data");
			if (ret != 0) {
				return ret;
			}
		}
	}

	*mhandle = ret_handle.release();
	return 0;
}

void nccl_ofi_rdma_gin_ep_t::dereg_mr(nccl_ofi_gin_mr_handle_t *handle_ptr)
{
	if (OFI_UNLIKELY(handle_ptr == NULL)) {
		NCCL_OFI_WARN("Attempted to deregister NULL memory region handle");
		return;
	}

	delete handle_ptr;
}

int nccl_ofi_rdma_gin_ep_t::freelist_regmr_fn(void *ep_ptr, void *data, size_t size, void **mhandle)
{
	auto ep = static_cast<nccl_ofi_rdma_gin_ep_t *>(ep_ptr);
	/* Setting ep to nullptr for the cache key -- we don't use the MR cache for GIN */
	auto ckey = nccl_ofi_mr_ckey_mk_vec(data, size, nullptr);
	return ep->reg_mr(&ckey, NCCL_PTR_HOST,
			  reinterpret_cast<nccl_ofi_gin_mr_handle_t **>(mhandle));
}

int nccl_ofi_rdma_gin_ep_t::freelist_deregmr_fn(void *handle)
{
	auto mr_handle = static_cast<nccl_ofi_gin_mr_handle_t *>(handle);

	delete mr_handle;

	return 0;
}

/**
 * Much of this taken from the same function in RDMA transport
 */
/**
 * EFA reports FI_RX_CQ_DATA as the way to avoid consuming receive buffers for
 * write-with-immediate. Endpoint-MR providers (CXI) deliver remote CQ data
 * without consuming a receive and do not advertise the mode bit.
 */
static inline bool gin_provider_needs_rx_cq_data(const struct fi_info *ref_info)
{
	return !(ref_info->domain_attr->mr_mode & FI_MR_ENDPOINT);
}

static inline void get_gin_hints(struct fi_info &hints, struct fi_info *ref_info)
{
	hints.caps = 0;

	/* Primary Capabilities */
	hints.caps = FI_MSG | FI_RMA | FI_HMEM;

	/* Secondary Capabilities. GIN requires FI_SOURCE for `fi_cq_readfrom`. */
	hints.caps |= FI_LOCAL_COMM | FI_REMOTE_COMM | FI_SOURCE;

	/* Mode. GIN sets `FI_RX_CQ_DATA` as an indirect way to disable unsolicited write */
	hints.mode = FI_CONTEXT | FI_CONTEXT2;
	if (gin_provider_needs_rx_cq_data(ref_info)) {
		hints.mode |= FI_RX_CQ_DATA;
	}

	hints.ep_attr->type = FI_EP_RDM;

	if (ref_info->domain_attr->mr_mode & FI_MR_ENDPOINT) {
		/* Endpoint-MR providers (CXI): mirror the mode the NET transport
		   negotiated; keys are provider-generated and as wide as the
		   provider says. */
		hints.domain_attr->mr_mode = FI_MR_HMEM | FI_MR_ALLOCATED | FI_MR_PROV_KEY |
					     FI_MR_ENDPOINT;
		hints.domain_attr->mr_key_size = ref_info->domain_attr->mr_key_size;
	} else {
		hints.domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM | FI_MR_VIRT_ADDR |
					     FI_MR_ALLOCATED | FI_MR_PROV_KEY;
		hints.domain_attr->mr_key_size = (size_t)ofi_nccl_mr_key_size();
	}
	hints.domain_attr->threading = FI_THREAD_COMPLETION;

	hints.domain_attr->control_progress =
		nccl_ofi_translate_progress_enum(ofi_nccl_progress_model.get());
	hints.domain_attr->data_progress =
		nccl_ofi_translate_progress_enum(ofi_nccl_progress_model.get());

	/* Endpoint-MR completions encode the sender rank in the upper 32 bits. */
	hints.domain_attr->cq_data_size = gin_required_cq_data_size(
		(ref_info->domain_attr->mr_mode & FI_MR_ENDPOINT) != 0);

	/* Copy domain and fabric names from previously opened net transport.
	   Note the domain and fabric names are required to obtain exactly one result. */
	hints.domain_attr->name = strdup(ref_info->domain_attr->name);
	hints.fabric_attr->name = strdup(ref_info->fabric_attr->name);
}

static inline struct fi_info *get_gin_info(struct fi_info *info)
{
	/* We need to call fi_getinfo again, but this time pass FI_RX_CQ_DATA */
	/* Start with a new info object and set GIN-specific hints */
	ofi_info_ptr gin_hints = ofi_info_ptr(fi_allocinfo());
	get_gin_hints(*gin_hints.get(), info);

	/*
	 * Select libfabric API version:
	 *   - GDA-capable EFA provider (GDAKI compiled in, DMA-BUF viable): 2.5,
	 *     the hardware-counter ABI the GDAKI plugin reuses via fi_open_ops.
	 *   - Otherwise (non-EFA / tcp / proxy-only build): 1.18 is sufficient;
	 *     the GIN endpoint hints (FI_RX_CQ_DATA, cq_data_size = 4) don't use
	 *     any 1.20+ features, so we avoid requesting a contract we don't
	 *     consume.
	 */
	uint32_t api_version;
	if (nccl_ofi_gin_gdaki_capable(info)) {
		api_version = FI_VERSION(2, 5);
	} else {
		api_version = FI_VERSION(1, 18);
	}

	struct fi_info *raw_results = nullptr;
	int ret = fi_getinfo(api_version, nullptr, nullptr, 0ULL, gin_hints.get(), &raw_results);
	if (ret != 0) {
		throw std::runtime_error(
			"fi_getinfo for GIN failed: " +
			std::string(fi_strerror(-ret)));
	};

	/* Wrap immediately so every throw path below frees the provider result
	 * list via fi_freeinfo() instead of leaking it. */
	ofi_info_ptr results(raw_results);

	/* There should only be exactly one result, that supports everything we asked for */
	if (results == nullptr) {
		throw std::runtime_error("No providers available for GIN");
	} else if (results->next != nullptr) {
		throw std::runtime_error("Unexpectedly got more than one GIN provider");
	} else if (gin_provider_needs_rx_cq_data(info) && !(results->mode & FI_RX_CQ_DATA)) {
		throw std::runtime_error("Provider does not support FI_RX_CQ_DATA");
	} else if (results->domain_attr->cq_data_size < gin_hints->domain_attr->cq_data_size) {
		throw std::runtime_error("Provider CQ data is narrower than the GIN wire format");
	}

	return results.release();
}

nccl_ofi_gin_ep_rail_t::nccl_ofi_gin_ep_rail_t(uint16_t rail_id_, nccl_net_ofi_domain_t &domain,
			      size_t configured_rx_cq_size,
			      ofi_info_ptr &&cached_gin_info)
    : rail_id(rail_id_)
{
	ofi_domain_ptr &ofi_domain = domain.get_ofi_domain(rail_id);

	/* Create cq */
	fi_cq_attr cq_attr = {};
	cq_attr.format = FI_CQ_FORMAT_DATA;
	cq_attr.size = configured_rx_cq_size;
	auto cq_result = nccl_ofi_ofiutils_cq_create(ofi_domain, &cq_attr);
	if (OFI_UNLIKELY(cq_result.is_failure())) {
		NCCL_OFI_WARN("Couldn't open CQ. RC: %d, ERROR: %s", cq_result.error_code,
			      fi_strerror(-cq_result.error_code));
		throw std::runtime_error("GIN: ofi cq creation failed");
	}

	this->rail_cq = std::move(cq_result.resource);

	/* Create an av */
	auto av_result = nccl_ofi_ofiutils_av_create(ofi_domain);
	if (av_result.is_failure()) {
		throw std::runtime_error("Failed to create av");
	}

	av = std::move(av_result.resource);

	/* Reuse the caller-provided fi_info (already GIN-negotiated for this
	 * rail's device) when available, avoiding a redundant fi_getinfo()
	 * call; otherwise fetch it ourselves. Ownership is only taken here
	 * (via std::move), after CQ/AV creation above have already succeeded,
	 * so if either of those throws, the still-owning caller-side unique_ptr
	 * cleans up the fi_info list during stack unwinding instead of leaking
	 * it. */
	ofi_info_ptr gin_info(
		cached_gin_info ? std::move(cached_gin_info)
				: ofi_info_ptr(get_gin_info(
				domain.get_device()->get_ofi_info(rail_id))));

	/* Create ep */
	auto ep_result = nccl_ofi_ofiutils_ep_create(gin_info.get(), ofi_domain, av, this->rail_cq);
	if (ep_result.is_failure()) {
		throw std::runtime_error("Failed to create ep");
	}

	ofi_ep = std::move(ep_result.resource);
}

nccl_ofi_gin_mr_handle_t::nccl_ofi_gin_mr_handle_t(nccl_net_ofi_domain_t &domain_arg,
						   uint16_t num_rails, uint64_t mr_key_arg)
    : nccl_net_ofi_mr_handle_t(mr_key_arg), mr(num_rails), domain(domain_arg)
{
	auto &mr_rkey_pool = *(domain.mr_rkey_pool);

	if (mr_rkey_pool.get_size() != 0) {
		mr_key = mr_rkey_pool.allocate_id();
		if (OFI_UNLIKELY(mr_key == FI_KEY_NOTAVAIL)) {
			NCCL_OFI_WARN("MR key allocation failed");
			throw std::runtime_error("MR key allocation failed");
		}
	}
}

nccl_ofi_gin_mr_handle_t::~nccl_ofi_gin_mr_handle_t()
{
	auto *mr_rkey_pool = domain.mr_rkey_pool;

	if (mr_rkey_pool->get_size() != 0) {
		mr_rkey_pool->free_id(this->mr_key);
	}
}

void nccl_ofi_gin_resources::post_rx_buffs_on_rail(nccl_ofi_gin_ep_rail_t &rail, size_t num_buffers)
{
	for (size_t i = 0; i < num_buffers; i++) {
		recv_reqs.emplace_back(*this, rail);
		int ret = recv_reqs.back().post_or_add_pending();
		if (ret != 0) {
			throw std::runtime_error("Failed to post recv req");
		}
	}
}

nccl_ofi_gin_resources::nccl_ofi_gin_resources(
	const std::shared_ptr<nccl_net_ofi_ep_t> &ep_arg, uint32_t nranks)
    : ep_holder(ep_arg),
      gin_ep(ep_arg->get_domain(), nranks),
      dev(ep_arg->get_domain().get_device()->dev_id),
      req_fl(nullptr, &freelist_deleter),
      rx_buff_fl(nullptr, &freelist_deleter),
      ack_send_fl(nullptr, &freelist_deleter),
      comm_id_pool(NCCL_GIN_MAX_COMMS)
{
	auto num_rails = gin_ep.get_num_rails();
	/* Create freelist for RX buffers */
	constexpr size_t num_buffers = 2048; /* TODO param*/
	assert_always(num_rails > 0 && num_buffers % num_rails == 0);
	const size_t num_buffers_per_rail = num_buffers / num_rails;

	nccl_ofi_freelist *rx_buff_fl_tmp = nullptr;
	rx_buff_fl_tmp = new nccl_ofi_freelist(sizeof(nccl_net_ofi_gin_signal_metadata_msg_t),
					       num_buffers, 0, num_buffers, nullptr, nullptr,
					       gin_ep.freelist_regmr_fn, gin_ep.freelist_deregmr_fn,
					       &gin_ep, 1, "GIN Rx Buffers", true);
	this->rx_buff_fl.reset(rx_buff_fl_tmp);

	/* Create freelist for ACK send buffers */
	this->ack_send_fl.reset(new nccl_ofi_freelist(sizeof(gin_ack_msg), 64, 64, 0, nullptr,
						      nullptr, gin_ep.freelist_regmr_fn,
						      gin_ep.freelist_deregmr_fn, &gin_ep, 1,
						      "GIN ACK Send", true));

	/* Create the receive pool for all rails */
	recv_reqs.reserve(num_buffers);
	for (uint16_t r = 0; r < num_rails; ++r) {
		post_rx_buffs_on_rail(gin_ep.get_rail(r), num_buffers_per_rail);
	}

	/* Freelist for requests */
	nccl_ofi_freelist *req_fl_tmp = nullptr;
	req_fl_tmp = new nccl_ofi_freelist(sizeof(nccl_net_ofi_gin_union_req), 1024, 1024, 0,
					   nullptr, nullptr, "GIN Requests", true);
	this->req_fl.reset(req_fl_tmp);

	init_flush_buffers(num_rails);

	if (endpoint_mr) {
		gin_ep.remote_write_cb = [this](struct fi_cq_data_entry *e, fi_addr_t src,
						uint16_t rail) NO_THREAD_SAFETY_ANALYSIS {
			auto *gin_comm = find_comm(GIN_IMM_GET_COMM_ID(e->data));
			if (OFI_UNLIKELY(gin_comm == nullptr)) {
				NCCL_OFI_WARN("Dropping GIN remote-write completion "
					      "for stale comm id");
				return 0;
			}
			return gin_comm->handle_signal_write_completion(e, src, rail);
		};
	}
}

uint32_t nccl_ofi_gin_resources::get_shared_cq_window_limit(uint32_t nranks) const
{
	return gin_worst_case_window_get_limit_with_reserve(
		gin_ep.get_configured_rx_cq_size(), nranks,
		gin_ep.get_rx_cq_reserve_margin(), ref_cnt);
}

void nccl_ofi_gin_resources::increment_ref_cnt()
{
	this->ref_cnt++;
	refresh_shared_cq_window_limits();
}

void nccl_ofi_gin_resources::release()
{
	assert_always(this->ref_cnt > 0);
	this->ref_cnt--;
	refresh_shared_cq_window_limits();
}

void nccl_ofi_gin_resources::refresh_shared_cq_window_limits()
{
	/* NCCL setup path assigns one proxy thread per listen_seq % nthreads bucket,
	 * so a given endpoint shared-CQ window bookkeeping is single-threaded during
	 * steady-state use. Teardown keeps that invariant too: ncclGinHostFinalize()
	 * stops and joins every proxy thread before closeColl() reaches release() here
	 * (see nccl-src/src/gin/gin_host.cc). */
	if (gin_ep.get_uses_posted_recv_flow()) {
		return;
	}
	for (auto *comm : gin_comms) {
		if (comm != nullptr) {
			comm->update_shared_cq_window_limit(
				get_shared_cq_window_limit(comm->get_nranks()));
		}
	}
}

void nccl_ofi_gin_resources::maybe_warn_cq_sizing_mismatch(uint32_t requested_nranks)
{
	/* The static CQ budget on gin_worst_case_window_limit() is still computed
	 * once for the CQ owned by this cached endpoint, but writedata-routed
	 * communicators now divide that allowance by ref_cnt whenever they derive
	 * a per-comm sender window. This warning therefore only needs to describe
	 * nranks mismatches across reused communicators, not a multi-comm gap. */
	if (requested_nranks == gin_ep.get_cq_sizing_nranks()) {
		return;
	}

	if (!cq_sizing_mismatch_warned.exchange(true, std::memory_order_relaxed)) {
		NCCL_OFI_INFO(NCCL_NET,
			      "GIN RX CQ already sized for nranks=%u (CQ entries=%zu); "
			      "later connect() requested nranks=%u and reuses the "
			      "existing CQ because live CQ resize is unsupported",
			      gin_ep.get_cq_sizing_nranks(), gin_ep.get_configured_rx_cq_size(),
			      requested_nranks);
	}
}

void nccl_ofi_gin_resources::init_flush_buffers(uint16_t num_rails)
{
	size_t flush_buff_size = NCCL_OFI_DEFAULT_CPU_CACHE_LINE_SIZE * num_rails;

	auto cleanup = [&]() {
		if (flush_buff_gpu_mr_handle) { gin_ep.dereg_mr(flush_buff_gpu_mr_handle); flush_buff_gpu_mr_handle = nullptr; }
		if (flush_buff_gpu) { nccl_net_ofi_gpu_mem_free(flush_buff_gpu); flush_buff_gpu = nullptr; }
		if (flush_buff_mr_handle) { gin_ep.dereg_mr(flush_buff_mr_handle); flush_buff_mr_handle = nullptr; }
		if (flush_buff) { std::free(flush_buff); flush_buff = nullptr; }
	};

	flush_buff = std::aligned_alloc(NCCL_OFI_DEFAULT_CPU_CACHE_LINE_SIZE, flush_buff_size);
	if (!flush_buff) {
		throw std::runtime_error("Failed to allocate flush buffer");
	}

	try {
		auto ckey = nccl_ofi_mr_ckey_mk_vec(flush_buff, flush_buff_size, nullptr);
		if (gin_ep.reg_mr(&ckey, NCCL_PTR_HOST, &flush_buff_mr_handle) != 0) {
			throw std::runtime_error("Failed to register flush buffer");
		}

		if (nccl_net_ofi_gpu_mem_alloc(&flush_buff_gpu, flush_buff_size) != 0) {
			throw std::runtime_error("Failed to allocate GPU flush buffer");
		}

		auto gpu_ckey = nccl_ofi_mr_ckey_mk_vec(flush_buff_gpu, flush_buff_size, nullptr);
		if (gin_ep.reg_mr(&gpu_ckey, NCCL_PTR_CUDA, &flush_buff_gpu_mr_handle) != 0) {
			throw std::runtime_error("Failed to register GPU flush buffer");
		}

		/* Fill GPU buffer with sentinel. Polling host buffer for this
		   value after fi_read confirms all prior operations are fenced. */
		uint64_t sentinel_line[NCCL_OFI_DEFAULT_CPU_CACHE_LINE_SIZE / sizeof(uint64_t)];
		for (auto &v : sentinel_line) {
			v = NCCL_OFI_GIN_FLUSH_SENTINEL_VAL;
		}
		for (uint16_t i = 0; i < num_rails; i++) {
			if (nccl_net_ofi_gpu_mem_copy_host_to_device(
				    static_cast<uint8_t *>(flush_buff_gpu) +
					    (NCCL_OFI_DEFAULT_CPU_CACHE_LINE_SIZE * i),
				    sentinel_line, sizeof(sentinel_line)) != 0) {
				throw std::runtime_error("Failed to init GPU flush buffer sentinel");
			}
		}
	} catch (...) {
		cleanup();
		throw;
	}
}

nccl_ofi_gin_resources::~nccl_ofi_gin_resources()
{
	/* Drop our slot in the registry. The entry is a weak_ptr so a stale one
	   would be harmless, but releasing keeps the table from accumulating dead
	   keys as endpoints come and go. */
	gin_resources_registry().release(ep_holder.get(), this);

	/* Deregister and free flush buffer before closing endpoints */
	if (flush_buff_gpu_mr_handle) {
		gin_ep.dereg_mr(flush_buff_gpu_mr_handle);
		flush_buff_gpu_mr_handle = nullptr;
	}
	if (flush_buff_gpu) {
		nccl_net_ofi_gpu_mem_free(flush_buff_gpu);
		flush_buff_gpu = nullptr;
	}
	if (flush_buff_mr_handle) {
		gin_ep.dereg_mr(flush_buff_mr_handle);
		flush_buff_mr_handle = nullptr;
	}
	if (flush_buff) {
		std::free(flush_buff);
		flush_buff = nullptr;
	}

	/*
	 * For non-FI_MR_ENDPOINT providers, we close the OFI endpoint here,
	 * then let the remaining members clean up in reverse declaration
	 * order.
	 *
	 * For FI_MR_ENDPOINT providers, we skip closing the endpoint here so
	 * that ack_send_fl and rx_buff_fl (whose MRs are bound to gin_ep) can
	 * deregister those MRs before gin_ep tears down the bound endpoints
	 * below, in reverse declaration order; see the gin_ep declaration in
	 * nccl_ofi_gin_resources.h for the matching member-ordering invariant.
	 *
	 * recv_reqs holds fi_recv()s posted against rx_buff_fl's buffers, so
	 * those must be retired before rx_buff_fl deregisters below; otherwise
	 * a still-outstanding recv could complete (or the provider could DMA
	 * into it) after the MR backing it is gone. cancel_and_drain_recv_reqs()
	 * drains with a bounded retry budget. If cancellation/completion does
	 * not retire every receive, endpoints are closed before the MRs are
	 * released to prevent further DMA into their backing buffers.
	 */
	if (endpoint_mr) {
		if (!cancel_and_drain_recv_reqs()) {
			gin_ep.close_ofi_eps();
		}
	} else {
		gin_ep.close_ofi_eps();
	}
}

int nccl_ofi_gin_resources::progress()
{
	int ret = gin_ep.process_cq();
	if (OFI_UNLIKELY(ret < 0)) {
		return ret;
	}

	return retry_pending_reqs();
}

int nccl_ofi_gin_resources::retry_pending_reqs()
{
	for (auto it = pending_requests.begin(); it != pending_requests.end();) {
		auto req = *it;
		int ret = req->post();
		if (ret == 0) {
			it = pending_requests.erase(it);
		} else if (ret == -FI_EAGAIN) {
			return 0;
		} else {
			return ret;
		}
	}

	return 0;
}

bool nccl_ofi_gin_resources::cancel_and_drain_recv_reqs()
{
	/* Remove receives that have not yet posted so retry_pending_reqs cannot
	 * repost them after cancellation begins. */
	for (auto &req : recv_reqs) {
		auto pending = std::find(pending_requests.begin(), pending_requests.end(), &req);
		if (pending != pending_requests.end()) {
			pending_requests.erase(pending);
		}
		req.cancel();
	}

	constexpr size_t max_drain_iterations = 10000;
	for (size_t attempt = 0; attempt < max_drain_iterations; ++attempt) {
		const bool outstanding = std::any_of(recv_reqs.begin(), recv_reqs.end(),
						      [](const auto &req) {
							      return req.has_outstanding_post();
						      });
		if (!outstanding) {
			return true;
		}
		int ret = gin_ep.process_cq();
		if (OFI_UNLIKELY(ret < 0)) {
			NCCL_OFI_WARN("Error draining CQ while cancelling GIN receives: %d", ret);
			return false;
		}
		if (ret == 0) {
			std::this_thread::yield();
		}
	}

	NCCL_OFI_WARN("Timed out draining cancelled GIN receives; "
		       "closing endpoints before MR teardown");
	return false;
}
