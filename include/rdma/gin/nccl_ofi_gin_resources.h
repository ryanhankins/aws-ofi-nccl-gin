/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_GIN_RESOURCES_H
#define NCCL_OFI_RDMA_GIN_RESOURCES_H

#include "rdma/fabric.h"

#include <array>
#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <unordered_map>

#include "rdma/gin/nccl_ofi_gin_reqs.h"
#include "rdma/gin/nccl_ofi_gin_comm_id_pool.h"
#include "rdma/gin/nccl_ofi_gin_types.h"
#include "nccl_ofi_gin_base.h"
#include "nccl_ofi_freelist.h"
#include "nccl_ofi_scheduler.h"
#include "nccl_ofi_tsa.h"

static inline void freelist_deleter(nccl_ofi_freelist *fl)
{
	delete fl;
}

/**
 * A rail of the GIN endpoint
 */
struct nccl_ofi_gin_ep_rail_t {
	/**
	 * @param cached_gin_info: Optional pre-fetched GIN-negotiated fi_info,
	 *   as returned by get_gin_info(). Pass non-null only when the caller
	 *   already called get_gin_info() for this exact rail's device (e.g.
	 *   rail 0, whose result the EP constructor also needs to inspect the
	 *   negotiated mode bit) so we avoid a second redundant fi_getinfo()
	 *   call. If null, this rail calls get_gin_info() itself. Passed by
	 *   rvalue-ref unique_ptr (rather than a raw owning pointer) so
	 *   ownership transfer is exception-safe: this constructor can throw
	 *   before it gets a chance to wrap the pointer (e.g. CQ/AV creation
	 *   failure) without leaking the provider fi_info list.
	 */
	nccl_ofi_gin_ep_rail_t(uint16_t rail_id_, nccl_net_ofi_domain_t &domain,
			      size_t configured_rx_cq_size,
			      ofi_info_ptr &&cached_gin_info = ofi_info_ptr(nullptr));

	~nccl_ofi_gin_ep_rail_t() = default;
	nccl_ofi_gin_ep_rail_t(nccl_ofi_gin_ep_rail_t &&other) = default;

	const uint16_t rail_id;

	/* Pointer to completion queue */
	ofi_cq_ptr rail_cq;

	/* Pointer to address vector */
	ofi_av_ptr av;

	/* Pointer to endpoint */
	ofi_ep_ptr ofi_ep;
};

/**
 * The GIN endpoint type
 */
class nccl_ofi_rdma_gin_ep_t : public nccl_ofi_gin_ep_t {
public:
	/**
	 * Create a GIN EP using the provided domain object
	 *
	 * @param domain_arg: Domain object from net transport
	 */
	nccl_ofi_rdma_gin_ep_t(nccl_net_ofi_domain_t &domain_arg,
		       uint32_t cq_sizing_nranks_arg);

	/**
	 * Handler for remote-write CQ entries that carry no op_context. Endpoint-MR
	 * providers (CXI) deliver write-with-immediate data without consuming a
	 * posted receive, so there is no rx request to dispatch through. Set by
	 * nccl_ofi_gin_resources; only used when endpoint_mr is true.
	 */
	std::function<int(struct fi_cq_data_entry *, fi_addr_t, uint16_t)> remote_write_cb;

	nccl_ofi_rdma_gin_ep_t(const nccl_ofi_rdma_gin_ep_t &) = delete;
	nccl_ofi_rdma_gin_ep_t &operator=(const nccl_ofi_rdma_gin_ep_t &) = delete;

	~nccl_ofi_rdma_gin_ep_t() override;

	uint16_t get_num_rails() const
	{
		return num_rails;
	}

	nccl_ofi_gin_ep_rail_t &get_rail(uint16_t rail_id)
	{
		return rails[rail_id];
	}

	/**
	 * @brief Accessor to the net transport domain backing this GIN EP.
	 *
	 * Exposed so GDAKI createContext can reuse the same libfabric domain
	 * as the proxy path. Same fid_domain → same PD → memory registered
	 * via extGin->regMrSym produces keys (lkey/rkey) that are valid on
	 * the GDAKI endpoint opened in createContext.
	 */
	nccl_net_ofi_domain_t &get_domain()
	{
		return domain;
	}

	nccl_net_ofi_scheduler *get_scheduler()
	{
		return scheduler;
	}

	size_t get_configured_rx_cq_size() const
	{
		return configured_rx_cq_size;
	}

	size_t get_rx_cq_reserve_margin() const
	{
		return rx_cq_reserve_margin;
	}

	uint32_t get_cq_sizing_nranks() const
	{
		return cq_sizing_nranks;
	}
	/**
	 * Register memory region with this endpoint
	 *
	 * @param ckey: cache key, created by nccl_ofi_mr_ckey_mk_vec or nccl_ofi_mr_ckey_mk_dmabuf
	 */
	int reg_mr(nccl_ofi_mr_ckey_ref ckey, int type, nccl_ofi_gin_mr_handle_t **mhandle);

	void dereg_mr(nccl_ofi_gin_mr_handle_t *handle_ptr);

	/**
	 * Memory de/registration interfaces suitable for freelist use
	 */
	static int freelist_regmr_fn(void *ep_ptr, void *data, size_t size, void **mhandle);
	static int freelist_deregmr_fn(void *handle);

	/**
	 * Process completions for all rails
	 *
	 * @return: 0 or -errno for error
	 */
	int process_cq();


	bool get_uses_posted_recv_flow() const
	{
		return uses_posted_recv_flow;
	}

	std::unique_ptr<gin_flow_policy> make_flow_policy(nccl_ofi_gin_resources &resources,
					      uint32_t nranks) const;

	/**
	 * Close Libfabric endpoint object. For non-endpoint-MR providers, this
	 * should be done before memory can be deregistered.
	 */
	void close_ofi_eps();

	std::mutex ep_lock;
private:
	nccl_net_ofi_domain_t &domain;

	uint16_t num_rails;

	std::vector<nccl_ofi_gin_ep_rail_t> rails;

	nccl_net_ofi_scheduler *scheduler;

	/* Cached from param at construction; avoids mutex in CQ loop */
	size_t cq_process_max_iter;
	size_t rx_cq_reserve_margin;
	bool uses_posted_recv_flow = false;

	/* CQ sizing is fixed at EP creation, then clamped to the smallest CQ
	 * depth fi_cq_open reports across rails. Later connects on the same
	 * cached endpoint reuse that value because libfabric does not support
	 * live CQ resize/replacement without rebuilding the endpoint. */
	size_t configured_rx_cq_size;
	const uint32_t cq_sizing_nranks;

	using process_completions_fn = int (nccl_ofi_rdma_gin_ep_t::*)(
		struct fi_cq_data_entry *cq_entry, fi_addr_t *src_addrs, uint64_t num_cqes,
		uint16_t rail_id);


	/*
	 * FI_REMOTE_WRITE completion handling diverges by provider:
	 *  - Providers that do not request FI_RX_CQ_DATA (e.g. CXI) report
	 *    target-side fi_writedata completions against the bound MR, not a
	 *    posted fi_recv() context, so they must be classified by
	 *    completion flags/immediate data and routed before op_context is
	 *    ever touched.
	 *  - Providers that request FI_RX_CQ_DATA (e.g. EFA) always deliver
	 *    writedata completions against a posted-recv op_context.
	 * process_completions_impl is selected once at construction from the
	 * negotiated mode bit, so the CQ loop never has to branch on
	 * cq_entry->flags per completion. CQ error entries are handled in a
	 * single common path (see gin_process_cq_rail()) since the volume of
	 * error completions doesn't warrant the same per-completion dispatch
	 * cost avoidance.
	 */
	process_completions_fn process_completions_impl;

	/**
	 * Handler for list of CQ entries. Dispatches to the provider-specific
	 * implementation selected at construction time.
	 */
	int gin_process_completions(struct fi_cq_data_entry *cq_entry, fi_addr_t *src_addrs,
				    uint64_t num_cqes, uint16_t rail_id)
	{
		return (this->*process_completions_impl)(cq_entry, src_addrs, num_cqes, rail_id);
	}

	/* CXI-like: route FI_REMOTE_WRITE completions directly by flags +
	 * immediate data before touching op_context. */
	int gin_process_completions_writedata_routed(struct fi_cq_data_entry *cq_entry,
						     fi_addr_t *src_addrs, uint64_t num_cqes,
						     uint16_t rail_id);
	/* EFA-like: writedata completions always carry a valid posted-recv
	 * op_context, so every completion dispatches the same way. */
	int gin_process_completions_posted_recv(struct fi_cq_data_entry *cq_entry,
						fi_addr_t *src_addrs, uint64_t num_cqes,
						uint16_t rail_id);

	/* Shared op_context dispatch used by both completion implementations
	 * above for every completion that isn't routed by MR/immediate data. */
	int dispatch_via_op_context(struct fi_cq_data_entry *cq_entry, fi_addr_t src_addr,
				    uint16_t rail_id);

	/**
	 * CQ error entry handler
	 */
	int gin_process_error_entry(struct fi_cq_err_entry *err_entry, struct fid_cq *cq,
				    uint16_t rail_id);
	/**
	 * Process all completions for the given rail
	 */
	int gin_process_cq_rail(uint16_t rail_id);
};

/**
 * Represents a GIN local memory registration
 */
class nccl_ofi_gin_mr_handle_t : public nccl_net_ofi_mr_handle_t {
public:
	nccl_ofi_gin_mr_handle_t(nccl_net_ofi_domain_t &domain_arg, uint16_t num_rails,
				 uint64_t mr_key_arg);

	~nccl_ofi_gin_mr_handle_t();

	/**
	 * @brief	Get first MR key for GIN MR handle
	 * 		This interface isn't necessary for GIN.
	 */
	int get_mr_key(uint64_t *mr_key_ptr) override
	{
		return -ENOTSUP;
	}

	void set_mr(uint16_t rail_id, ofi_mr_ptr &mr_ptr)
	{
		assert(rail_id < mr.size());
		this->mr[rail_id] = std::move(mr_ptr);
	}

	struct fid_mr *get_mr(uint16_t rail_id) const
	{
		assert(rail_id < mr.size());
		return mr[rail_id].get();
	}

private:
	/* Array of size `num_rails' */
	std::vector<ofi_mr_ptr> mr;

	/* Back-pointer to net domain */
	nccl_net_ofi_domain_t &domain;
};

/**
 * Registry mapping a net endpoint to the GIN resources built on top of it.
 *
 * GIN owns this endpoint->resources association itself so that the net
 * transport does not need to know GIN exists. Entries are weak references:
 * the resources live only as long as some GIN comm holds the shared_ptr
 * returned by get(), and the slot is dropped by release() when the last comm
 * lets go.
 *
 * TODO: This registry only exists because the net transport, not the common
 * layer, owns endpoint creation and caching (device::get_ep / domain::
 * create_endpoint / ep_table). Once that machinery is lifted to the common
 * layer, GIN endpoints can be cached there the same way net endpoints are and
 * this GIN-private registry (and its global state) goes away.
 */
class nccl_ofi_gin_resources_registry {
public:
	/* Return the resources for `ep`, creating them on first use. */
	std::shared_ptr<nccl_ofi_gin_resources>
	get(const std::shared_ptr<nccl_net_ofi_ep_t> &ep, uint32_t nranks);

	/* Drop the slot for `ep`; called by the resources destructor. */
	void release(nccl_net_ofi_ep_t *ep);

private:
	std::mutex lock;
	std::unordered_map<nccl_net_ofi_ep_t *, std::weak_ptr<nccl_ofi_gin_resources>>
		by_ep;
};

/* Process-wide registry instance. */
nccl_ofi_gin_resources_registry &gin_resources_registry();

/**
 * Resources associated with a plugin per-thread endpoint
 */
class nccl_ofi_gin_resources {
public:
	nccl_ofi_gin_resources(const std::shared_ptr<nccl_net_ofi_ep_t> &ep_arg,
				uint32_t nranks);

	~nccl_ofi_gin_resources();

	/* Delete copy constructor */
	nccl_ofi_gin_resources(const nccl_ofi_gin_resources &) = delete;

	/** Methods **/

	nccl_ofi_freelist *get_rx_buff_fl()
	{
		return rx_buff_fl.get();
	}

	/**
	 * Get GIN communicator with given comm_id from map. Throw exception if
	 * not found.
	 */
	nccl_ofi_rdma_gin_put_comm &get_comm(uint16_t comm_id)
	{
		if (OFI_UNLIKELY(comm_id >= NCCL_GIN_MAX_COMMS || gin_comms[comm_id] == nullptr)) {
			NCCL_OFI_WARN("Invalid comm_id %d", comm_id);
			throw std::runtime_error("Failed to find comm_id");
		}

		return *gin_comms[comm_id];
	}

	nccl_ofi_rdma_gin_put_comm *find_comm(uint16_t comm_id) const
	{
		if (OFI_UNLIKELY(comm_id >= NCCL_GIN_MAX_COMMS)) {
			return nullptr;
		}
		return gin_comms[comm_id];
	}

	/**
	 * Set a GIN communicator with given comm_id in map. Throw exception if
	 * comm_id already exists. Caller must hold ep_lock.
	 */
	void set_comm(uint16_t comm_id, nccl_ofi_rdma_gin_put_comm &comm)
	{
		if (OFI_UNLIKELY(comm_id >= NCCL_GIN_MAX_COMMS || gin_comms[comm_id] != nullptr)) {
			NCCL_OFI_WARN("Failed to insert duplicate comm_id %d", comm_id);
			throw std::runtime_error("Failed to insert comm_id");
		}
		gin_comms[comm_id] = &comm;
	}

	/**
	 * Remove a GIN communicator with given comm_id from map. Throw exception if
	 * comm_id does not exist. Caller must hold ep_lock.
	 */
	void remove_comm(uint16_t comm_id)
	{
		if (OFI_UNLIKELY(comm_id >= NCCL_GIN_MAX_COMMS || gin_comms[comm_id] == nullptr)) {
			NCCL_OFI_WARN("Failed to remove comm_id %d (not found or out of range)", comm_id);
			throw std::runtime_error("Failed to remove comm_id");
		}
		gin_comms[comm_id] = nullptr;
	}

	nccl_ofi_rdma_gin_ep_t &get_ep()
	{
		return gin_ep;
	}

	int get_dev() const
	{
		return dev;
	}

	void maybe_warn_cq_sizing_mismatch(uint32_t requested_nranks);

	uint32_t get_shared_cq_window_limit(uint32_t nranks) const;

	size_t alloc_comm_id()
	{
		return comm_id_pool.allocate();
	}

	void rollback_unpublished_comm_id(size_t id)
	{
		comm_id_pool.rollback_unpublished(id);
	}

	nccl_ofi_freelist *get_ack_send_fl()
	{
		return ack_send_fl.get();
	}

	void *get_flush_buff() const
	{
		return flush_buff;
	}

	void *get_flush_buff_gpu() const
	{
		return flush_buff_gpu;
	}

	nccl_ofi_gin_mr_handle_t *get_flush_buff_mr_handle() const
	{
		return flush_buff_mr_handle;
	}

	nccl_ofi_gin_mr_handle_t *get_flush_buff_gpu_mr_handle() const
	{
		return flush_buff_gpu_mr_handle;
	}

	/**
	 * Get a request from the freelist
	 *
	 * Note: this is doing some unflattering template manipulation to allow
	 * for a freelist that can store generic request types. Eventually,
	 * freelist should be refactored to support this properly.
	 */
	template <typename T, typename... U> T *get_req_from_pool(U &&...args)
	{
		static_assert(sizeof(T) <= sizeof(nccl_net_ofi_gin_union_req),
			      "Request size too large for freelist");
		auto freelist_elem = req_fl.get()->entry_alloc();
		if (OFI_UNLIKELY(freelist_elem == nullptr)) {
			throw std::runtime_error("Failed to allocate request from freelist");
		}

		/* Construct the request */
		new (freelist_elem->ptr) T(std::forward<U>(args)...);
		auto *req = static_cast<T *>(freelist_elem->ptr);

		/* Keep a backpointer to freelist element to free */
		req->set_fl_entry(freelist_elem);

		return req;
	}

	template <typename T> void return_req_to_pool(T *req)
	{
		/* Cache the fl_elem member since we will destruct the req. */
		auto *fl_elem = req->get_fl_entry();

		/* Run req's destructor */
		req->~T();
		req = nullptr;

		/* Return to freelist */
		req_fl.get()->entry_free(fl_elem);
	}

	/**
	 * Add a request which returned -FI_EAGAIN, to be retried later
	 */
	void add_pending_req(nccl_net_ofi_gin_op_req_t *req)
	{
		pending_requests.push_back(req);
	}

	/**
	 * Get next rail for transfer. Uses round-robin scheduling.
	 *
	 * @return	Next rail id
	 */
	uint16_t get_next_rail()
	{
		uint16_t rail_id = next_rail_id;
		next_rail_id = (next_rail_id + 1) % gin_ep.get_num_rails();
		return rail_id;
	}

	/**
	 * Progress completion queue and retry any pending requests.
	 * Caller must hold ep_lock.
	 */
	int progress();

	/**
	 * Called when a new communicator is associated with this resource object
	 */
	void increment_ref_cnt();

	/**
	 * Called when an associated communicator is closed
	 */
	void release();

private:
	void init_flush_buffers(uint16_t num_rails);

	/* === Tier 1 — accessed every CQ completion and/or iputSignal === */
	/* Keeps the underlying net endpoint alive for the resources' lifetime. */
	std::shared_ptr<nccl_net_ofi_ep_t> ep_holder;

	nccl_ofi_rdma_gin_ep_t gin_ep;

	/* Device id; cached for cheap access in tracepoint emission. */
	int dev;

	/* Requests pool used by all comms of this resource */
	// FIXME: Get rid of freelist_deleter and change to embedded
	std::unique_ptr<nccl_ofi_freelist, decltype(&freelist_deleter)> req_fl;

	/* Pool of buffers for recv requests */
	std::unique_ptr<nccl_ofi_freelist, decltype(&freelist_deleter)> rx_buff_fl;

	/* Reqs for RX buffers */
	std::vector<nccl_net_ofi_gin_recv_req_t> recv_reqs;

	/* === Tier 2 — accessed on iputSignal / ACK / retry paths === */
	/* For rail scheduling. Currently we do round-robin among rails. */
	uint16_t next_rail_id = 0;

	/* Pool of registered buffers for ACK send messages */
	std::unique_ptr<nccl_ofi_freelist, decltype(&freelist_deleter)> ack_send_fl;

	/**
	 * Queue of pending Libfabric requests to be retried
	 */
	std::deque<nccl_net_ofi_gin_op_req_t *> pending_requests;

	/* Pre-allocated host buffer used as the local DMA target for iflush
	   fi_read operations. One cache-line slot per rail. */
	void *flush_buff = nullptr;
	nccl_ofi_gin_mr_handle_t *flush_buff_mr_handle = nullptr;

	/* GPU-resident flush buffer — source of the local fi_read. The NIC
	   reads from this GPU buffer into flush_buff (host) via loopback,
	   which forces the NIC to drain its PCIe write/read queue. */
	void *flush_buff_gpu = nullptr;
	nccl_ofi_gin_mr_handle_t *flush_buff_gpu_mr_handle = nullptr;

	/* === Tier 3 — accessed only at connect/disconnect time === */
	nccl_ofi_gin_comm_id_pool comm_id_pool;

	/* Number of associated comms */
	size_t ref_cnt = 0;

	std::atomic<bool> cq_sizing_mismatch_warned{false};

	void refresh_shared_cq_window_limits();

	/* === Self-contained lookup — 8KB array at end to avoid pushing
	   other hot members apart === */
	std::array<nccl_ofi_rdma_gin_put_comm *, NCCL_GIN_MAX_COMMS> gin_comms{};

	/**
	 * Retry requests that were pending due to EAGAIN or lack of space in
	 * completion queue
	 */
	int retry_pending_reqs();

	/* Post all recv buffers for a given rail */
	void post_rx_buffs_on_rail(nccl_ofi_gin_ep_rail_t &rail, size_t num_buffers);
};

#endif
