#include "config.h"

#include <memory>
#include <unordered_map>

#include "unit_test.h"
#include "rdma/gin/nccl_ofi_gin_resources.h"

struct test_registry_entry {
	const int *identity;
	std::weak_ptr<int> resource;
};

int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	unit_test_init();

	std::unordered_map<int, test_registry_entry> entries;
	auto old_resource = std::make_shared<int>(1);
	const int *old_identity = old_resource.get();
	entries.emplace(7, test_registry_entry{old_identity, old_resource});

	/* The old weak entry has expired; get() installs a replacement before
	 * the old resource's destructor reaches registry cleanup. */
	old_resource.reset();
	auto replacement = std::make_shared<int>(2);
	entries[7] = test_registry_entry{replacement.get(), replacement};

	if (gin_registry_erase_if_same_resource(entries, 7, old_identity)) {
		printf("Stale resource cleanup erased a replacement registry entry\n");
		return 1;
	}
	if (entries.at(7).resource.lock() != replacement) {
		printf("Replacement resource was not retained in the registry\n");
		return 1;
	}

	if (!gin_registry_erase_if_same_resource(entries, 7, replacement.get()) ||
	    !entries.empty()) {
		printf("Current resource cleanup failed to erase its registry entry\n");
		return 1;
	}

	printf("PASS: GIN resource registry identity cleanup\n");
	return 0;
}
