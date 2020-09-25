#include "sld_threadpool_manager.h"
#include "threadpool_manager.h"

namespace solids
{
namespace lib
{
namespace misc
{
namespace threadpool
{

	manager::manager(void)
	{
		_core = new solids::lib::misc::threadpool::manager::core(this);
	}

	manager::~manager(void)
	{
		if (_core)
			delete _core;
		_core = NULL;
	}

	int32_t manager::initialize(int32_t nthread)
	{
		return _core->initialize(nthread);
	}

	int32_t manager::release(void)
	{
		return _core->release();
	}

	int32_t manager::run(int32_t ID, const uint8_t* bytes, int32_t nbytes, void* user)
	{
		return _core->run(ID, bytes, nbytes, user);
	}

	int32_t manager::add_worker(std::shared_ptr<solids::lib::misc::threadpool::worker> worker)
	{
		return _core->add_worker(worker);
	}

	int32_t manager::run_worker(PTP_WORK work)
	{
		return _core->run_worker(work);
	}

};
};
};
};

