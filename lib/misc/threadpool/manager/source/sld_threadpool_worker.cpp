#include "sld_threadpool_worker.h"
#include "threadpool_worker.h"

namespace solids
{
namespace lib
{
namespace misc
{
namespace threadpool
{

	worker::worker(solids::lib::misc::threadpool::manager * mngr, int32_t id)
	{
		_core = new solids::lib::misc::threadpool::worker::core(this, mngr, id);
	}

	worker::~worker(void)
	{
		if (_core)
			delete _core;
		_core = NULL;
	}

	int32_t	worker::id(void)
	{
		return _core->id();
	}

	BOOL worker::is_running(void)
	{
		return _core->is_running();
	}

	WORKERFUNC worker::worker_cb(void)
	{
		return _core->worker_cb();
	}

	void worker::set_work(PTP_WORK work)
	{
		return _core->set_work(work);
	}

	void worker::run(const uint8_t* bytes, int32_t nbytes, void* user)
	{
		return _core->run(bytes, nbytes, user);
	}

};
};
};
};

