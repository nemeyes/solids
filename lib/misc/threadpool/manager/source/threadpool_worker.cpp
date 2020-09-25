#include "threadpool_worker.h"
#include "sld_threadpool_manager.h"

namespace solids
{
namespace lib
{
namespace misc
{
namespace threadpool
{

	worker::core::core(solids::lib::misc::threadpool::worker* front, solids::lib::misc::threadpool::manager* mngr, int32_t id)
		: _front(front)
		, _manager(mngr)
		, _id(id)
	{

	}

	worker::core::~core(void)
	{

	}

	int32_t	worker::core::id(void)
	{
		return _id;
	}

	BOOL worker::core::is_running(void)
	{
		BOOL status = FALSE;
		{
			worker::core::tp_worker_scopped_lock mutex(&_srwl);
			if (_tp_ready_workers.size() > 0)
				status = TRUE;
		}
		return status;
	}

	WORKERFUNC worker::core::worker_cb(void)
	{
		return &worker::core::tp_worker_cb;
	}

	void worker::core::set_work(PTP_WORK work)
	{
		_work = work;
	}

	void worker::core::run(const uint8_t * bytes, int32_t nbytes, void * user)
	{
		_execute(bytes, nbytes, user);
	}

	void worker::core::_execute(const uint8_t * bytes, int32_t nbytes, void * user)
	{
		worker::core::tp_worker_scopped_lock mutex(&_srwl);
		std::shared_ptr<worker::core::tp_worker> worker = std::shared_ptr<worker::core::tp_worker>(new worker::core::tp_worker);
		worker->handler = this;
		if (bytes && nbytes > 0)
		{
			worker->nbytes = nbytes;
			worker->bytes = static_cast<uint8_t*>(malloc(worker->nbytes));
			memmove(worker->bytes, bytes, worker->nbytes);
		}
		if (user)
			worker->user = user;
		if (worker)
			_tp_ready_workers.push_back(worker);

		if (_manager)
			_manager->run_worker(_work);
	}

	void worker::core::__execute(void)
	{
		std::shared_ptr<worker::core::tp_worker> worker = NULL;
		{
			worker::core::tp_worker_scopped_lock mutex(&_srwl);
			if (_tp_ready_workers.size() > 0)
			{
				worker = _tp_ready_workers.front();
				_tp_ready_workers.pop_front();
			}
		}
		if (worker && _front)
			_front->execute(worker->bytes, worker->nbytes, worker->user);
	}

	void worker::core::tp_worker_cb(PTP_CALLBACK_INSTANCE instance, PVOID parameter, PTP_WORK work)
	{
		solids::lib::misc::threadpool::worker* front = static_cast<solids::lib::misc::threadpool::worker*>(parameter);
		if (front && front->_core)
			front->_core->__execute();
	}

};
};
};
};