#include "threadpool_manager.h"
#include "sld_threadpool_worker.h"
#include <sld_locks.h>


namespace solids
{
namespace lib
{
namespace misc
{
namespace threadpool
{

	manager::core::core(solids::lib::misc::threadpool::manager * front)
		: _front(front)
		, _threads(0)
		, _pool(NULL)
	{
		::InitializeCriticalSection(&_lock);
	}

	manager::core::~core(void)
	{
		::DeleteCriticalSection(&_lock);
	}

	int32_t manager::core::initialize(int32_t nthread)
	{
		solids::lib::autolock lock(&_lock);
		_threads = nthread;
		if (_pool)
			return solids::lib::misc::threadpool::manager::err_code_t::generic_fail;

		::InitializeThreadpoolEnvironment(&_cb_env);
		if (_threads == 0)
		{
			SYSTEM_INFO si;
			GetSystemInfo(&si);
			_threads = (si.dwNumberOfProcessors * 2) + 2;
		}

		do
		{
			_pool = ::CreateThreadpool(NULL);
			if (_pool == NULL)
				break;

			::SetThreadpoolThreadMaximum(_pool, _threads);
			BOOL status = ::SetThreadpoolThreadMinimum(_pool, 1);
			if (!status)
			{
				::CloseThreadpool(_pool);
				break;
			}

			_cleanup_group = ::CreateThreadpoolCleanupGroup();
			if (!_cleanup_group)
			{
				::CloseThreadpool(_pool);
				break;
			}

			::SetThreadpoolCallbackPool(&_cb_env, _pool);
			::SetThreadpoolCallbackCleanupGroup(&_cb_env, _cleanup_group, NULL);
		} while (FALSE);

		return solids::lib::misc::threadpool::manager::err_code_t::generic_fail;
	}

	int32_t manager::core::release(void)
	{
		solids::lib::autolock lock(&_lock);

		if (!_pool)
			return solids::lib::misc::threadpool::manager::err_code_t::generic_fail;

		::CloseThreadpoolCleanupGroupMembers(_cleanup_group, FALSE, NULL);
		::CloseThreadpoolCleanupGroup(_cleanup_group);
		::CloseThreadpool(_pool);
		_pool = NULL;

		_workers.clear();

		return solids::lib::misc::threadpool::manager::err_code_t::success;
	}

	int32_t manager::core::run(int32_t ID, const uint8_t* bytes, int32_t nbytes, void* user)
	{
		std::map<int32_t, std::shared_ptr<solids::lib::misc::threadpool::worker>>::iterator iter;
		iter = _workers.find(ID);
		if (iter != _workers.end())
		{
			std::shared_ptr<solids::lib::misc::threadpool::worker> worker = iter->second;
			worker->run(bytes, nbytes, user);
		}

		return solids::lib::misc::threadpool::manager::err_code_t::success;
	}

	int32_t manager::core::add_worker(std::shared_ptr<solids::lib::misc::threadpool::worker> worker)
	{
		if (!worker)
			return solids::lib::misc::threadpool::manager::err_code_t::generic_fail;
		{
			solids::lib::autolock mutex(&_lock);
			if (!_pool)
				return solids::lib::misc::threadpool::manager::err_code_t::generic_fail;
		}

		_workers.insert(std::make_pair(worker->id(), worker));

		PTP_WORK work = ::CreateThreadpoolWork(worker->worker_cb(), worker.get(), &_cb_env);
		worker->set_work(work);

		return solids::lib::misc::threadpool::manager::err_code_t::success;
	}

	int32_t manager::core::run_worker(PTP_WORK work)
	{
		if (!work)
			return solids::lib::misc::threadpool::manager::err_code_t::generic_fail;
		{
			solids::lib::autolock mutex(&_lock);
			if (!_pool)
				return solids::lib::misc::threadpool::manager::err_code_t::generic_fail;
		}
		::SubmitThreadpoolWork(work);

		return solids::lib::misc::threadpool::manager::err_code_t::success;
	}

};
};
};
};

