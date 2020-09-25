#pragma once

#include "sld_threadpool_manager.h"

namespace solids
{
	namespace lib
	{
		namespace misc
		{
			namespace threadpool
			{
				class manager::core
				{
				public:
					core(solids::lib::misc::threadpool::manager * front);
					virtual ~core(void);

					int32_t initialize(int32_t nthread);
					int32_t release(void);

					int32_t run(int32_t ID, const uint8_t* bytes, int32_t nbytes, void* user);
					int32_t add_worker(std::shared_ptr<solids::lib::misc::threadpool::worker> worker);
					int32_t run_worker(PTP_WORK work);

				private:
					solids::lib::misc::threadpool::manager * _front;
					int32_t					_threads;
					TP_CALLBACK_ENVIRON		_cb_env;
					PTP_POOL				_pool;
					PTP_CLEANUP_GROUP		_cleanup_group;
					CRITICAL_SECTION		_lock;

					std::map<int32_t, std::shared_ptr<solids::lib::misc::threadpool::worker>> _workers;
				};
			};
		};
	};
};
