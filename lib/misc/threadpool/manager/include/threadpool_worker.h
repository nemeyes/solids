#pragma once

#include "sld_threadpool_worker.h"

namespace solids
{
	namespace lib
	{
		namespace misc
		{
			namespace threadpool
			{
				class worker::core
				{
				public:
					class tp_worker
					{
					public:
						tp_worker(void)
							: handler(NULL)
							, bytes(NULL)
							, nbytes(0)
							, user(NULL)
						{
						}

						~tp_worker(void)
						{
						}

					public:
						solids::lib::misc::threadpool::worker::core* handler;
						uint8_t* bytes;
						int32_t nbytes;
						void* user;
					};

					class tp_worker_scopped_lock
					{
					public:
						tp_worker_scopped_lock(SRWLOCK* srwl)
							: _srwl(srwl)
						{
							::AcquireSRWLockExclusive(_srwl);
						}
						~tp_worker_scopped_lock(void)
						{
							::ReleaseSRWLockExclusive(_srwl);
						}
					private:
						SRWLOCK* _srwl;
					};

					core(solids::lib::misc::threadpool::worker * front, solids::lib::misc::threadpool::manager * mngr, int32_t id);
					~core(void);

					int32_t		id(void);
					BOOL		is_running(void);
					WORKERFUNC	worker_cb(void);
					void		set_work(PTP_WORK work);
					void		run(const uint8_t* bytes, int32_t nbytes, void* user);

					void		_execute(const uint8_t* bytes, int32_t nbytes, void* user);
					void		__execute(void);
				private:
					static void __stdcall tp_worker_cb(PTP_CALLBACK_INSTANCE instance, PVOID parameter, PTP_WORK work);


				private:
					solids::lib::misc::threadpool::worker * _front;
					solids::lib::misc::threadpool::manager * _manager;
					int32_t		_id;
					PTP_WORK	_work;
					SRWLOCK		_srwl;
					std::deque<std::shared_ptr<solids::lib::misc::threadpool::worker::core::tp_worker>> _tp_ready_workers;
					void* _user;
				};

			};
		};
	};
};

