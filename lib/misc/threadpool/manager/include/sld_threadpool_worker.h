#pragma once

#if defined(EXPORT_SLD_THREADPOOLMANAGER_LIB)
#define EXP_SLD_THREADPOOLMANAGER_CALSS __declspec(dllexport)
#else
#define EXP_SLD_THREADPOOLMANAGER_CALSS __declspec(dllimport)
#endif

#include <sld.h>

typedef void (*WORKERFUNC)(PTP_CALLBACK_INSTANCE instance, PVOID parameter, PTP_WORK work);

namespace solids
{
	namespace lib
	{
		namespace misc
		{
			namespace threadpool
			{
				
				class manager;
				class EXP_SLD_THREADPOOLMANAGER_CALSS worker
				{
					class core;
				public:
					worker(solids::lib::misc::threadpool::manager * mngr, int32_t id);
					virtual ~worker(void);

					int32_t			id(void);
					BOOL			is_running(void);
					WORKERFUNC		worker_cb(void);
					void			set_work(PTP_WORK work);
					void			run(const uint8_t* bytes, int32_t nbytes, void* user);

					virtual void	execute(const uint8_t* bytes, int32_t nbytes, void* user) = 0;

				private:
					solids::lib::misc::threadpool::worker::core * _core;
				};

			};
		};
	};
};