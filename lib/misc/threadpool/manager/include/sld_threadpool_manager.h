#pragma once

#if defined(EXPORT_SLD_THREADPOOLMANAGER_LIB)
#  define EXP_SLD_THREADPOOLMANAGER_CALSS __declspec(dllexport)
#else
#  define EXP_SLD_THREADPOOLMANAGER_CALSS __declspec(dllimport)
#endif

#include <sld.h>

namespace solids
{
	namespace lib
	{
		namespace misc
		{
			namespace threadpool
			{
				class worker;
				class EXP_SLD_THREADPOOLMANAGER_CALSS manager
					: public solids::lib::base
				{
					class core;
				public:
					manager(void);
					virtual ~manager(void);

					int32_t initialize(int32_t nthread = 0);
					int32_t release(void);


					int32_t run(int32_t ID, const uint8_t* bytes, int32_t nbytes, void* user);
					int32_t add_worker(std::shared_ptr<solids::lib::misc::threadpool::worker> worker);
					int32_t run_worker(PTP_WORK work);

				private:
					solids::lib::misc::threadpool::manager::core * _core;

				};
			};
		};
	};
};
