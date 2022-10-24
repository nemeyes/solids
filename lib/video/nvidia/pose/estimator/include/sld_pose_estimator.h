#pragma once

#if defined(EXP_SLD_POSE_ESTIMATOR_LIB)
#define EXP_SLD_POSE_ESTIMATOR_CLS	__declspec(dllexport)
#else
#define EXP_SLD_POSE_ESTIMATOR_CLS	__declspec(dllimport)
#endif

#include <sld.h>

namespace solids
{
	namespace lib
	{
		namespace video
		{
			namespace nvidia
			{
				namespace pose
				{

					class EXP_SLD_POSE_ESTIMATOR_CLS estimator
						: public solids::lib::base
					{
						class core;
					public:
						typedef struct _context_t
						{
							int32_t width;
							int32_t height;
							std::string enginePath;
							_context_t(void)
								: width(-1)
								, height(-1)
								, enginePath("")
							{}

							~_context_t(void)
							{}
						} context_t;

						estimator(void);
						virtual ~estimator(void);

						int32_t initialize(solids::lib::video::nvidia::pose::estimator::context_t* ctx);
						int32_t release(void);

						int32_t estimate(uint8_t* input, int32_t inputStride, uint8_t* srcBBox, int32_t bboxSize, uint8_t** output, int32_t& outputStride);

					private:
						solids::lib::video::nvidia::pose::estimator::core* _core;
					};

				};
			};
		};
	};
};

