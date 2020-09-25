#include "pose_estimator.h"

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

	estimator::core::core(solids::lib::video::nvidia::pose::estimator* front)
		: _front(front)
		, _ctx(NULL)
	{
	}

	estimator::core::~core(void)
	{
	}

	int32_t estimator::core::initialize(solids::lib::video::nvidia::pose::estimator::context_t* ctx)
	{
		_ctx = ctx;
		return solids::lib::video::nvidia::pose::estimator::err_code_t::success;
	}

	int32_t estimator::core::release(void)
	{
		return solids::lib::video::nvidia::pose::estimator::err_code_t::success;
	}

	int32_t estimator::core::estimate(uint8_t* input, int32_t inputStride, uint8_t** output, int32_t& outputStride)
	{
		*output = input;
		outputStride = inputStride;
		return solids::lib::video::nvidia::pose::estimator::err_code_t::success;
	}

};
};
};
};
};