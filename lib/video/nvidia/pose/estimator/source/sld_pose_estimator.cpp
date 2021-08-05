#include "sld_pose_estimator.h"
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

	estimator::estimator(void)
	{
		_core = new solids::lib::video::nvidia::pose::estimator::core(this);
	}

	estimator::~estimator(void)
	{
		if (_core)
			delete _core;
		_core = NULL;
	}

	int32_t estimator::initialize(solids::lib::video::nvidia::pose::estimator::context_t* ctx)
	{
		return _core->initialize(ctx);
	}

	int32_t estimator::release(void)
	{
		return _core->release();
	}

	int32_t estimator::estimate(uint8_t* input, int32_t inputStride, uint8_t* srcBBox, int32_t bboxSize, uint8_t** output, int32_t& outputStride)
	{
		return _core->estimate(input, inputStride, srcBBox, bboxSize, output, outputStride);
	}

};
};
};
};
};

