#include "sld_object_detector.h"
#include "object_detector.h"

namespace solids
{
namespace lib
{
namespace video
{
namespace nvidia
{
namespace object
{

	detector::detector(void)
	{
		_core = new solids::lib::video::nvidia::object::detector::core(this);
	}

	detector::~detector(void)
	{
		if (_core)
			delete _core;
		_core = NULL;
	}

	int32_t detector::initialize(solids::lib::video::nvidia::object::detector::context_t* ctx)
	{
		return _core->initialize(ctx);
	}

	int32_t detector::release(void)
	{
		return _core->release();
	}

	int32_t detector::detect(uint8_t* input, int32_t inputStride, uint8_t** output, int32_t& outputStride)
	{
		return _core->detect(input, inputStride, output, outputStride);
	}

};
};
};
};
};

