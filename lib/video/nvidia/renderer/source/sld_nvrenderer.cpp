#include "sld_nvrenderer.h"
#include "nvrenderer.h"

namespace solids
{
namespace lib
{
namespace video
{
namespace nvidia
{

	renderer::renderer(void)
	{
		_core = new solids::lib::video::nvidia::renderer::core(this);
	}

	renderer::~renderer(void)
	{
		if (_core)
			delete _core;
		_core = NULL;
	}

	BOOL renderer::is_initialized(void)
	{
		return _core->is_initialized();
	}

	int32_t	renderer::initialize(solids::lib::video::nvidia::renderer::context_t* ctx)
	{
		return _core->initialize(ctx);
	}

	int32_t	renderer::release(void)
	{
		return _core->release();
	}

	int32_t	renderer::render(uint8_t * deviceptr, int32_t pitch)
	{
		return _core->render(deviceptr, pitch);
	}


};
};
};
};