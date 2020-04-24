#include "d3d11_component.h"

namespace solids
{
namespace lib
{
namespace video
{
namespace sink
{
namespace d3d11
{
namespace base
{

	RTTI_DEFINITIONS(component)

	component::component(solids::lib::video::sink::d3d11::base::engine& core)
		: _core(&core)
	{

	}

	solids::lib::video::sink::d3d11::base::engine* component::get_engine(void)
	{
		return _core;
	}

	void component::set_engine(solids::lib::video::sink::d3d11::base::engine& core)
	{
		_core = gsl::not_null<solids::lib::video::sink::d3d11::base::engine*>(&core);
	}

	BOOL component::enabled(void) const
	{
		return _enabled;
	}

	void component::set_enabled(BOOL enabled)
	{
		_enabled = enabled;
	}

	void component::initialize(void)
	{
	}

	void component::release(void)
	{
	}

	void component::update(const solids::lib::video::sink::d3d11::base::time&)
	{
	}
	
};
};
};
};
};
};
