#include "d3d11_drawable_component.h"

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

	RTTI_DEFINITIONS(drawable_component)

	drawable_component::drawable_component(solids::lib::video::sink::d3d11::base::engine& core, std::shared_ptr<solids::lib::video::sink::d3d11::base::camera> cam)
		: solids::lib::video::sink::d3d11::base::component(core)
		, _camera(cam)
	{
	}

	BOOL drawable_component::visible(void) const
	{
		return _visible;
	}

	void drawable_component::set_visible(BOOL visible)
	{
		_visible = visible;
	}

	std::shared_ptr<solids::lib::video::sink::d3d11::base::camera> drawable_component::get_camera(void)
	{
		return _camera;
	}

	void drawable_component::set_camera(const std::shared_ptr<solids::lib::video::sink::d3d11::base::camera>& cam)
	{
		_camera = cam;
	}

	void drawable_component::draw(const solids::lib::video::sink::d3d11::base::time&)
	{

	}

};
};
};
};
};
};