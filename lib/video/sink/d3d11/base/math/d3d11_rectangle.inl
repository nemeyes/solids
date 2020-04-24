#include "d3d11_rectangle.h"

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

	inline std::int32_t rectangle::left(void) const
	{
		return x;
	}

	inline std::int32_t rectangle::right(void) const
	{
		return x + width;
	}

	inline std::int32_t rectangle::top(void) const
	{
		return y;
	}

	inline std::int32_t rectangle::bottom(void) const
	{
		return y + height;
	}

	inline solids::lib::video::sink::d3d11::base::point rectangle::size(void) const
	{
		return solids::lib::video::sink::d3d11::base::point(width, height);
	}

	inline void rectangle::set_size(const solids::lib::video::sink::d3d11::base::point& size)
	{
		width = size.x;
		height = size.y;
	}

	inline solids::lib::video::sink::d3d11::base::point rectangle::center(void) const
	{
		return solids::lib::video::sink::d3d11::base::point(x + (width / 2), y + (height / 2));
	}

	inline solids::lib::video::sink::d3d11::base::point rectangle::location(void) const
	{
		return solids::lib::video::sink::d3d11::base::point(x, y);
	}

	inline void rectangle::SetLocation(solids::lib::video::sink::d3d11::base::point& location)
	{
		x = location.x;
		y = location.y;
	}

};
};
};
};
};
};