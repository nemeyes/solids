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

	const rectangle rectangle::empty{ 0, 0, 0, 0 };

	rectangle::rectangle(const solids::lib::video::sink::d3d11::base::point& location, const solids::lib::video::sink::d3d11::base::point& size)
		: x(location.x)
		, y(location.y)
		, width(size.x)
		, height(size.y)
	{
	}

	rectangle::rectangle(int32_t x, int32_t y, int32_t width, int32_t height)
		: x(x)
		, y(y)
		, width(width)
		, height(height)
	{
	}

	BOOL rectangle::intersects(const solids::lib::video::sink::d3d11::base::rectangle& other) const
	{
		return other.left() < right() && left() < other.right() && other.top() < bottom() && top() < other.bottom();
	}

};
};
};
};
};
};