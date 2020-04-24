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

	inline BOOL point::operator<(const solids::lib::video::sink::d3d11::base::point& rhs) const
	{
		return (x == rhs.x ? y < rhs.y : x < rhs.x);
	}

};
};
};
};
};
};