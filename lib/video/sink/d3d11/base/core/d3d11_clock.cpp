#include "d3d11_clock.h"
#include "d3d11_time.h"

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

	clock::clock(void)
	{
		reset();
	}

	const std::chrono::high_resolution_clock::time_point& clock::begin_time(void) const
	{
		return _begin_time;
	}

	const std::chrono::high_resolution_clock::time_point& clock::current_time(void) const
	{
		return _current_time;
	}

	const std::chrono::high_resolution_clock::time_point& clock::end_time(void) const
	{
		return _end_time;
	}

	void clock::reset(void)
	{
		_begin_time = std::chrono::high_resolution_clock::now();
		_current_time = _begin_time;
		_end_time = _current_time;
	}

	void clock::update(solids::lib::video::sink::d3d11::base::time& tm)
	{
		_current_time = std::chrono::high_resolution_clock::now();
		tm.set_current_time(_current_time);
		tm.set_total_time(std::chrono::duration_cast<std::chrono::milliseconds>(_current_time - _begin_time));
		tm.set_elapsed_time(std::chrono::duration_cast<std::chrono::milliseconds>(_current_time - _end_time));
		_end_time = _current_time;
	}

};
};
};
};
};
};
