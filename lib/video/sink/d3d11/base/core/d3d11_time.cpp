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

	const std::chrono::high_resolution_clock::time_point& time::current_time(void) const
	{
		return _current_time;
	}

	void time::set_current_time(const std::chrono::high_resolution_clock::time_point& curtime)
	{
		_current_time = curtime;
	}

	const std::chrono::milliseconds& time::total_time(void) const
	{
		return _total_time;
	}

	void time::set_total_time(const std::chrono::milliseconds& time)
	{
		_total_time = time;
	}

	const std::chrono::milliseconds& time::elapsed_time(void) const
	{
		return _elapsed_time;
	}

	void time::set_elapsed_time(const std::chrono::milliseconds& time)
	{
		_elapsed_time = time;
	}

	std::chrono::duration<float> time::total_time_in_seconds(void) const
	{
		return std::chrono::duration_cast<std::chrono::duration<float>>(_total_time);
	}

	std::chrono::duration<float> time::elapsed_time_in_seconds(void) const
	{
		return std::chrono::duration_cast<std::chrono::duration<float>>(_elapsed_time);
	}

};
};
};
};
};
};
