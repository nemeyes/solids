#pragma once

#include <exception>
#include <chrono>

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
						class time;
						class clock final
						{
						public: 
							clock(void) = default;
							clock(const solids::lib::video::sink::d3d11::base::clock&) = default;
							clock& operator=(const solids::lib::video::sink::d3d11::base::clock&) = default;
							clock(solids::lib::video::sink::d3d11::base::clock&&) = default;
							clock& operator=(solids::lib::video::sink::d3d11::base::clock&&) = default;
							~clock(void) = default;

							const std::chrono::high_resolution_clock::time_point& begin_time(void) const;
							const std::chrono::high_resolution_clock::time_point& current_time(void) const;
							const std::chrono::high_resolution_clock::time_point& end_time(void) const;

							void reset(void);
							void update(solids::lib::video::sink::d3d11::base::time& tm);

						private:
							std::chrono::high_resolution_clock::time_point _begin_time;
							std::chrono::high_resolution_clock::time_point _current_time;
							std::chrono::high_resolution_clock::time_point _end_time;
						};
					};
				};
			};
		};
	};
};