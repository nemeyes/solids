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
						class time final
						{
						public:
							const std::chrono::high_resolution_clock::time_point& current_time(void) const;
							void set_current_time(const std::chrono::high_resolution_clock::time_point& curtime);

							const std::chrono::milliseconds& total_time(void) const;
							void set_total_time(const std::chrono::milliseconds& time);

							const std::chrono::milliseconds& elapsed_time(void) const;
							void set_elapsed_time(const std::chrono::milliseconds& time);

							std::chrono::duration<float> total_time_in_seconds(void) const;
							std::chrono::duration<float> elapsed_time_in_seconds(void) const;

						private:
							std::chrono::high_resolution_clock::time_point _current_time;
							std::chrono::milliseconds _total_time{ 0 };
							std::chrono::milliseconds _elapsed_time{ 0 };
						};
					};
				};
			};
		};
	};
};