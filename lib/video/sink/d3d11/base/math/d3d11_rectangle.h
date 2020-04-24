#pragma once

#include <sld.h>
#include "d3d11_point.h"

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
						class rectangle final
						{
						public:
							std::int32_t x;
							std::int32_t y;
							std::int32_t width;
							std::int32_t height;

							rectangle(const solids::lib::video::sink::d3d11::base::point& location, const solids::lib::video::sink::d3d11::base::point& size);
							rectangle(std::int32_t x, std::int32_t y, std::int32_t width, std::int32_t height);
							rectangle(const solids::lib::video::sink::d3d11::base::rectangle&) = default;
							rectangle(solids::lib::video::sink::d3d11::base::rectangle&&) = default;
							rectangle& operator=(const solids::lib::video::sink::d3d11::base::rectangle&) = default;
							rectangle& operator=(solids::lib::video::sink::d3d11::base::rectangle&&) = default;
							~rectangle(void) = default;

							std::int32_t left(void) const;
							std::int32_t right(void) const;
							std::int32_t top(void) const;
							std::int32_t bottom(void) const;

							solids::lib::video::sink::d3d11::base::point size(void) const;
							void	set_size(const solids::lib::video::sink::d3d11::base::point& size);
							solids::lib::video::sink::d3d11::base::point center(void) const;
							solids::lib::video::sink::d3d11::base::point location(void) const;
							void	set_location(solids::lib::video::sink::d3d11::base::point& location);

							BOOL	intersects(const solids::lib::video::sink::d3d11::base::rectangle& other) const;

							static const solids::lib::video::sink::d3d11::base::rectangle empty;
						};
					};
				};
			};
		};
	};
};

#include "d3d11_rectangle.inl"