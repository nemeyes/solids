#pragma once

#include <sld.h>

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
						class point final
						{
						public:
							std::int32_t x;
							std::int32_t y;

							explicit point(const std::int32_t x = 0, const std::int32_t y = 0);
							point(const solids::lib::video::sink::d3d11::base::point&) = default;
							point(solids::lib::video::sink::d3d11::base::point&&) = default;
							point& operator=(const solids::lib::video::sink::d3d11::base::point&) = default;
							point& operator=(solids::lib::video::sink::d3d11::base::point&&) = default;
							~point(void) = default;

							BOOL operator<(const solids::lib::video::sink::d3d11::base::point& rhs) const;

							static const solids::lib::video::sink::d3d11::base::point zero;
						};
					};
				};
			};
		};
	};
};

#include "d3d11_point.inl"