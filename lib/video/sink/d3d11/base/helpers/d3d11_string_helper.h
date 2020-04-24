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
						class string_helper final
						{
						public:
							static BOOL ends_with(const std::string& value, const std::string& ending);
							static BOOL ends_with(const std::wstring& value, const std::wstring& ending);

							string_helper(void) = delete;
							string_helper(const solids::lib::video::sink::d3d11::base::string_helper&) = delete;
							string_helper& operator=(const solids::lib::video::sink::d3d11::base::string_helper&) = delete;
							string_helper(solids::lib::video::sink::d3d11::base::string_helper&&) = delete;
							string_helper& operator=(solids::lib::video::sink::d3d11::base::string_helper&&) = delete;
							~string_helper(void) = delete;
						};
					};
				};
			};
		};
	};
};