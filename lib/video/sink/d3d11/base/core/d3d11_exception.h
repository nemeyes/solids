#pragma once

#include <sld.h>
#include <exception>

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
						class exception
							: public std::exception
						{
						public:
							exception(const char* const message, HRESULT hr = S_OK);

							HRESULT hr(void) const;
							std::wstring whatw(void) const;

						private:
							HRESULT _hr;
						};

						inline void throw_if_failed(HRESULT hr, const char* const message = "")
						{
							if (FAILED(hr))
							{
								throw solids::lib::video::sink::d3d11::base::exception(message, hr);
							}
						}
					};
				};
			};
		};
	};
};