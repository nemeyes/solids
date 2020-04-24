#pragma once

#include <winrt\Windows.Foundation.h>
#include <sld.h>
#include <gsl/gsl>
#include "d3d11_rtti.h"

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
						class shader
							: public solids::lib::video::sink::d3d11::base::rtti
						{
							RTTI_DECLARATIONS(shader, rtti)

						public:
							shader(const solids::lib::video::sink::d3d11::base::shader&) = default;
							shader& operator=(const solids::lib::video::sink::d3d11::base::shader&) = default;
							shader(solids::lib::video::sink::d3d11::base::shader&&) = default;
							shader& operator=(solids::lib::video::sink::d3d11::base::shader&&) = default;
							virtual ~shader(void) = default;

							static winrt::com_ptr<ID3D11ClassLinkage> create_class_linkage(gsl::not_null<ID3D11Device*> device);

						protected:
							shader(void) = default;
						};
					};
				};
			};
		};
	};
};