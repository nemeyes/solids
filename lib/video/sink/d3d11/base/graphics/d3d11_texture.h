#pragma once

#include <winrt\Windows.Foundation.h>
#include <d3d11.h>
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
						class texture
							: public solids::lib::video::sink::d3d11::base::rtti
						{
							RTTI_DECLARATIONS(texture, rtti)

						public:
							texture(const solids::lib::video::sink::d3d11::base::texture&) = default;
							texture& operator=(const solids::lib::video::sink::d3d11::base::texture&) = default;
							texture(solids::lib::video::sink::d3d11::base::texture&&) = default;
							texture& operator=(solids::lib::video::sink::d3d11::base::texture&&) = default;
							virtual ~texture(void) = default;

							winrt::com_ptr<ID3D11ShaderResourceView> shader_resource_view(void) const;

						protected:
							texture(const winrt::com_ptr<ID3D11ShaderResourceView>& shaderResourceView);


						protected:
							winrt::com_ptr<ID3D11ShaderResourceView> _shader_resource_view;
						};
					};
				};
			};
		};
	};
};

#include "d3d11_texture.inl"