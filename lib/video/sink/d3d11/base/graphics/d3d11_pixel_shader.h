#pragma once

#include "d3d11_shader.h"

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
						class pixel_shader final
							: public solids::lib::video::sink::d3d11::base::shader
						{
							RTTI_DECLARATIONS(pixel_shader, shader)

						public:
							pixel_shader(const winrt::com_ptr<ID3D11PixelShader>& pixelShader);
							pixel_shader(const solids::lib::video::sink::d3d11::base::pixel_shader&) = default;
							pixel_shader& operator=(const solids::lib::video::sink::d3d11::base::pixel_shader&) = default;
							pixel_shader(solids::lib::video::sink::d3d11::base::pixel_shader&&) = default;
							pixel_shader& operator=(solids::lib::video::sink::d3d11::base::pixel_shader&&) = default;
							~pixel_shader(void) = default;

							winrt::com_ptr<ID3D11PixelShader> shader(void) const;

						private:
							winrt::com_ptr<ID3D11PixelShader> _shader;
						};
					};
				};
			};
		};
	};
};