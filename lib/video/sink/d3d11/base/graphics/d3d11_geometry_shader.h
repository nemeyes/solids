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
						class geometry_shader final
							: public solids::lib::video::sink::d3d11::base::shader
						{
							RTTI_DECLARATIONS(geometry_shader, shader)
						public:
							geometry_shader(const winrt::com_ptr<ID3D11GeometryShader>& pixelShader);
							geometry_shader(const solids::lib::video::sink::d3d11::base::geometry_shader&) = default;
							geometry_shader& operator=(const solids::lib::video::sink::d3d11::base::geometry_shader&) = default;
							geometry_shader(solids::lib::video::sink::d3d11::base::geometry_shader&&) = default;
							geometry_shader& operator=(solids::lib::video::sink::d3d11::base::geometry_shader&&) = default;
							~geometry_shader() = default;

							winrt::com_ptr<ID3D11GeometryShader> shader(void) const;

						private:
							winrt::com_ptr<ID3D11GeometryShader> _shader;
						};
					};
				};
			};
		};
	};
};
