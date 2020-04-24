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
						class hull_shader final
							: public solids::lib::video::sink::d3d11::base::shader
						{
							RTTI_DECLARATIONS(hull_shader, shader)

						public:
							hull_shader(const winrt::com_ptr<ID3D11HullShader>& hullShader);
							hull_shader(const solids::lib::video::sink::d3d11::base::hull_shader&) = default;
							hull_shader& operator=(const solids::lib::video::sink::d3d11::base::hull_shader&) = default;
							hull_shader(solids::lib::video::sink::d3d11::base::hull_shader&&) = default;
							hull_shader& operator=(solids::lib::video::sink::d3d11::base::hull_shader&&) = default;
							~hull_shader(void) = default;

							winrt::com_ptr<ID3D11HullShader> shader(void) const;

						private:
							winrt::com_ptr<ID3D11HullShader> _shader;
						};
					};
				};
			};
		};
	};
};