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
						class compute_shader final
							: public solids::lib::video::sink::d3d11::base::shader
						{
							RTTI_DECLARATIONS(compute_shader, shader)

						public:
							compute_shader(const winrt::com_ptr<ID3D11ComputeShader>& hullShader);
							compute_shader(const solids::lib::video::sink::d3d11::base::compute_shader&) = default;
							compute_shader& operator=(const solids::lib::video::sink::d3d11::base::compute_shader&) = default;
							compute_shader(solids::lib::video::sink::d3d11::base::compute_shader&&) = default;
							compute_shader& operator=(solids::lib::video::sink::d3d11::base::compute_shader&&) = default;
							~compute_shader(void) = default;

							winrt::com_ptr<ID3D11ComputeShader> shader(void) const;

						private:
							winrt::com_ptr<ID3D11ComputeShader> _shader;
						};
					};
				};
			};
		};
	};
};