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
						class domain_shader final
							: public solids::lib::video::sink::d3d11::base::shader
						{
							RTTI_DECLARATIONS(domain_shader, shader)

						public:
							domain_shader(const winrt::com_ptr<ID3D11DomainShader>& hullShader);
							domain_shader(const solids::lib::video::sink::d3d11::base::domain_shader&) = default;
							domain_shader& operator=(const solids::lib::video::sink::d3d11::base::domain_shader&) = default;
							domain_shader(solids::lib::video::sink::d3d11::base::domain_shader&&) = default;
							domain_shader& operator=(solids::lib::video::sink::d3d11::base::domain_shader&&) = default;
							~domain_shader(void) = default;

							winrt::com_ptr<ID3D11DomainShader> shader(void) const;

						private:
							winrt::com_ptr<ID3D11DomainShader> _shader;
						};
					};
				};
			};
		};
	};
};