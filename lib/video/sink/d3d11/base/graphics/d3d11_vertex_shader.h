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
						class vertex_shader final
							: public solids::lib::video::sink::d3d11::base::shader
						{
							RTTI_DECLARATIONS(vertex_shader, shader)

						public:
							vertex_shader(const solids::lib::video::sink::d3d11::base::vertex_shader&) = default;
							vertex_shader& operator=(const solids::lib::video::sink::d3d11::base::vertex_shader&) = default;
							vertex_shader(solids::lib::video::sink::d3d11::base::vertex_shader&&) = default;
							vertex_shader& operator=(solids::lib::video::sink::d3d11::base::vertex_shader&&) = default;
							~vertex_shader(void) = default;

							const std::vector<char>&			compiled_shader(void) const;
							winrt::com_ptr<ID3D11VertexShader>	shader(void) const;
							winrt::com_ptr<ID3D11InputLayout>	input_layout(void) const;

							template <typename T>
							void create_input_layout(gsl::not_null<ID3D11Device*> device, BOOL releaseCompiledShader = FALSE)
							{
								create_input_layout(device, T::InputElements, releaseCompiledShader);
							}

							void create_input_layout(gsl::not_null<ID3D11Device*> device, const gsl::span<const D3D11_INPUT_ELEMENT_DESC>& inputElementDescriptions, BOOL releaseCompiledShader = FALSE);

						private:
							friend class vertex_shader_reader;
							vertex_shader(const std::vector<char>& compiledShader, const winrt::com_ptr<ID3D11VertexShader>& vertexShader);
							vertex_shader(std::vector<char>&& compiledShader, const winrt::com_ptr<ID3D11VertexShader>& vertexShader);

							std::vector<char>					_compiled_shader;
							winrt::com_ptr<ID3D11VertexShader>	_shader;
							winrt::com_ptr<ID3D11InputLayout>	_input_layout;
						};
					};
				};
			};
		};
	};
};