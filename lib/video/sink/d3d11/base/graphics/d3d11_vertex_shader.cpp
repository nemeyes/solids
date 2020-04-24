#include "d3d11_vertex_shader.h"
#include "d3d11_exception.h"

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

	RTTI_DEFINITIONS(vertex_shader)

	vertex_shader::vertex_shader(const std::vector<char>& compiledShader, const winrt::com_ptr<ID3D11VertexShader>& vertexShader)
		: _compiled_shader(compiledShader)
		, _shader(vertexShader)
	{
	}

	vertex_shader::vertex_shader(std::vector<char>&& compiledShader, const winrt::com_ptr<ID3D11VertexShader>& vertexShader)
		: _compiled_shader(compiledShader)
		, _shader(vertexShader)
	{
	}

	const std::vector<char>& vertex_shader::compiled_shader(void) const
	{
		return _compiled_shader;
	}

	winrt::com_ptr<ID3D11VertexShader> vertex_shader::shader(void) const
	{
		return _shader;
	}

	winrt::com_ptr<ID3D11InputLayout> vertex_shader::input_layout(void) const
	{
		return _input_layout;
	}

	void vertex_shader::create_input_layout(gsl::not_null<ID3D11Device*> device, const gsl::span<const D3D11_INPUT_ELEMENT_DESC>& inputElementDescriptions, BOOL releaseCompiledShader)
	{
		_input_layout = nullptr;
		throw_if_failed(device->CreateInputLayout(&inputElementDescriptions[0], gsl::narrow_cast<uint32_t>(inputElementDescriptions.size()), &_compiled_shader[0], _compiled_shader.size(), _input_layout.put()), "ID3D11Device::CreateInputLayout() failed.");

		if (releaseCompiledShader)
		{
			_compiled_shader.clear();
			_compiled_shader.shrink_to_fit();
		}
	}

};
};
};
};
};
};