#include "d3d11_vertex_shader_reader.h"
#include "d3d11_engine.h"
#include "d3d11_exception.h"
#include "d3d11_utility.h"

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
	
	RTTI_DEFINITIONS(vertex_shader_reader)

	vertex_shader_reader::vertex_shader_reader(solids::lib::video::sink::d3d11::base::engine & core)
		: content_type_reader(core, solids::lib::video::sink::d3d11::base::vertex_shader::type_id_class())
	{
	}

	std::shared_ptr<solids::lib::video::sink::d3d11::base::vertex_shader> vertex_shader_reader::_read(const std::wstring& name)
	{
		winrt::com_ptr<ID3D11VertexShader> vertexShader;
		std::vector<char> compiledVertexShader;
		solids::lib::video::sink::d3d11::base::utility::load_binary_file(name, compiledVertexShader);
		solids::lib::video::sink::d3d11::base::throw_if_failed(_engine->d3d11dev()->CreateVertexShader(&compiledVertexShader[0], compiledVertexShader.size(), nullptr, vertexShader.put()), "ID3D11Device::CreatedVertexShader() failed.");

		return std::shared_ptr<solids::lib::video::sink::d3d11::base::vertex_shader>(new solids::lib::video::sink::d3d11::base::vertex_shader(std::move(compiledVertexShader), std::move(vertexShader)));
	}

};
};
};
};
};
};
