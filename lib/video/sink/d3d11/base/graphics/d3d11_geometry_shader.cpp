#include "d3d11_geometry_shader.h"

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

	RTTI_DEFINITIONS(geometry_shader)

	geometry_shader::geometry_shader(const winrt::com_ptr<ID3D11GeometryShader>& vertexShader) 
		: _shader(vertexShader)
	{
	}

	winrt::com_ptr<ID3D11GeometryShader> geometry_shader::shader(void) const
	{
		return _shader;
	}

};
};
};
};
};
};