#include "d3d11_hull_shader.h"

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

	RTTI_DEFINITIONS(hull_shader)

	hull_shader::hull_shader(const winrt::com_ptr<ID3D11HullShader>& hullShader)
		: _shader(hullShader)
	{
	}

	winrt::com_ptr<ID3D11HullShader> hull_shader::shader(void) const
	{
		return _shader;
	}

};
};
};
};
};
};