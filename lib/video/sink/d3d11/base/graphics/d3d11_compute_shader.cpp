#include "d3d11_compute_shader.h"

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

	RTTI_DEFINITIONS(compute_shader)

	compute_shader::compute_shader(const winrt::com_ptr<ID3D11ComputeShader>& hullShader)
		: _shader(hullShader)
	{
	}

	winrt::com_ptr<ID3D11ComputeShader> compute_shader::shader(void) const
	{
		return _shader;
	}

};
};
};
};
};
};