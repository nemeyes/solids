#include "d3d11_pixel_shader.h"

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

	RTTI_DEFINITIONS(pixel_shader)

	pixel_shader::pixel_shader(const winrt::com_ptr<ID3D11PixelShader>& vertexShader)
		: _shader(vertexShader)
	{
	}

	winrt::com_ptr<ID3D11PixelShader> pixel_shader::shader(void) const
	{
		return _shader;
	}

};
};
};
};
};
};