#include "d3d11_texture_cube.h"

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

	RTTI_DEFINITIONS(texture_cube)

	texture_cube::texture_cube(const winrt::com_ptr<ID3D11ShaderResourceView>& shaderResourceView)
		: texture(shaderResourceView)
	{
	}

};
};
};
};
};
};