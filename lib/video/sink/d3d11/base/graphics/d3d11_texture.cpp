#include "d3d11_texture.h"

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

	RTTI_DEFINITIONS(texture)

	texture::texture(const winrt::com_ptr<ID3D11ShaderResourceView>& shaderResourceView)
		: _shader_resource_view(shaderResourceView)
	{
	}

};
};
};
};
};
};