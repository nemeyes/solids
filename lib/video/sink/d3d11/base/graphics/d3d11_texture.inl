#pragma once

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

	inline winrt::com_ptr<ID3D11ShaderResourceView> texture::shader_resource_view(void) const
	{
		return _shader_resource_view;
	}

};
};
};
};
};
};