#include "d3d11_shader.h"
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

	RTTI_DEFINITIONS(shader)

	winrt::com_ptr<ID3D11ClassLinkage> shader::create_class_linkage(gsl::not_null<ID3D11Device*> device)
	{
		winrt::com_ptr<ID3D11ClassLinkage> clslinkage;
		throw_if_failed(device->CreateClassLinkage(clslinkage.put()), "ID3D11Device::CreateClassLinkage() failed.");
		return clslinkage;
	}

};
};
};
};
};
};