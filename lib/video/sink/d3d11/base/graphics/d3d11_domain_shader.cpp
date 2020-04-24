#include "d3d11_domain_shader.h"

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

	RTTI_DEFINITIONS(domain_shader)

	domain_shader::domain_shader(const winrt::com_ptr<ID3D11DomainShader>& hullShader)
		: _shader(hullShader)
	{
	}

	winrt::com_ptr<ID3D11DomainShader> domain_shader::shader(void) const
	{
		return _shader;
	}

};
};
};
};
};
};