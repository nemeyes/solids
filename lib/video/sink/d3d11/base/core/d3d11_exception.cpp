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

	exception::exception(const char* const message, HRESULT hr)
		: std::exception(message)
		, _hr(hr)
	{

	}

	HRESULT exception::hr(void) const
	{
		return _hr;
	}

	std::wstring exception::whatw(void) const
	{
		std::wstringstream whatw;
		whatw << what();
		return whatw.str();
	}

};
};
};
};
};
};
