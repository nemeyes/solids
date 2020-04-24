#include "d3d11_string_helper.h"

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

	BOOL string_helper::ends_with(const std::string& value, const std::string& ending)
	{
		if (ending.size() > value.size())
			return FALSE;
		return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
	}

	BOOL string_helper::ends_with(const std::wstring& value, const std::wstring& ending)
	{
		if (ending.size() > value.size())
			return FALSE;
		return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
	}

};
};
};
};
};
};