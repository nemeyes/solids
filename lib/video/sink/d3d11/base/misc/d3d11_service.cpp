#include "d3d11_service.h"

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

	void service::add(std::uint64_t type_id, void* service)
	{
		_services.insert(std::pair<uint64_t, void*>(type_id, service));
	}

	void service::remove(std::uint64_t type_id)
	{
		_services.erase(type_id);
	}

	void* service::get(std::uint64_t type_id) const
	{
		std::map<uint64_t, void*>::const_iterator it = _services.find(type_id);
		return (it != _services.end() ? it->second : nullptr);
	}

};
};
};
};
};
};
