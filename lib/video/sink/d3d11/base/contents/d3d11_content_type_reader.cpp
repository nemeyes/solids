#include "d3d11_content_type_reader.h"

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
	RTTI_DEFINITIONS(base_content_type_reader)

	uint64_t base_content_type_reader::target_type_id(void) const
	{
		return _target_type_id;
	}

	base_content_type_reader::base_content_type_reader(solids::lib::video::sink::d3d11::base::engine& core, const std::uint64_t target_type_id)
		: _engine(&core)
		, _target_type_id(target_type_id)
	{
	}
};
};
};
};
};
};
