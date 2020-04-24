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
	template<typename T>
	inline content_type_reader<T>::content_type_reader(solids::lib::video::sink::d3d11::base::engine& core, const std::uint64_t target_type_id)
		: base_content_type_reader(core, target_type_id)
	{
	}

	template<typename T>
	inline std::shared_ptr<solids::lib::video::sink::d3d11::base::rtti> content_type_reader<T>::read(const std::wstring& name)
	{
		return _read(name);
	}
};
};
};
};
};
};


