#include "d3d11_content_manager.h"
#include "d3d11_content_type_reader_manager.h"

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

	content_manager::content_manager(solids::lib::video::sink::d3d11::base::engine& core, const std::wstring& root_dir)
		: _engine(core)
		, _root_dir(root_dir)
	{
	}

	void content_manager::add(const std::wstring& name, const std::shared_ptr<solids::lib::video::sink::d3d11::base::rtti>& asset)
	{
		_loaded_assets[name] = asset;
	}

	void content_manager::remove(const std::wstring& name)
	{
		_loaded_assets.erase(name);
	}

	void content_manager::clear(void)
	{
		_loaded_assets.clear();
	}

	std::shared_ptr<solids::lib::video::sink::d3d11::base::rtti> content_manager::read(const int64_t target_type_id, const std::wstring& name)
	{
		const auto& ctr = solids::lib::video::sink::d3d11::base::content_type_reader_manager::content_type_readers();
		auto it = ctr.find(target_type_id);
		if (it == ctr.end())
		{
			throw solids::lib::video::sink::d3d11::base::exception("content type reader not registered.");
		}

		auto& reader = it->second;
		return reader->read(name);
	}

};
};
};
};
};
};



