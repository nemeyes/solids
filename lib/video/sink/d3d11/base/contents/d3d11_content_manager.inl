#pragma once
#include "d3d11_content_manager.h"

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

	inline const std::map<std::wstring, std::shared_ptr<solids::lib::video::sink::d3d11::base::rtti>>& content_manager::loaded_assets(void) const
	{
		return _loaded_assets;
	}

	inline const std::wstring& content_manager::root_directory(void) const
	{
		return _root_dir;
	}

	inline void content_manager::set_root_directory(const std::wstring& root_dir)
	{
		_root_dir = root_dir + (solids::lib::video::sink::d3d11::base::string_helper::ends_with(root_dir, L"\\") ? std::wstring() : L"\\");
	}

	template<typename T>
	inline std::shared_ptr<T> content_manager::load(const std::wstring& name, BOOL reload, std::function<std::shared_ptr<T>(std::wstring&)> reader)
	{
		if (reload == FALSE)
		{
			auto it = _loaded_assets.find(name);
			if (it != _loaded_assets.end())
			{
				return std::static_pointer_cast<T>(it->second);
			}
		}

		uint64_t target_type_id = T::type_id_class();
		auto path = _root_dir + name;
		auto asset = (reader != nullptr ? reader(path) : read(target_type_id, path));
		_loaded_assets[name] = asset;

		return std::static_pointer_cast<T>(asset);
	}

};
};
};
};
};
};