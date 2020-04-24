#pragma once

#include <sld.h>
#include "d3d11_rtti.h"
#include "d3d11_exception.h"
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
						class engine;
						class content_manager final
						{
						public:
							content_manager(solids::lib::video::sink::d3d11::base::engine& core, const std::wstring& root_dir = _def_root_dir);
							content_manager(solids::lib::video::sink::d3d11::base::content_manager&&) = default;
							content_manager(const solids::lib::video::sink::d3d11::base::content_manager&) = delete;
							content_manager& operator=(const solids::lib::video::sink::d3d11::base::content_manager&) = delete;
							content_manager& operator=(solids::lib::video::sink::d3d11::base::content_manager&&) = default;
							~content_manager(void) = default;

							const std::map<std::wstring, std::shared_ptr<solids::lib::video::sink::d3d11::base::rtti>>& loaded_assets(void) const;
							const std::wstring& root_directory(void) const;
							void set_root_directory(const std::wstring& root_dir);

							template<typename T>
							std::shared_ptr<T>	load(const std::wstring& name, BOOL reload = FALSE, std::function<std::shared_ptr<T>(std::wstring&)> reader = nullptr);
							void				add(const std::wstring& name, const std::shared_ptr<solids::lib::video::sink::d3d11::base::rtti>& asset);
							void				remove(const std::wstring& name);
							void				clear(void);

						private:
							std::shared_ptr<solids::lib::video::sink::d3d11::base::rtti> read(const std::int64_t target_type_id, const std::wstring& name);

						private:
							static const std::wstring _def_root_dir;
							solids::lib::video::sink::d3d11::base::engine& _engine;
							std::map<std::wstring, std::shared_ptr<solids::lib::video::sink::d3d11::base::rtti>> _loaded_assets;
							std::wstring _root_dir;
						};
					};
				};
			};
		};
	};
};

#include "d3d11_content_manager.inl"