#pragma once

#include "d3d11_component.h"
#include <memory>

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
						class camera;
						class drawable_component
							: public solids::lib::video::sink::d3d11::base::component
						{
							RTTI_DECLARATIONS(drawable_component, component)

						public:
							drawable_component(void) = default;
							explicit drawable_component(solids::lib::video::sink::d3d11::base::engine& core, std::shared_ptr<solids::lib::video::sink::d3d11::base::camera> cam = nullptr);
							drawable_component(const drawable_component&) = default;
							drawable_component& operator=(const drawable_component&) = default;
							drawable_component(drawable_component&&) = default;
							drawable_component& operator=(drawable_component&&) = default;
							virtual ~drawable_component(void) = default;

							BOOL visible(void) const;
							void set_visible(BOOL visible);

							std::shared_ptr<solids::lib::video::sink::d3d11::base::camera> get_camera(void);
							void set_camera(const std::shared_ptr<solids::lib::video::sink::d3d11::base::camera>& camera);

							virtual void draw(const solids::lib::video::sink::d3d11::base::time& tm);

						protected:
							BOOL _visible{ TRUE };
							std::shared_ptr<solids::lib::video::sink::d3d11::base::camera> _camera;
						};
					};
				};
			};
		};
	};
};