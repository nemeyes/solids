#pragma once

#include "d3d11_rtti.h"
#include <gsl/gsl>

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
						class time;
						class component
							: public solids::lib::video::sink::d3d11::base::rtti
						{
							RTTI_DECLARATIONS(component, rtti)

						public:
							component(void) = default;
							explicit component(solids::lib::video::sink::d3d11::base::engine& core);
							component(const component&) = default;
							component& operator=(const component&) = default;
							component(component&&) = default;
							component& operator=(component&&) = default;
							virtual ~component(void) = default;

							solids::lib::video::sink::d3d11::base::engine* get_engine(void);
							void set_engine(solids::lib::video::sink::d3d11::base::engine& core);
							BOOL enabled(void) const;
							void set_enabled(BOOL enabled);

							virtual void initialize(void);
							virtual void release(void);
							virtual void update(const solids::lib::video::sink::d3d11::base::time& tm);

						protected:
							gsl::not_null<solids::lib::video::sink::d3d11::base::engine*> _core;
							BOOL _enabled{ TRUE };
						};
					};
				};
			};
		};
	};
};