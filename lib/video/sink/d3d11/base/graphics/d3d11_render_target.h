#pragma once

#include <sld.h>
#include "d3d11_rtti.h"
#include <stack>
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
						class render_target
							: public solids::lib::video::sink::d3d11::base::rtti
						{
							RTTI_DECLARATIONS(render_target, rtti)

						public:
							render_target(void) = default;
							render_target(const solids::lib::video::sink::d3d11::base::render_target&) = delete;
							render_target(solids::lib::video::sink::d3d11::base::render_target&&) = default;
							render_target& operator=(const solids::lib::video::sink::d3d11::base::render_target&) = delete;
							render_target& operator=(solids::lib::video::sink::d3d11::base::render_target&) = delete;
							virtual ~render_target(void) = default;

							virtual void begin(void) = 0;
							virtual void end(void) = 0;

						protected:
							typedef struct _render_target_data_t
							{
								std::vector<ID3D11RenderTargetView*>	render_target_views;
								gsl::not_null<ID3D11DepthStencilView*>	depth_stencil_view;
								D3D11_VIEWPORT							view_port;

								_render_target_data_t(const gsl::span<ID3D11RenderTargetView*>& rtv, gsl::not_null<ID3D11DepthStencilView*> dsv, const D3D11_VIEWPORT& vp)
									: render_target_views(rtv.begin(), rtv.end())
									, depth_stencil_view(dsv)
									, view_port(vp)
								{

								}
								uint32_t view_count(void) const
								{
									return gsl::narrow_cast<uint32_t>(render_target_views.size());
								}

							} render_target_data_t;

							void begin(gsl::not_null<ID3D11DeviceContext*> dc, const gsl::span<ID3D11RenderTargetView*>& rtv, gsl::not_null<ID3D11DepthStencilView*> dsv, const D3D11_VIEWPORT& vp);
							void end(gsl::not_null<ID3D11DeviceContext*> dc);
							void rebind_current_render_targets(gsl::not_null<ID3D11DeviceContext*> dc);

						private:
							static std::stack<solids::lib::video::sink::d3d11::base::render_target::render_target_data_t> _render_target_stack;
						};
					};
				};
			};
		};
	};
};