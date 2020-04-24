#include "d3d11_render_target.h"

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
	RTTI_DEFINITIONS(render_target)

	std::stack<solids::lib::video::sink::d3d11::base::render_target::render_target_data_t> render_target::_render_target_stack;

	void render_target::begin(gsl::not_null<ID3D11DeviceContext*> dc, const gsl::span<ID3D11RenderTargetView*>& rtv, gsl::not_null<ID3D11DepthStencilView*> dsv, const D3D11_VIEWPORT& vp)
	{
		_render_target_stack.emplace(rtv, dsv, vp);
		dc->OMSetRenderTargets(gsl::narrow_cast<uint32_t>(rtv.size()), &rtv[0], dsv);
		dc->RSSetViewports(1, &vp);
	}

	void render_target::end(gsl::not_null<ID3D11DeviceContext*> dc)
	{
		_render_target_stack.pop();
		solids::lib::video::sink::d3d11::base::render_target::render_target_data_t rtd = _render_target_stack.top();
		dc->OMSetRenderTargets(rtd.view_count(), rtd.render_target_views.data(), rtd.depth_stencil_view);
		dc->RSSetViewports(1, &rtd.view_port);
	}

	void render_target::rebind_current_render_targets(gsl::not_null<ID3D11DeviceContext*> dc)
	{
		solids::lib::video::sink::d3d11::base::render_target::render_target_data_t rtd = _render_target_stack.top();
		dc->OMSetRenderTargets(rtd.view_count(), rtd.render_target_views.data(), rtd.depth_stencil_view);
		dc->RSSetViewports(1, &rtd.view_port);
	}
};
};
};
};
};
};