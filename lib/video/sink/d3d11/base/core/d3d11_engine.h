#pragma once

#include <sld.h>
#include <d2d1_3.h>
#include <dwrite_3.h>
#include <wincodec.h>
#include <DirectXMath.h>
#include <DirectXColors.h>

#include "d3d11_clock.h"
#include "d3d11_time.h"
#include "d3d11_service.h"
#include "d3d11_render_target.h"
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
						class component;
						class base_device_notify
						{
						public:
							virtual ~base_device_notify(void) {};
							virtual void on_device_lost(void) = 0;
							virtual void on_device_restored(void) = 0;

						protected:
							base_device_notify(void) {};
						};

						class engine
							: public solids::lib::video::sink::d3d11::base::render_target
						{
							RTTI_DECLARATIONS(engine, render_target)

						public:
							engine(std::function<void* ()> getWindowCB, std::function<void(SIZE&)> getRenderTargetSizeCB);
							engine(const engine&) = delete;
							engine& operator=(const engine&) = delete;
							engine(engine&&) = delete;
							engine& operator=(engine&&) = delete;
							virtual ~engine(void) = default;

							gsl::not_null<ID3D11Device3*>			d3d11dev(void) const;
							gsl::not_null<ID3D11DeviceContext3*>	d3d11ctx(void) const;
							gsl::not_null<IDXGISwapChain1*>			swapchain(void) const;
							gsl::not_null<ID3D11RenderTargetView*>	render_target_view(void) const;
							gsl::not_null<ID3D11DepthStencilView*>	depth_stencil_view(void) const;
							SIZE									render_target_size(void) const;
							float									aspect_ratio(void) const;
							BOOL									is_full_screen(void) const;
							const D3D11_TEXTURE2D_DESC&				backbuffer_desc(void) const;
							const D3D11_VIEWPORT&					viewport(void) const;
							std::uint32_t							multisampling_count(void) const;
							std::uint32_t							multisampling_quality_levels(void) const;

							const std::vector<std::shared_ptr<solids::lib::video::sink::d3d11::base::component>>&	components(void) const;
							const solids::lib::video::sink::d3d11::base::service &									services(void) const;

							virtual void			initialize(void);
							virtual void			run(void);
							virtual void			release(void);

							virtual void			update(const solids::lib::video::sink::d3d11::base::time& tm);
							virtual void			draw(const solids::lib::video::sink::d3d11::base::time& tm);

							void					update_render_target_size(void);
							void					register_device_notify(solids::lib::video::sink::d3d11::base::base_device_notify * deviceNotify);

							std::function<void* ()> get_window_cb(void) const;

							solids::lib::video::sink::d3d11::base::content_manager&	content(void);

						protected:
							virtual void handle_device_lost(void);
							virtual void begin(void) override;
							virtual void end(void) override;

						protected:
							virtual void create_device_independent_resources(void);
							virtual void create_device_resources(void);
							virtual void create_window_size_dependent_resources(void);

							inline static const D3D_FEATURE_LEVEL	_default_feature_level{ D3D_FEATURE_LEVEL_9_1 };
							inline static const std::uint32_t		_default_fps{ 60 };
							inline static const std::uint32_t		_default_multisampling_count{ 4 };
							inline static const std::uint32_t		_default_buffer_count{ 2 };

							winrt::com_ptr<ID3D11Device3>			_d3d11_dev;
							winrt::com_ptr<ID3D11DeviceContext3>	_d3d11_ctx;
							winrt::com_ptr<IDXGISwapChain1>			_swapchain;
							D3D_FEATURE_LEVEL						_feature_level = _default_feature_level;

							D3D11_TEXTURE2D_DESC					_backbuffer_desc;
							winrt::com_ptr<ID3D11RenderTargetView>	_render_target_view;
							winrt::com_ptr<ID3D11DepthStencilView>	_depth_stencil_view;
							D3D11_VIEWPORT							_viewport;

							std::uint32_t							_fps{ _default_fps };
							BOOL									_bfullscreen{ FALSE };
							std::uint32_t							_multisampling_count{ _default_multisampling_count };
							std::uint32_t							_multisampling_quality_levels{ 0 };

							std::function<void* ()>					_fn_get_window;
							std::function<void(SIZE&)>				_fn_get_render_target_size;
							SIZE									_render_target_size;
							base_device_notify*						_device_notify;

							solids::lib::video::sink::d3d11::base::clock	_clock;
							solids::lib::video::sink::d3d11::base::time		_time;
							std::vector<std::shared_ptr<solids::lib::video::sink::d3d11::base::component>>	_components;
							solids::lib::video::sink::d3d11::base::service									_services;
							solids::lib::video::sink::d3d11::base::content_manager							_content_manager;
						};
					};
				};
			};
		};
	};
};

#include "d3d11_engine.inl"