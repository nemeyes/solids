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
						inline gsl::not_null<ID3D11Device3*> engine::d3d11dev(void) const
						{
							return gsl::not_null<ID3D11Device3*>(_d3d11_dev.get());
						}

						inline gsl::not_null<ID3D11DeviceContext3*> engine::d3d11ctx(void) const
						{
							return gsl::not_null<ID3D11DeviceContext3*>(_d3d11_ctx.get());
						}

						inline gsl::not_null<IDXGISwapChain1*> engine::swapchain(void) const
						{
							return gsl::not_null<IDXGISwapChain1*>(_swapchain.get());
						}

						inline gsl::not_null<ID3D11RenderTargetView*> engine::render_target_view(void) const
						{
							return gsl::not_null<ID3D11RenderTargetView*>(_render_target_view.get());
						}

						inline gsl::not_null<ID3D11DepthStencilView*> engine::depth_stencil_view(void) const
						{
							return gsl::not_null<ID3D11DepthStencilView*>(_depth_stencil_view.get());
						}

						inline SIZE engine::render_target_size(void) const
						{
							return _render_target_size;
						}

						inline float engine::aspect_ratio(void) const
						{
							return static_cast<float>(_render_target_size.cx) / _render_target_size.cy;
						}

						inline BOOL engine::is_full_screen(void) const
						{
							return _bfullscreen;
						}

						inline const D3D11_TEXTURE2D_DESC& engine::backbuffer_desc(void) const
						{
							return _backbuffer_desc;
						}

						inline const D3D11_VIEWPORT& engine::viewport(void) const
						{
							return _viewport;
						}

						inline std::uint32_t engine::multisampling_count(void) const
						{
							return _multisampling_count;
						}

						inline std::uint32_t engine::multisampling_quality_levels(void) const
						{
							return _multisampling_quality_levels;
						}

						inline const std::vector<std::shared_ptr<solids::lib::video::sink::d3d11::base::component>>& engine::components(void) const
						{
							return _components;
						}

						inline const solids::lib::video::sink::d3d11::base::service& engine::services(void) const
						{
							return _services;
						}

						inline void engine::register_device_notify(solids::lib::video::sink::d3d11::base::base_device_notify * deviceNotify)
						{
							_device_notify = deviceNotify;
						}

						inline std::function<void* ()> engine::get_window_cb(void) const
						{
							return _fn_get_window;
						}

						inline solids::lib::video::sink::d3d11::base::content_manager& engine::content(void)
						{
							return _content_manager;
						}
					};
				};
			};
		};
	};
};