#include "d3d11_engine.h"
#include "d3d11_exception.h"
#include "d3d11_drawable_component.h"
#include "d3d11_dx_helper.h"
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
	RTTI_DEFINITIONS(engine)

	engine::engine(std::function<void* ()> getWindowCB, std::function<void(SIZE&)> getRenderTargetSizeCB)
		: _fn_get_window(getWindowCB)
		, _fn_get_render_target_size(getRenderTargetSizeCB)
		, _content_manager(*this)
	{
		assert(getWindowCB != nullptr);
		assert(getRenderTargetSizeCB != nullptr);

		create_device_independent_resources();
		create_device_resources();
	}

	void engine::initialize(void)
	{
		solids::lib::video::sink::d3d11::base::content_type_reader_manager::initialize(*this);
		_clock.reset();

		for (auto& cmpnt : _components)
		{
			cmpnt->initialize();
		}
	}

	void engine::run(void)
	{
		_clock.update(_time);
		update(_time);
		draw(_time);
	}

	void engine::release(void)
	{
		for (auto& cmpnt : _components)
		{
			cmpnt->release();
		}

		_d3d11_ctx->ClearState();
		_d3d11_ctx->Flush();

		_components.clear();
		_components.shrink_to_fit();

		_depth_stencil_view = nullptr;
		_render_target_view = nullptr;
		_swapchain = nullptr;
		_d3d11_ctx = nullptr;
		_d3d11_dev = nullptr;

		_content_manager.clear();
		solids::lib::video::sink::d3d11::base::content_type_reader_manager::release();

#if defined(DEBUG) || defined(_DEBUG)
		dump_d3d11_debug();
#endif
	}

	void engine::update(const solids::lib::video::sink::d3d11::base::time& tm)
	{
		for (auto& cmpnt : _components)
		{
			if (cmpnt->enabled())
			{
				cmpnt->update(tm);
			}
		}
	}

	void engine::draw(const solids::lib::video::sink::d3d11::base::time& tm)
	{
		for (auto& cmpnt : _components)
		{
			solids::lib::video::sink::d3d11::base::drawable_component* dcmpnt = cmpnt->as<solids::lib::video::sink::d3d11::base::drawable_component>();
			if (dcmpnt != nullptr && dcmpnt->visible())
			{
				dcmpnt->draw(tm);
			}
		}
	}

	void engine::update_render_target_size(void)
	{
		create_window_size_dependent_resources();
	}

	void engine::begin(void)
	{
		ID3D11RenderTargetView* views[] = { _render_target_view.get() };
		const gsl::span<ID3D11RenderTargetView*> rtvs{ views };
		solids::lib::video::sink::d3d11::base::render_target::begin(d3d11ctx(), rtvs, gsl::not_null<ID3D11DepthStencilView*>(_depth_stencil_view.get()), _viewport);
	}

	void engine::end(void)
	{
		solids::lib::video::sink::d3d11::base::render_target::end(d3d11ctx());
	}

	void engine::create_device_independent_resources(void)
	{

	}

	void engine::create_device_resources(void)
	{
		uint32_t createDeviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

#if defined(_DEBUG)
		if (is_sdklayer_available())
		{
			createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
		}
#endif
		D3D_FEATURE_LEVEL featureLevels[] = {
			D3D_FEATURE_LEVEL_11_1,
			D3D_FEATURE_LEVEL_11_0,
			D3D_FEATURE_LEVEL_10_1,
			D3D_FEATURE_LEVEL_10_0
		};

		winrt::com_ptr<ID3D11Device> d3d11dev;
		winrt::com_ptr<ID3D11DeviceContext> d3d11ctx;
		solids::lib::video::sink::d3d11::base::throw_if_failed(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, NULL, createDeviceFlags, featureLevels, gsl::narrow_cast<uint32_t>(std::size(featureLevels)), 
																					D3D11_SDK_VERSION, d3d11dev.put(), &_feature_level, d3d11ctx.put()), "D3D11CreateDevice() failed.");
		_d3d11_dev = d3d11dev.as<ID3D11Device3>();
		assert(_d3d11_dev != nullptr);

		_d3d11_ctx = d3d11ctx.as<ID3D11DeviceContext3>();
		assert(_d3d11_ctx != nullptr);

		solids::lib::video::sink::d3d11::base::throw_if_failed(_d3d11_dev->CheckMultisampleQualityLevels(DXGI_FORMAT_R8G8B8A8_UNORM, _multisampling_count, &_multisampling_quality_levels), "CheckMultisampleQualityLevels() failed.");
		if (_multisampling_quality_levels == 0)
		{
			throw solids::lib::video::sink::d3d11::base::exception("unsupported multi-sampling quality.");
		}

#ifndef NDEBUG
		winrt::com_ptr<ID3D11Debug> d3d11Debug = _d3d11_dev.as<ID3D11Debug>();
		if (d3d11Debug)
		{
			winrt::com_ptr<ID3D11InfoQueue> d3d11InfoQueue = d3d11Debug.as<ID3D11InfoQueue>();
			if (d3d11InfoQueue)
			{
#ifdef _DEBUG
				d3d11InfoQueue->SetBreakOnSeverity(D3D11_MESSAGE_SEVERITY_CORRUPTION, TRUE);
				d3d11InfoQueue->SetBreakOnSeverity(D3D11_MESSAGE_SEVERITY_ERROR, TRUE);
#endif
				D3D11_MESSAGE_ID hide[] =
				{
					D3D11_MESSAGE_ID_SETPRIVATEDATA_CHANGINGPARAMS
				};
				D3D11_INFO_QUEUE_FILTER filter{ 0 };
				filter.DenyList.NumIDs = gsl::narrow_cast<uint32_t>(std::size(hide));
				filter.DenyList.pIDList = hide;
				d3d11InfoQueue->AddStorageFilterEntries(&filter);
			}
		}
#endif
	}

	void engine::create_window_size_dependent_resources(void)
	{
		ID3D11RenderTargetView* nullViews[] = { nullptr };
		_d3d11_ctx->OMSetRenderTargets(gsl::narrow_cast<uint32_t>(std::size(nullViews)), nullViews, nullptr);
		_render_target_view = nullptr;
		_depth_stencil_view = nullptr;
		_d3d11_ctx->Flush();

		_fn_get_render_target_size(_render_target_size);
		if (_swapchain == nullptr)
		{
			DXGI_SWAP_CHAIN_DESC1 swapChainDesc{ 0 };
			swapChainDesc.Width = _render_target_size.cx;
			swapChainDesc.Height = _render_target_size.cy;
			swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			swapChainDesc.SampleDesc.Count = _multisampling_count;
			swapChainDesc.SampleDesc.Quality = _multisampling_quality_levels - 1;
			swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
			swapChainDesc.BufferCount = _default_buffer_count;
			swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

			winrt::com_ptr<IDXGIDevice> dxgiDevice = _d3d11_dev.as<IDXGIDevice>();
			assert(dxgiDevice != nullptr);

			winrt::com_ptr<IDXGIAdapter> dxgiAdapter;
			solids::lib::video::sink::d3d11::base::throw_if_failed(dxgiDevice->GetAdapter(dxgiAdapter.put()));

			winrt::com_ptr<IDXGIFactory2> dxgiFactory;
			solids::lib::video::sink::d3d11::base::throw_if_failed(dxgiFactory->GetParent(IID_PPV_ARGS(dxgiFactory.put())));

			void* window = _fn_get_window();

			DXGI_SWAP_CHAIN_FULLSCREEN_DESC fullScreenDesc{ 0 };
			fullScreenDesc.RefreshRate.Numerator = _fps;
			fullScreenDesc.RefreshRate.Denominator = 1;
			fullScreenDesc.Windowed = !_bfullscreen;
			solids::lib::video::sink::d3d11::base::throw_if_failed(dxgiFactory->CreateSwapChainForHwnd(_d3d11_dev.get(), reinterpret_cast<HWND>(window), &swapChainDesc, &fullScreenDesc, nullptr, _swapchain.put()), "IDXGIDevice::CreateSwapChainForHwnd() failed.");
		}
		else
		{
			HRESULT hr = _swapchain->ResizeBuffers(_default_buffer_count, _render_target_size.cx, _render_target_size.cy, DXGI_FORMAT_R8G8B8A8_UNORM, 0);
			if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET)
			{
				handle_device_lost();
				return;
			}
			else
			{
				solids::lib::video::sink::d3d11::base::throw_if_failed(hr, "IDXGISwapChain::ResizeBuffers() failed.");
			}
		}

		winrt::com_ptr<ID3D11Texture2D> backbuffer;
		solids::lib::video::sink::d3d11::base::throw_if_failed(_swapchain->GetBuffer(0, IID_PPV_ARGS(backbuffer.put())), "IDXGISwapChain::GetBuffer() failed.");
		backbuffer->GetDesc(&_backbuffer_desc);
		solids::lib::video::sink::d3d11::base::throw_if_failed(_d3d11_dev->CreateRenderTargetView(backbuffer.get(), nullptr, _render_target_view.put()), "IDXGIDevice::CreateRenderTargetView() failed.");

		D3D11_TEXTURE2D_DESC depthStencilDesc{ 0 };
		depthStencilDesc.Width = _render_target_size.cx;
		depthStencilDesc.Height = _render_target_size.cy;
		depthStencilDesc.MipLevels = 1;
		depthStencilDesc.ArraySize = 1;
		depthStencilDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
		depthStencilDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
		depthStencilDesc.Usage = D3D11_USAGE_DEFAULT;
		depthStencilDesc.SampleDesc.Count = _multisampling_count;
		depthStencilDesc.SampleDesc.Quality = _multisampling_quality_levels - 1;
		winrt::com_ptr<ID3D11Texture2D> depthStencilBuffer;
		solids::lib::video::sink::d3d11::base::throw_if_failed(_d3d11_dev->CreateTexture2D(&depthStencilDesc, nullptr, depthStencilBuffer.put()), "IDXGIDevice::CreateTexture2D() failed.");

		CD3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc(D3D11_DSV_DIMENSION_TEXTURE2DMS);
		solids::lib::video::sink::d3d11::base::throw_if_failed(_d3d11_dev->CreateDepthStencilView(depthStencilBuffer.get(), &depthStencilViewDesc, _depth_stencil_view.put()), "IDXGIDevice::CreateDepthStencilView() failed.");
		
		//set the viewport to the entire window
		_viewport = CD3D11_VIEWPORT(0.0f, 0.0f, static_cast<float>(_render_target_size.cx), static_cast<float>(_render_target_size.cy));

		//set render targets and viewport through render target stack
		begin();
	}

	void engine::handle_device_lost(void)
	{
		_swapchain = nullptr;
		if (_device_notify != nullptr)
		{
			_device_notify->on_device_lost();
		}

		create_device_resources();
		create_window_size_dependent_resources();

		if (_device_notify != nullptr)
		{
			_device_notify->on_device_restored();
		}
	}

};
};
};
};
};
};