#include "Nvrenderer.h"
#include <sld_locks.h>

namespace solids
{
	namespace lib
	{
		namespace video
		{
			namespace nvidia
			{

				renderer::core::core(solids::lib::video::nvidia::renderer* front)
					: _front(front)
					, _ctx(NULL)
					, _is_initialized(FALSE)
					, _dev(NULL)
					, _dev_ctx(NULL)
					, _swc(NULL)
					, _back_buffer(NULL)
					, _staging(NULL)
					, _rtv(NULL)
					, _cvs(NULL)
					, _cps(NULL)
					, _vs(NULL)
					, _ps(NULL)
				{
					::InitializeCriticalSection(&_lock);
				}

				renderer::core::~core(void)
				{
					::DeleteCriticalSection(&_lock);
				}

				BOOL renderer::core::is_initialized(void)
				{
					return _is_initialized;
				}

				int32_t	renderer::core::initialize(solids::lib::video::nvidia::renderer::context_t* ctx)
				{
					IDXGIFactory3* pDXGIFactory = NULL;
					IDXGIAdapter1* pDXGIAdapter = NULL;
					if (ctx->hwnd == NULL)
						return solids::lib::video::nvidia::renderer::err_code_t::generic_fail;
					release();
					_ctx = ctx;
					_cu_ctx = *((CUcontext*)_ctx->cuctx);

					HRESULT hr = ::CreateDXGIFactory1(__uuidof(IDXGIFactory), (void**)&pDXGIFactory);
					if (FAILED(hr))
						return solids::lib::video::nvidia::renderer::err_code_t::generic_fail;

					BOOL bFound = FALSE;
					int32_t index = 0;
					for (int32_t i = 0; (pDXGIFactory->EnumAdapters1(i, &pDXGIAdapter) != DXGI_ERROR_NOT_FOUND); i++)
					{
						DXGI_ADAPTER_DESC1 desc;
						pDXGIAdapter->GetDesc1(&desc);

						if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
							continue;

						if (desc.VendorId == 0x000010DE)
						{
							bFound = TRUE;
							break;
						}

						index++;

						if (pDXGIAdapter)
							pDXGIAdapter->Release();
						pDXGIAdapter = NULL;
					}

					if (pDXGIFactory)
						pDXGIFactory->Release();
					pDXGIFactory = NULL;

					DXGI_SWAP_CHAIN_DESC sc = { 0 };
					sc.BufferCount = 1;
					sc.BufferDesc.Width = _ctx->width;
					sc.BufferDesc.Height = _ctx->height;
					sc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
					sc.BufferDesc.RefreshRate.Numerator = 0;
					sc.BufferDesc.RefreshRate.Denominator = 1;
					sc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
					sc.OutputWindow = _ctx->hwnd;
					sc.SampleDesc.Count = 1;
					sc.SampleDesc.Quality = 0;
					sc.Windowed = TRUE;

					hr = S_OK;
					if (pDXGIAdapter)
						hr = ::D3D11CreateDeviceAndSwapChain(pDXGIAdapter, D3D_DRIVER_TYPE_UNKNOWN, NULL, 0, NULL, 0, D3D11_SDK_VERSION, &sc, &_swc, &_dev, NULL, &_dev_ctx);
					else
						hr = ::D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, 0, NULL, 0, D3D11_SDK_VERSION, &sc, &_swc, &_dev, NULL, &_dev_ctx);

					hr = _swc->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&_back_buffer);

					D3D11_TEXTURE2D_DESC desc;
					_back_buffer->GetDesc(&desc);
					desc.BindFlags = 0;
					desc.MipLevels = 1;
					desc.Usage = D3D11_USAGE_DEFAULT;
					desc.BindFlags = (D3D11_USAGE_DEFAULT | D3D11_BIND_SHADER_RESOURCE);
					desc.CPUAccessFlags = 0;
					desc.SampleDesc.Count = 1;
					desc.SampleDesc.Quality = 0;
					hr = _dev->CreateTexture2D(&desc, NULL, &_staging);

					{
						::cuInit(0);
						::cuCtxPushCurrent(_cu_ctx);
						::cuGraphicsD3D11RegisterResource(&_cu_resource, _staging, CU_GRAPHICS_REGISTER_FLAGS_NONE);
						::cuGraphicsResourceSetMapFlags(_cu_resource, CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD);
						::cuCtxPopCurrent(NULL);
					}

					if (pDXGIAdapter)
						pDXGIAdapter->Release();
					pDXGIAdapter = NULL;

					_is_initialized = TRUE;
					return solids::lib::video::nvidia::renderer::err_code_t::success;
				}

				int32_t	renderer::core::release(void)
				{
					if (!_is_initialized)
						return solids::lib::video::nvidia::renderer::err_code_t::success;

					solids::lib::autolock lock(&_lock);
					{
						::cuCtxPushCurrent(_cu_ctx);
						::cuGraphicsUnregisterResource(_cu_resource);
						::cuCtxPopCurrent(NULL);
					}

					if (_staging)
					{
						_staging->Release();
						_staging = NULL;
					}
					if (_back_buffer)
					{
						_back_buffer->Release();
						_back_buffer = NULL;
					}
					if (_dev_ctx)
					{
						_dev_ctx->Release();
						_dev_ctx = NULL;
					}
					if (_dev)
					{
						_dev->Release();
						_dev = NULL;
					}
					if (_swc)
					{
						_swc->Release();
						_swc = NULL;
					}
					_is_initialized = FALSE;
					return solids::lib::video::nvidia::renderer::err_code_t::success;
				}

				int32_t	renderer::core::render(uint8_t* deviceptr, int32_t pitch)
				{
					::cuCtxPushCurrent(_cu_ctx);
					::cuGraphicsMapResources(1, &_cu_resource, 0);

					CUarray dstArray;
					::cuGraphicsSubResourceGetMappedArray(&dstArray, _cu_resource, 0, 0);

					CUDA_MEMCPY2D m = { 0 };
					m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
					m.srcDevice = (CUdeviceptr)deviceptr;
					m.srcPitch = pitch;
					m.dstMemoryType = CU_MEMORYTYPE_ARRAY;
					m.dstArray = dstArray;
					m.WidthInBytes = _ctx->width << 2;
					m.Height = _ctx->height;
					::cuMemcpy2D(&m);

					::cuGraphicsUnmapResources(1, &_cu_resource, 0);
					::cuCtxPopCurrent(NULL);

					HRESULT hr;
					ID3D11Resource* d3d11Resource = NULL;
					if (SUCCEEDED(hr = _staging->QueryInterface(__uuidof(ID3D11Resource), (void**)&d3d11Resource)))
					{
						_dev_ctx->ResolveSubresource(d3d11Resource, 0, _staging, 0, DXGI_FORMAT_B8G8R8A8_UNORM);
						_dev_ctx->CopyResource(_back_buffer, _staging);
						if (d3d11Resource)
						{
							d3d11Resource->Release();
							d3d11Resource = NULL;
						}
						_swc->Present(0, 0);
					}
					return renderer::err_code_t::success;
				}

			};
		};
	};
};

