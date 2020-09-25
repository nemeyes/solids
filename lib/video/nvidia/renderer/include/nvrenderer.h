#pragma once

#include "sld_nvrenderer.h"
#include <directxmath.h>
#include <d3dcompiler.h>
#include <atlbase.h>
#include <d3d11.h>
#include <d3d11_1.h>
#include <d3d11_2.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#include <dxgi1_3.h>

#include <cuda.h>
#include <cudaD3D11.h>

namespace solids
{
	namespace lib
	{
		namespace video
		{
			namespace nvidia
			{
				class renderer::core
				{
				public:
					core(solids::lib::video::nvidia::renderer* front);
					virtual ~core(void);

					BOOL	is_initialized(void);
					int32_t	initialize(solids::lib::video::nvidia::renderer::context_t* ctx);
					int32_t	release(void);
					int32_t	render(uint8_t* deviceptr, int32_t pitch);

				private:
					solids::lib::video::nvidia::renderer* _front;
					solids::lib::video::nvidia::renderer::context_t* _ctx;
					BOOL						_is_initialized;
					CRITICAL_SECTION			_lock;
					CUcontext					_cu_ctx;
					CUgraphicsResource			_cu_resource;

					ID3D11Device*				_dev;
					ID3D11DeviceContext*		_dev_ctx;
					IDXGISwapChain*				_swc;
					ID3D11Texture2D*			_back_buffer;

					ID3D11Texture2D*			_staging;

					ID3D11RenderTargetView*		_rtv;
					ID3DBlob*					_cvs;
					ID3DBlob*					_cps;
					ID3D11VertexShader*			_vs;
					ID3D11PixelShader*			_ps;
				};

			};
		};
	};
};


