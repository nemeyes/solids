#pragma once

#include <winrt\Windows.Foundation.h>
#include <cstdint>
#include <d3d11.h>
#include <DirectXMath.h>
#include <dxgidebug.h>
#include <gsl\gsl>
#include "d3d11_exception.h"

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
						typedef struct _shader_stage_t
						{
							static const int32_t IA = 0;
							static const int32_t VS = 1;
							static const int32_t HS = 2;
							static const int32_t DS = 3;
							static const int32_t GS = 4;
							static const int32_t SO = 5;
							static const int32_t RS = 6;
							static const int32_t PS = 7;
							static const int32_t OM = 8;
							static const int32_t CS = 9;
						} shader_stage_t;

						const std::array<int32_t, 6> programmable_graphicshader_states
						{
							shader_stage_t::VS,
							shader_stage_t::HS,
							shader_stage_t::DS,
							shader_stage_t::GS,
							shader_stage_t::PS,
							shader_stage_t::CS,
						};

						inline BOOL is_shaderstage_programmable(int32_t stage)
						{
							static const std::map<int32_t, bool> programmableMap
							{
								{ shader_stage_t::IA, FALSE },
								{ shader_stage_t::VS, TRUE },
								{ shader_stage_t::HS, TRUE },
								{ shader_stage_t::DS, TRUE },
								{ shader_stage_t::GS, TRUE },
								{ shader_stage_t::SO, FALSE },
								{ shader_stage_t::RS, FALSE },
								{ shader_stage_t::PS, TRUE },
								{ shader_stage_t::OM, FALSE },
								{ shader_stage_t::CS, TRUE },
							};
							return programmableMap.at(stage);
						}

						void create_index_buffer(gsl::not_null<ID3D11Device*> device, const gsl::span<const std::uint16_t>& indices, gsl::not_null<ID3D11Buffer**> indexBuffer);
						void create_index_buffer(gsl::not_null<ID3D11Device*> device, const gsl::span<const std::uint32_t>& indices, gsl::not_null<ID3D11Buffer**> indexBuffer);
						void create_constant_buffer(gsl::not_null<ID3D11Device*> device, std::size_t byteWidth, gsl::not_null<ID3D11Buffer**> constantBuffer);

						inline float convert_dips_to_pixels(float dips, float dpi)
						{
							static const float dipsPerInch = 96.0f;
							return floorf(dips * dpi / dipsPerInch + 0.5f);
						}

#if defined(DEBUG) || defined(_DEBUG)
						inline BOOL is_sdklayer_available(void)
						{
							HRESULT hr = D3D11CreateDevice(
								nullptr,
								D3D_DRIVER_TYPE_NULL,       // There is no need to create a real hardware device.
								0,
								D3D11_CREATE_DEVICE_DEBUG,  // Check for the SDK layers.

								nullptr,                    // Any feature level will do.
								0,
								D3D11_SDK_VERSION,          // Always set this to D3D11_SDK_VERSION for Windows Store apps.
								nullptr,                    // No need to keep the D3D device reference.
								nullptr,                    // No need to know the feature level.
								nullptr                     // No need to keep the D3D device context reference.
							);
							return SUCCEEDED(hr);
						}
#endif

#if defined(DEBUG) || defined(_DEBUG)
						inline void dump_d3d11_debug(void)
						{
							winrt::com_ptr<IDXGIDebug1> debugInterface = nullptr;
							solids::lib::video::sink::d3d11::base::throw_if_failed(DXGIGetDebugInterface1(0, IID_PPV_ARGS(debugInterface.put())));
							solids::lib::video::sink::d3d11::base::throw_if_failed(debugInterface->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_ALL));
						}
#endif
					};
				};
			};
		};
	};
};