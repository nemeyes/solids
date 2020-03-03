#ifndef _D3D11_RENDERER_H_
#define _D3D11_RENDERER_H_

#include "sld_d3d11_renderer.h"

#include <d3d11.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#include <DirectxMath.h>
#include <d3dcompiler.h>

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
					namespace plain
					{
						class renderer::core
						{
						public:
							core(void);
							~core(void);

							BOOL			is_initialized(void);
							int32_t			initialize(void);
							int32_t			release(void);

							BOOL			present(void);

							ID3D11Device *	d3d11_dev(void);

							void			set_fullscreen(BOOL fs);
							BOOL			get_fullscreen(void);

							void			set_image_resolution(int32_t width, int32_t height);
							void			get_image_resolution(int32_t & width, int32_t & height);

							void			set_display_rect(RECT display_rect);
							void			set_real_display_resolution(int32_t width, int32_t height);

							HRESULT			create_d3d11_dev(int32_t useDebugLayer);

							HRESULT			process_sample(HWND hwnd, ID3D11Texture2D * input, int32_t vi, RECT rcDst, D3D11_VIDEO_FRAME_FORMAT interlace, ID3D11Texture2D ** output);

							void			release_d3d11_dev(void);
							void			release_d3d11_video_dev(void);
							void			release_d3d11_video_processor_enum(void);
							void			release_d3d11_video_processor(void);
							void			release_swap_chain(void);
							BOOL			is_media_type_supported(int32_t input_width, int32_t input_height, int32_t output_width, int32_t output_height, int32_t input_num_fps, int32_t input_den_fps, int32_t output_num_fps, int32_t output_den_fps, DXGI_FORMAT dxgi_format);
							BOOL			check_swap_chain(void);

						private:
							HRESULT			update_swap_chain(int32_t width, int32_t height, HWND hwnd);
							void			update_rectangles(RECT* dst, RECT* src);
							void			letter_box_dst_rect(LPRECT lprcLBDst, const RECT& src, const RECT& dst);
							HRESULT			find_bob_processor_index(DWORD* index);
							void			set_video_context_parameters(ID3D11VideoContext* vc, const RECT* src, const RECT* dst, const RECT* target, D3D11_VIDEO_FRAME_FORMAT interlace);

						private:
							BOOL					_is_initialized;
							BOOL					_fullscreen;
							IDXGIFactory2 *			_dxgi_factory;
							ID3D11Device *			_d3d11_dev;
							ID3D11DeviceContext *	_d3d11_dev_ctx;

							ID3D11VideoDevice *					_d3d11_video_dev;
							ID3D11VideoProcessorEnumerator *	_d3d11_video_processor_enum;
							ID3D11VideoProcessor *				_d3d11_video_processor;
							IDXGISwapChain1 *					_swap_chain;

							IDXGIOutput1 *						_dxgi_output;

							int32_t								_image_width;
							int32_t								_image_height;

							RECT								_rc_src;
							RECT								_rc_dst;
							RECT								_display_rect;
							int32_t								_real_display_width;
							int32_t								_real_display_height;
						};
					};
				};
			};
		};
	};
};












#endif