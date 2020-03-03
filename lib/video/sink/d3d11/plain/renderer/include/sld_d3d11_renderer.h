#ifndef _SLD_D3D11_RENDERER_H_
#define _SLD_D3D11_RENDERER_H_

#if defined(EXPORT_SLD_D3D11_LIB)
#  define EXP_SLD_D3D11_CLASS __declspec(dllexport)
#else
#  define EXP_SLD_D3D11_CLASS __declspec(dllimport)
#endif

#include <sld.h>

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
						class EXP_SLD_D3D11_CLASS renderer
							: public solids::lib::base
						{
							class core;
						public:
							renderer(void);
							~renderer(void);

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

							BOOL			is_media_type_supported(int32_t input_width, int32_t input_height, int32_t output_width, int32_t output_height,  int32_t input_num_fps, int32_t input_den_fps, int32_t output_num_fps, int32_t output_den_fps, int32_t dxgi_format);
							BOOL			check_swap_chain(void);

						private:
							solids::lib::video::sink::d3d11::plain::renderer::core * _core;
						};
					};
				};
			};
		};
	};
};













#endif