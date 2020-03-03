#ifndef _SLD_D3D11_MV_RENDERER_H_
#define _SLD_D3D11_MV_RENDERER_H_

#if defined(EXPORT_SLD_D3D11_MV_LIB)
#  define EXP_SLD_D3D11_MV_CLASS __declspec(dllexport)
#else
#  define EXP_SLD_D3D11_MV_CLASS __declspec(dllimport)
#endif

#include <sld.h>
#include <d3d11.h>
#include <dxgi.h>
#include <dxgi1_2.h>

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
					namespace multiview
					{
						class EXP_SLD_D3D11_MV_CLASS renderer
							: public solids::lib::base
						{
							class core;
						public:
							typedef struct _layout_t
							{
								static const int32_t layout_2x2 = 0;
								static const int32_t layout_3x3 = 1;
								static const int32_t layout_1_1x3 = 2;
							} layout_t;

							typedef struct _view_session_t
							{
								ID3D11Texture2D* buffer;
								ID3D11ShaderResourceView* shader_resource_view;
								ID3D11VideoProcessorEnumerator* video_processor_enum;
								ID3D11VideoProcessor* video_processor;
								uint32_t	src_height;
								uint32_t	src_width;
								uint32_t	dst_height;
								uint32_t	dst_width;
								float		position[4];
								float		active_video_ratio[2];	// width, height
								int32_t		control[4];	//left, right, up, down
								_view_session_t(void)
									: src_height(0)
									, src_width(0)
									, dst_height(0)
									, dst_width(0)
									, video_processor_enum(NULL)
									, video_processor(NULL)
									, buffer(NULL)
									, shader_resource_view(NULL)
								{
									active_video_ratio[0] = 1;
									active_video_ratio[1] = 1;
									::memset(control, 0x00, sizeof(control));
								};
							} view_session_t;

							typedef struct _context_t
							{
								ID3D11Device* dev;
								ID3D11DeviceContext* devctx;
								IDXGISwapChain1* sw;
								int32_t					vc;
								solids::lib::video::sink::d3d11::multiview::renderer::view_session_t* vi;
								int32_t					width;
								int32_t					height;
							} context_t;

							renderer(void);
							~renderer(void);

							void	initialize(solids::lib::video::sink::d3d11::multiview::renderer::context_t * ctx);
							void	release(void);
							void	select(int32_t index);
							void	maximize(void);
							void	set_render_mode(int32_t mode);
							void	set_shader_resource_view(uint32_t index, ID3D11Texture2D * buffer);
							void	render(void);

						private:
							solids::lib::video::sink::d3d11::multiview::renderer* _core;

							/*
							private:
								void create_render_target_view(int32_t type = sld::lib::video::sink::d3d11::multiview::renderer::layout_t::layout_2x2);
								void create_shader(void);
								void create_vertex_buffer(void);
								void create_index_buffer(void);
								void create_input_layout(void);
								void create_view_port(void);
								void create_sample_state(void);
								void create_constant_buffer(void);
								void create_matrix(void);
							*/
						};
					};
				};
			};
		};
	};
};

#endif