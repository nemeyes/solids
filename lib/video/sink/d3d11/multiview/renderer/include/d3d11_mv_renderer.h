#ifndef _D3D11_MV_RENDERER_H_
#define _D3D11_MV_RENDERER_H_

#include "sld_d3d11_mv_renderer.h"
#include <DirectxMath.h>
#include <d3dcompiler.h>

const D3D11_INPUT_ELEMENT_DESC input_element_desc[] =
{
	{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
};

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
						class renderer::core
						{
						public:
							core(void);
							~core(void);

							void initialize(solids::lib::video::sink::d3d11::multiview::renderer::context_t* ctx);
							void release(void);
							void select(int32_t index);
							void maximize(void);
							void set_render_mode(int32_t mode);
							void set_shader_resource_view(uint32_t index, ID3D11Texture2D* buffer);
							void render(void);

						private:
							void create_render_target_view(int32_t type = solids::lib::video::sink::d3d11::multiview::renderer::layout_t::layout_2x2);
							void create_shader(void);
							void create_vertex_buffer(void);
							void create_index_buffer(void);
							void create_input_layout(void);
							void create_view_port(void);
							void create_sample_state(void);
							void create_constant_buffer(void);
							void create_matrix(void);
							void create_background_srv(void);

							template <class T> inline void safe_release(T*& pt)
							{
								if (pt != NULL)
								{
									pt->Release();
									pt = NULL;
								}
							}

						private:
							typedef struct _vertex_t	//Vertex Structure
							{
								_vertex_t(void) {}
								_vertex_t(float x, float y, float u, float v)
									: pos(x, y, 0)
									, texCoord(u, v) 
								{}

								_vertex_t(float x, float y, float z, float u, float v)
									: pos(x, y, z)
									, texCoord(u, v) 
								{}

								DirectX::XMFLOAT3 pos;
								DirectX::XMFLOAT2 texCoord;
							} vertex_t;

							typedef struct _matrix_t
							{
								DirectX::XMMATRIX  WVP;
							} matrix_t;

						private:
							ID3D11Device *				_device;
							ID3D11DeviceContext *		_device_context;
							IDXGISwapChain1 *			_swap_chain;


							ID3D11RenderTargetView *	_render_target_view;
							ID3DBlob *					_vs_buffer;
							ID3DBlob *					_ps_buffer;
							ID3D11VertexShader *		_vs;
							ID3D11PixelShader *			_ps;
							ID3D11Buffer *				_vertex_buffer;
							ID3D11Buffer *				_index_buffer;
							ID3D11InputLayout *			_vertex_layout;
							ID3D11SamplerState *		_sampler_state;
							ID3D11Buffer *				_pcb_matrix_buffer;

							matrix_t					_cb_matrix_buffer;
							DirectX::XMMATRIX			_world_matrix;
							DirectX::XMMATRIX			_view_matrix;
							DirectX::XMMATRIX			_proj_matrix;

							DirectX::XMVECTOR			_cam_position;
							DirectX::XMVECTOR			_cam_target;
							DirectX::XMVECTOR			_cam_up;

							int32_t						_view_count;
							solids::lib::video::sink::d3d11::multiview::renderer::view_session_t * _view_info;
							int32_t						_layout;

							int32_t						_selected_index;
							BOOL						_initialized;
							int32_t						_render_type;
							BOOL						_is_maximize;
							//internal helper member value
							int32_t						_main_width;
							int32_t						_main_height;

							ID3D11Texture2D *			_background_buffer;
							ID3D11ShaderResourceView *	_background_srv;
						};
					};
				};
			};
		};
	};
};

#endif