#include "d3d11_mv_renderer.h"

#define BOARDER_RATIO_X		0.007
#define BOARDER_RATIO_Y		0.007

solids::lib::video::sink::d3d11::multiview::renderer::core::core(void)
	: _device(NULL)
	, _device_context(NULL)
	, _swap_chain(NULL)
	, _pcb_matrix_buffer(NULL)
	, _vs(NULL)
	, _ps(NULL)
	, _vs_buffer(NULL)
	, _ps_buffer(NULL)
	, _vertex_buffer(NULL)
	, _index_buffer(NULL)
	, _sampler_state(NULL)
	, _render_target_view(NULL)
	, _selected_index(0)
	, _initialized(FALSE)
	, _is_maximize(FALSE)
	, _render_type(solids::lib::video::sink::d3d11::multiview::renderer::render_mode_t::original)
	, _background_srv(NULL)
	, _background_buffer(NULL)
{

}

solids::lib::video::sink::d3d11::multiview::renderer::core::~core(void)
{

}

void solids::lib::video::sink::d3d11::multiview::renderer::core::initialize(solids::lib::video::sink::d3d11::multiview::renderer::context_t* ctx)
{
	_device = ctx->dev;
	_device_context = ctx->devctx;
	_view_info = ctx->vi;
	_swap_chain = ctx->sw;
	_view_count = ctx->vc;
	_main_width = ctx->width;
	_main_height = ctx->height;

	create_render_target_view();
	create_shader();
	create_vertex_buffer();
	create_index_buffer();
	create_input_layout();
	create_view_port();
	create_sample_state();
	create_constant_buffer();
	create_matrix();
	create_background_srv();

	_initialized = TRUE;
}

void solids::lib::video::sink::d3d11::multiview::renderer::core::release(void)
{
	safe_release(_vs);
	safe_release(_ps);
	safe_release(_vs_buffer);
	safe_release(_ps_buffer);
	safe_release(_vertex_buffer);
	safe_release(_index_buffer);
	safe_release(_sampler_state);
	safe_release(_pcb_matrix_buffer);
	safe_release(_background_srv);
	safe_release(_background_buffer);
	_initialized = FALSE;
}

void solids::lib::video::sink::d3d11::multiview::renderer::core::select(int32_t index)
{
	_selected_index = index;
}

void solids::lib::video::sink::d3d11::multiview::renderer::core::maximize(void)
{
	_is_maximize = !_is_maximize;
	if (_is_maximize)
	{

		float scaling_x = 2 / abs((_view_info[_selected_index].position[1] - _view_info[_selected_index].position[0]));
		float scaling_y = 2 / abs((_view_info[_selected_index].position[3] - _view_info[_selected_index].position[2]));
		float translate_x = -1 - _view_info[_selected_index].position[0] * scaling_x;
		float translate_y = 1 - _view_info[_selected_index].position[2] * scaling_y;
		_world_matrix = DirectX::XMMatrixScaling(scaling_x, scaling_y, 1) * DirectX::XMMatrixTranslation(translate_x, translate_y, 0);
	}
	else
	{
		_world_matrix = DirectX::XMMatrixIdentity();
	}
	//_cam_position = XMVectorSet(0.0f, 0.0f, -1.0f, 0.0f);
	//_cam_target = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
	//_cam_up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
	//_view_matrix = XMMatrixLookAtLH(_cam_position, _cam_target, _cam_up);
	//_projection_matrix = XMMatrixPerspectiveFovLH(0.5f * 3.14f, 1 / 1, 1.0f, 1000.0f);
	_cb_matrix_buffer.WVP = DirectX::XMMatrixTranspose(_world_matrix * _view_matrix * _proj_matrix);
	_device_context->UpdateSubresource(_pcb_matrix_buffer, 0, 0, &_cb_matrix_buffer, 0, 0);
	_device_context->VSSetConstantBuffers(0, 1, &_pcb_matrix_buffer);
}

void solids::lib::video::sink::d3d11::multiview::renderer::core::set_render_mode(int32_t mode)
{
	_render_type = mode;
}

void solids::lib::video::sink::d3d11::multiview::renderer::core::set_shader_resource_view(uint32_t index, ID3D11Texture2D * buffer)
{
	D3D11_TEXTURE2D_DESC surfaceDesc;
	buffer->GetDesc(&surfaceDesc);
	surfaceDesc.Width = _view_info[index].dst_width;
	surfaceDesc.Height = _view_info[index].dst_height;
	surfaceDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	_device->CreateTexture2D(&surfaceDesc, 0, &_view_info[index].buffer);

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	ZeroMemory(&srvDesc, sizeof(srvDesc));
	srvDesc.Format = surfaceDesc.Format;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MipLevels = 1;
	srvDesc.Texture2D.MostDetailedMip = 0;
	_device->CreateShaderResourceView(_view_info[index].buffer, &srvDesc, &_view_info[index].shader_resource_view);
}

void solids::lib::video::sink::d3d11::multiview::renderer::core::render(void)
{

}

void solids::lib::video::sink::d3d11::multiview::renderer::core::create_render_target_view(int32_t type)
{
	_layout = type;
	HRESULT hr = S_OK;
	safe_release(_render_target_view);
	ID3D11Texture2D* backBuffer = NULL;
	hr = _swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&backBuffer);
	hr = _device->CreateRenderTargetView(backBuffer, NULL, &_render_target_view);
	_device_context->OMSetRenderTargets(1, &_render_target_view, NULL);

	safe_release(backBuffer);
}

void solids::lib::video::sink::d3d11::multiview::renderer::core::create_shader(void)
{
	D3DReadFileToBlob(L"mv_vertex_shader.cso", &_vs_buffer);
	D3DReadFileToBlob(L"mv_pixel_shader.cso", &_ps_buffer);

	_device->CreateVertexShader(_vs_buffer->GetBufferPointer(), _vs_buffer->GetBufferSize(), 0, &_vs);
	_device->CreatePixelShader(_ps_buffer->GetBufferPointer(), _ps_buffer->GetBufferSize(), 0, &_ps);

	_device_context->VSSetShader(_vs, 0, 0);
	_device_context->PSSetShader(_ps, 0, 0);
}

void solids::lib::video::sink::d3d11::multiview::renderer::core::create_vertex_buffer(void)
{
	int32_t	vertex_count = _view_count * 3 + 1; /// +1 : Background , * 3 : view area, image area, focus area
	solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t * v = new solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t[vertex_count * 4];


	//View area
	int32_t start_index = 0;
	for (int i = 0; i < _view_count; i++)
	{
		start_index = i * 4;
		v[start_index] = solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t(_view_info[i].position[0], _view_info[i].position[2], 0, 0);
		v[start_index + 1] = solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t(_view_info[i].position[1], _view_info[i].position[2], _view_info[i].active_video_ratio[0], 0);
		v[start_index + 2] = solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t(_view_info[i].position[0], _view_info[i].position[3], 0, _view_info[i].active_video_ratio[1]);
		v[start_index + 3] = solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t(_view_info[i].position[1], _view_info[i].position[3], _view_info[i].active_video_ratio[0], _view_info[i].active_video_ratio[1]);
	}

	//Image area (via aspect ratio)
	/*if (_render_type == cap_base::render_type_t::stretch)
	{
		for (int i = _view_count; i < 2 * _view_count; i++)
		{
			start_index = i * 4;
			v[start_index] = Vertex(_view_info[i].position[0], _view_info[i].position[2], 0, 0);
			v[start_index + 1] = Vertex(_view_info[i].position[1], _view_info[i].position[2], _view_info[i].active_video_ratio[0], 0);
			v[start_index + 2] = Vertex(_view_info[i].position[0], _view_info[i].position[3], 0, _view_info[i].active_video_ratio[1]);
			v[start_index + 3] = Vertex(_view_info[i].position[1], _view_info[i].position[3], _view_info[i].active_video_ratio[0], _view_info[i].active_video_ratio[1]);
		}
	}
	else if (_render_type == cap_base::render_type_t::original)*/
	{
		//할당된 뷰 영역의 Swapchain 관점에서의 넓이, 높이
		float view_width;
		float view_height;

		float aspect_ratio_view_area; //뷰 영역의 비율
		float aspect_ratio_video_area;	//영상의 비율

		//최종적으로 영상이 그려질 좌표
		float determined_video_area_left = 0;
		float determined_video_area_right = 0;
		float determined_video_area_top = 0;
		float determined_video_area_bottom = 0;

		float offset = 0; //조정된 영상영역의 뷰 영역에 대한 offset (x or y)
		for (int i = 0; i < _view_count; i++)
		{
			view_width = ((float)_main_width) * (_view_info[i].position[1] - _view_info[i].position[0]) / 2;
			view_height = ((float)_main_height) * (_view_info[i].position[2] - _view_info[i].position[3]) / 2;
			aspect_ratio_view_area = view_height / view_width;
			aspect_ratio_video_area = (_view_info[i].dst_height * _view_info[i].active_video_ratio[1]) / (_view_info[i].dst_width * _view_info[i].active_video_ratio[0]);
			if (aspect_ratio_view_area > aspect_ratio_video_area)
			{
				determined_video_area_left = _view_info[i].position[0];
				determined_video_area_right = _view_info[i].position[1];
				offset = (_view_info[i].position[2] - _view_info[i].position[3]) * ((view_height - view_width * aspect_ratio_video_area) / 2) / view_height;
				determined_video_area_top = _view_info[i].position[2] - offset;
				determined_video_area_bottom = _view_info[i].position[3] + offset;
			}
			else if (aspect_ratio_view_area < aspect_ratio_video_area)
			{
				offset = (_view_info[i].position[1] - _view_info[i].position[0]) * ((view_width - view_height / aspect_ratio_video_area) / 2) / view_width;
				determined_video_area_left = _view_info[i].position[0] + offset;
				determined_video_area_right = _view_info[i].position[1] - offset;
				determined_video_area_top = _view_info[i].position[2];
				determined_video_area_bottom = _view_info[i].position[3];
			}
			else
			{
				determined_video_area_left = _view_info[i].position[0];
				determined_video_area_right = _view_info[i].position[1];
				determined_video_area_top = _view_info[i].position[2];
				determined_video_area_bottom = _view_info[i].position[3];
			}

			start_index = (i + _view_count) * 4;
			v[start_index] = solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t(determined_video_area_left, determined_video_area_top, 0, 0);
			v[start_index + 1] = solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t(determined_video_area_right, determined_video_area_top, _view_info[i].active_video_ratio[0], 0);
			v[start_index + 2] = solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t(determined_video_area_left, determined_video_area_bottom, 0, _view_info[i].active_video_ratio[1]);
			v[start_index + 3] = solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t(determined_video_area_right, determined_video_area_bottom, _view_info[i].active_video_ratio[0], _view_info[i].active_video_ratio[1]);
		}
	}


	//Focus Area(Selected)
	//Texture coordinates can't be minus, but here it is.
	//To distinguish focus area at Pixel Shader stage. 
	for (int i = 2 * _view_count; i < 3 * _view_count; i++)
	{
		start_index = i * 4;
		v[start_index] = solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t(_view_info[i - 2 * _view_count].position[0] - BOARDER_RATIO_X, _view_info[i - 2 * _view_count].position[2] + BOARDER_RATIO_Y, -1, -1);
		v[start_index + 1] = solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t(_view_info[i - 2 * _view_count].position[1] + BOARDER_RATIO_X, _view_info[i - 2 * _view_count].position[2] + BOARDER_RATIO_Y, -1, -1);
		v[start_index + 2] = solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t(_view_info[i - 2 * _view_count].position[0] - BOARDER_RATIO_X, _view_info[i - 2 * _view_count].position[3] - BOARDER_RATIO_Y, -1, -1);
		v[start_index + 3] = solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t(_view_info[i - 2 * _view_count].position[1] + BOARDER_RATIO_X, _view_info[i - 2 * _view_count].position[3] - BOARDER_RATIO_Y, -1, -1);
	}

	//BackGround
	v[_view_count * 3 * 4] = solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t(-1, 1, 0, 0);
	v[_view_count * 3 * 4 + 1] = solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t(1, 1, 1, 0);
	v[_view_count * 3 * 4 + 2] = solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t(-1, -1, 0, 1);
	v[_view_count * 3 * 4 + 3] = solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t(1, -1, 1, 1);



	D3D11_BUFFER_DESC vertexBufferDesc;
	ZeroMemory(&vertexBufferDesc, sizeof(vertexBufferDesc));

	vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
	vertexBufferDesc.ByteWidth = sizeof(solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t) * vertex_count * 4;
	vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vertexBufferDesc.CPUAccessFlags = 0;
	vertexBufferDesc.MiscFlags = 0;

	D3D11_SUBRESOURCE_DATA vertexBufferData;
	ZeroMemory(&vertexBufferData, sizeof(vertexBufferData));
	vertexBufferData.pSysMem = v;

	HRESULT hr = _device->CreateBuffer(&vertexBufferDesc, &vertexBufferData, &_vertex_buffer);

	//Set the vertex buffer
	UINT stride = sizeof(solids::lib::video::sink::d3d11::multiview::renderer::core::vertex_t);
	UINT offset = 0;
	_device_context->IASetVertexBuffers(0, 1, &_vertex_buffer, &stride, &offset);

	delete[] v;
}

void solids::lib::video::sink::d3d11::multiview::renderer::core::create_index_buffer(void)
{
	int32_t	vertex_count = _view_count * 3 + 1; /// +1 : Background , * 3 : view area, image area, focus area
	DWORD* indices = new DWORD[vertex_count * 6];

	int32_t _view_number = 0;
	int32_t _view_top_left = 0;

	for (int i = 0; i <= 3 * _view_count; i++)
	{
		_view_number = i * 6;
		_view_top_left = i * 4;
		indices[_view_number] = _view_top_left;
		indices[_view_number + 1] = _view_top_left + 1;
		indices[_view_number + 2] = _view_top_left + 3;
		indices[_view_number + 3] = _view_top_left;
		indices[_view_number + 4] = _view_top_left + 3;
		indices[_view_number + 5] = _view_top_left + 2;
	}

	D3D11_BUFFER_DESC indexBufferDesc;
	ZeroMemory(&indexBufferDesc, sizeof(indexBufferDesc));

	indexBufferDesc.Usage = D3D11_USAGE_DEFAULT;

	indexBufferDesc.ByteWidth = sizeof(DWORD) * vertex_count * 6;
	//indexBufferDesc.ByteWidth = sizeof(DWORD) * ARRAYSIZE(indices);

	indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	indexBufferDesc.CPUAccessFlags = 0;
	indexBufferDesc.MiscFlags = 0;

	D3D11_SUBRESOURCE_DATA iinitData;
	iinitData.pSysMem = indices;
	HRESULT hr = _device->CreateBuffer(&indexBufferDesc, &iinitData, &_index_buffer);
	_device_context->IASetIndexBuffer(_index_buffer, DXGI_FORMAT_R32_UINT, 0);

	delete[] indices;
}

void solids::lib::video::sink::d3d11::multiview::renderer::core::create_input_layout(void)
{
	_device->CreateInputLayout(input_element_desc, ARRAYSIZE(input_element_desc), _vs_buffer->GetBufferPointer(), _vs_buffer->GetBufferSize(), &_vertex_layout);
	_device_context->IASetInputLayout(_vertex_layout);
	_device_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
}

void solids::lib::video::sink::d3d11::multiview::renderer::core::create_view_port(void)
{
	D3D11_VIEWPORT viewport;
	ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));
	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;
	viewport.Width = _main_width;
	viewport.Height = _main_height;
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;
	//Set the Viewport
	_device_context->RSSetViewports(1, &viewport);
}

void solids::lib::video::sink::d3d11::multiview::renderer::core::create_sample_state(void)
{
	D3D11_SAMPLER_DESC sampDesc;
	ZeroMemory(&sampDesc, sizeof(sampDesc));
	sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	sampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	sampDesc.MinLOD = 0;
	sampDesc.MaxLOD = D3D11_FLOAT32_MAX;
	sampDesc.BorderColor;

	HRESULT hr = _device->CreateSamplerState(&sampDesc, &_sampler_state);
}

void solids::lib::video::sink::d3d11::multiview::renderer::core::create_constant_buffer(void)
{
	D3D11_BUFFER_DESC cbbd;
	ZeroMemory(&cbbd, sizeof(D3D11_BUFFER_DESC));

	cbbd.Usage = D3D11_USAGE_DEFAULT;
	cbbd.ByteWidth = sizeof(solids::lib::video::sink::d3d11::multiview::renderer::core::matrix_t);
	cbbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbbd.CPUAccessFlags = 0;
	cbbd.MiscFlags = 0;

	HRESULT hr = _device->CreateBuffer(&cbbd, NULL, &_pcb_matrix_buffer);
}

void solids::lib::video::sink::d3d11::multiview::renderer::core::create_matrix(void)
{
	_world_matrix = DirectX::XMMatrixIdentity();
	_cam_position = DirectX::XMVectorSet(0.0f, 0.0f, -1.0f, 0.0f);
	_cam_target = DirectX::XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
	_cam_up = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
	_view_matrix = DirectX::XMMatrixLookAtLH(_cam_position, _cam_target, _cam_up);
	_proj_matrix = DirectX::XMMatrixPerspectiveFovLH(0.5f * 3.14f, 1 / 1, 1.0f, 1000.0f);
	_cb_matrix_buffer.WVP = DirectX::XMMatrixTranspose(_world_matrix * _view_matrix * _proj_matrix);
	_device_context->UpdateSubresource(_pcb_matrix_buffer, 0, 0, &_cb_matrix_buffer, 0, 0);
	_device_context->VSSetConstantBuffers(0, 1, &_pcb_matrix_buffer);
}

void solids::lib::video::sink::d3d11::multiview::renderer::core::create_background_srv(void)
{
	D3D11_TEXTURE2D_DESC surfaceDesc;
	surfaceDesc.Width = 1;
	surfaceDesc.Height = 1;
	surfaceDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	_device->CreateTexture2D(&surfaceDesc, 0, &_background_buffer);

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	ZeroMemory(&srvDesc, sizeof(srvDesc));
	srvDesc.Format = surfaceDesc.Format;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MipLevels = 1;
	srvDesc.Texture2D.MostDetailedMip = 0;
	_device->CreateShaderResourceView(_background_buffer, &srvDesc, &_background_srv);
}
