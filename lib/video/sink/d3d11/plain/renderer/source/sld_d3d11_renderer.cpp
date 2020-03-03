#include "sld_d3d11_renderer.h"
#include "d3d11_renderer.h"

sld::lib::video::sink::d3d11::plain::renderer::renderer(void)
	: _core(NULL)
{
	_core = new sld::lib::video::sink::d3d11::plain::renderer::core();
}

sld::lib::video::sink::d3d11::plain::renderer::~renderer(void)
{
	if (_core)
		delete _core;
	_core = NULL;
}

BOOL sld::lib::video::sink::d3d11::plain::renderer::is_initialized(void)
{
	return _core->is_initialized();
}

int32_t sld::lib::video::sink::d3d11::plain::renderer::initialize(void)
{
	return _core->initialize();
}

int32_t sld::lib::video::sink::d3d11::plain::renderer::release(void)
{
	return _core->release();
}

BOOL sld::lib::video::sink::d3d11::plain::renderer::present(void)
{
	return _core->present();
}

ID3D11Device* sld::lib::video::sink::d3d11::plain::renderer::d3d11_dev(void)
{
	return _core->d3d11_dev();
}

void sld::lib::video::sink::d3d11::plain::renderer::set_fullscreen(BOOL fs)
{
	_core->set_fullscreen(fs);
}

BOOL sld::lib::video::sink::d3d11::plain::renderer::get_fullscreen(void)
{
	return _core->get_fullscreen();
}

void sld::lib::video::sink::d3d11::plain::renderer::set_image_resolution(int32_t width, int32_t height)
{
	_core->set_image_resolution(width, height);
}

void sld::lib::video::sink::d3d11::plain::renderer::get_image_resolution(int32_t& width, int32_t& height)
{
	_core->get_image_resolution(width, height);
}

void sld::lib::video::sink::d3d11::plain::renderer::set_display_rect(RECT display_rect)
{
	_core->set_display_rect(display_rect);
}

void sld::lib::video::sink::d3d11::plain::renderer::set_real_display_resolution(int32_t width, int32_t height)
{
	_core->set_real_display_resolution(width, height);
}

HRESULT sld::lib::video::sink::d3d11::plain::renderer::create_d3d11_dev(int32_t useDebugLayer)
{
	return _core->create_d3d11_dev(useDebugLayer);
}

HRESULT sld::lib::video::sink::d3d11::plain::renderer::process_sample(HWND hwnd, ID3D11Texture2D * input, int32_t vi, RECT rcDst, D3D11_VIDEO_FRAME_FORMAT interlace, ID3D11Texture2D ** output)
{
	return _core->process_sample(hwnd, input, vi, rcDst, interlace, output);
}

void sld::lib::video::sink::d3d11::plain::renderer::release_d3d11_dev(void)
{
	return _core->release_d3d11_dev();
}

void sld::lib::video::sink::d3d11::plain::renderer::release_d3d11_video_dev(void)
{
	return _core->release_d3d11_video_dev();
}

void sld::lib::video::sink::d3d11::plain::renderer::release_d3d11_video_processor_enum(void)
{
	return _core->release_d3d11_video_processor_enum();
}

void sld::lib::video::sink::d3d11::plain::renderer::release_d3d11_video_processor(void)
{
	return _core->release_d3d11_video_processor();
}

void sld::lib::video::sink::d3d11::plain::renderer::release_swap_chain(void)
{
	return _core->release_swap_chain();
}

BOOL sld::lib::video::sink::d3d11::plain::renderer::is_media_type_supported(int32_t input_width, int32_t input_height, int32_t output_width, int32_t output_height, int32_t input_num_fps, int32_t input_den_fps, int32_t output_num_fps, int32_t output_den_fps, int32_t dxgi_format)
{
	return _core->is_media_type_supported(input_width, input_height, output_width, output_height, input_num_fps, input_den_fps, output_num_fps, output_den_fps, DXGI_FORMAT(dxgi_format));
}

BOOL sld::lib::video::sink::d3d11::plain::renderer::check_swap_chain(void)
{
	return _core->check_swap_chain();
}