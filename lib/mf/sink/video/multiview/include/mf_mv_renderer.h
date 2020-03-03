#ifndef _SLD_MF_MV_RENDERER_H_
#define _SLD_MF_MV_RENDERER_H_

#include <mf_base.h>
#include <mf_video_display.h>
#include <sld_d3d11_mv_renderer.h>

namespace sld
{
	namespace lib
	{
		namespace mf
		{
			namespace sink
			{
				namespace video
				{
					namespace multiview
					{
						class d3d11_renderer;
						class media;
						class renderer
							: sld::lib::mf::base
							, sld::lib::mf::refcount_object
							, public IMFVideoDisplayControl
							, public IMFGetService
						{
						public:
							renderer(sld::lib::mf::sink::video::multiview::media * media_sink);
							virtual ~renderer(void);

							// IUnknown
							STDMETHODIMP_(ULONG) AddRef(void);
							STDMETHODIMP QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void** ppv);
							STDMETHODIMP_(ULONG) Release(void);

							// IMFVideoDisplayControl
							STDMETHODIMP GetAspectRatioMode(__RPC__out DWORD* pdwAspectRatioMode) { return E_NOTIMPL; }
							STDMETHODIMP GetBorderColor(__RPC__out COLORREF* pClr) { return E_NOTIMPL; }
							STDMETHODIMP GetCurrentImage(__RPC__inout BITMAPINFOHEADER* pBih, __RPC__deref_out_ecount_full_opt(*pcbDib) BYTE** pDib, __RPC__out DWORD* pcbDib, __RPC__inout_opt LONGLONG* pTimestamp) { return E_NOTIMPL; }
							STDMETHODIMP GetFullscreen(__RPC__out BOOL* pfFullscreen);
							STDMETHODIMP GetIdealVideoSize(__RPC__inout_opt SIZE* pszMin, __RPC__inout_opt SIZE* pszMax) { return E_NOTIMPL; }
							STDMETHODIMP GetNativeVideoSize(__RPC__inout_opt SIZE* pszVideo, __RPC__inout_opt SIZE* pszARVideo) { return E_NOTIMPL; }
							STDMETHODIMP GetRenderingPrefs(__RPC__out DWORD* pdwRenderFlags) { return E_NOTIMPL; }
							STDMETHODIMP GetVideoPosition(__RPC__out MFVideoNormalizedRect* pnrcSource, __RPC__out LPRECT prcDest) { return E_NOTIMPL; }
							STDMETHODIMP GetVideoWindow(__RPC__deref_out_opt HWND* phwndVideo) { return E_NOTIMPL; }
							STDMETHODIMP RepaintVideo(void) { return E_NOTIMPL; }
							STDMETHODIMP SetAspectRatioMode(DWORD dwAspectRatioMode) { return E_NOTIMPL; }
							STDMETHODIMP SetBorderColor(COLORREF Clr) { return E_NOTIMPL; }
							STDMETHODIMP SetFullscreen(BOOL fFullscreen);
							STDMETHODIMP SetRenderingPrefs(DWORD dwRenderingPrefs) { return E_NOTIMPL; }
							STDMETHODIMP SetVideoPosition(__RPC__in_opt const MFVideoNormalizedRect* pnrcSource, __RPC__in_opt const LPRECT prcDest) { return E_NOTIMPL; }
							STDMETHODIMP SetVideoWindow(__RPC__in HWND hwndVideo);

							// IMFGetService
							STDMETHODIMP GetService(__RPC__in REFGUID guidService, __RPC__in REFIID riid, __RPC__deref_out_opt LPVOID* ppvObject);

							BOOL    can_process_next_sample(void);
							HRESULT flush(void);
							HRESULT get_monitor_refresh_rate(DWORD* pdwMonitorRefreshRate);
							HRESULT is_media_type_supported(IMFMediaType* pMediaType, DXGI_FORMAT dxgiFormat);
							HRESULT render_samples(IMFSample* sample);
							HRESULT process_samples(IMFMediaType* pCurrentType, IMFSample* pSample, UINT32* punInterlaceMode, BOOL* pbDeviceChanged, BOOL* pbProcessAgain, IMFSample** ppOutputSample = NULL);
							HRESULT set_current_media_type(IMFMediaType* pMediaType);
							HRESULT release(void);

							//IPresenter
							void	set_view_count(INT interval);
							void	enable_coordinated_cs_converter(BOOL enable);
							void	set_view_position(INT index, FLOAT* position);
							void	set_selected(INT index);
							void	maximize(void);
							void	change_render_type(void);

						private:
							void    check_decode_switch_regkey(void);
							HRESULT check_device_state(BOOL* device_changed);
							BOOL    check_empty_rect(RECT* pDst);
							HRESULT check_shutdown(void) const;
							HRESULT create_dxgi_manager_and_device(D3D_DRIVER_TYPE DriverType = D3D_DRIVER_TYPE_HARDWARE);

							HRESULT	create_video_processor(UINT index, ID3D11Texture2D * pSrcTexture2D, UINT32 unInterlaceMode);
							HRESULT	colorspace_convert(UINT view_index, ID3D11Texture2D * pSrcTexture2D, UINT resource_index, UINT32 unInterlaceMode);
							HRESULT find_bob_processor_index(DWORD * pIndex, UINT index);
							HRESULT set_monitor(UINT adapterID);

							HRESULT set_video_monitor(HWND hwndVideo);
							_Post_satisfies_(this->_swap_chain1 != NULL)
							HRESULT update_dxgi_swap_chain(void);

						private:
							sld::lib::mf::critical_section		_lock;
							BOOL									_is_shutdown;
							BOOL									_enable_coordinated_cs_converter;

							D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC	_output_view_desc;
							D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC	_input_view_desc;
							D3D11_VIDEO_PROCESSOR_STREAM			_stream_data;

							int32_t									_view_count;
							sld::lib::video::sink::d3d11::multiview::renderer::view_session_t * _view_info;

							ID3D11VideoContext *					_d3d11_video_context;
							HWND									_hwnd;
							CMonitorArray *							_monitors;
							CAMDDrawMonitorInfo *					_current_monitor;
							UINT									_device_reset_token;
							UINT									_connection_guid;
							UINT									_dx_sw_switch;
							UINT									_use_dcomp_visual;
							UINT									_use_debug_layer;

							IMFDXGIDeviceManager *					_dxgi_manager;
							IDXGIFactory2 *							_dxgi_factory2;
							ID3D11Device *							_d3d11_device;
							ID3D11DeviceContext *					_d3d11_immediate_context;
							IDXGIOutput1 *							_dxgi_output1;
							IMFVideoSampleAllocatorEx *				_sample_allocator_ex;
							IDCompositionDevice *					_dcomp_device;
							IDCompositionTarget *					_hwnd_target;
							IDCompositionVisual *					_root_visual;
							ID3D11VideoDevice *						_video_device;
							IDXGISwapChain1 *						_swap_chain1;

							BOOL									_is_full_screen;
							BOOL									_can_process_next_sample;
							//RECT									_back_buffer;

							sld::lib::video::sink::d3d11::multiview::renderer::context_t _d3d_render_ctx;
							sld::lib::video::sink::d3d11::multiview::renderer * _d3d_renderer;

							int32_t									_current_time;
							BOOL									_first_sample;
							
							sld::lib::mf::sink::video::multiview::media * _ms;
							BOOL									_maximize;
							int32_t									_selected;
							int32_t									_render_type;
						};
					};
				};
			};
		};
	};
};

#endif