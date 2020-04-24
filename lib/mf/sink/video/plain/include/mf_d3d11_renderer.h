#ifndef _SLD_MF_D3D11_RENDERER_H_
#define _SLD_MF_D3D11_RENDERER_H_

#include <mf_base.h>
#include <mf_video_display.h>
#include <sld_d3d11_renderer.h>

namespace solids
{
	namespace lib
	{
		namespace mf
		{
			namespace sink
			{
				namespace video
				{
					namespace plain
					{
                        class renderer
                            : solids::lib::mf::base
                            , solids::lib::mf::refcount_object
                            , public IMFVideoDisplayControl
                            , public IMFGetService
                        {
                        public:
                            renderer(void);
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

                            BOOL        can_process_next_sample(void);
                            HRESULT     flush(void);
                            HRESULT     get_monitor_refresh_rate(DWORD* pdwMonitorRefreshRate);
                            HRESULT     is_media_type_supported(IMFMediaType* pMediaType, DXGI_FORMAT dxgiFormat);
                            HRESULT     present(void);
                            HRESULT     process_sample(IMFMediaType* pCurrentType, IMFSample* pSample, UINT32* punInterlaceMode, BOOL* pbDeviceChanged, BOOL* pbProcessAgain, IMFSample** ppOutputSample = NULL);
                            HRESULT     set_current_media_type(IMFMediaType* pMediaType);
                            HRESULT     shutdown(void);

                        private:
                            void        AspectRatioCorrectSize(LPSIZE lpSizeImage,  const SIZE& sizeAr, const SIZE& sizeOrig, BOOL ScaleXorY);
                            void        CheckDecodeSwitchRegKey(void);
                            HRESULT     CheckDeviceState(BOOL* pbDeviceChanged);
                            BOOL        check_empty_rect(RECT* pDst);
                            HRESULT     check_shutdown(void) const;
                            HRESULT     create_dxgi_manager_and_device();
                            HRESULT     GetVideoDisplayArea(IMFMediaType* pType, MFVideoArea* pArea);
                            void        PixelAspectToPictureAspect(int Width, int Height, int PixelAspectX, int PixelAspectY, int* pPictureAspectX, int* pPictureAspectY);
                            HRESULT     process_sample(ID3D11Texture2D * pTexture2D, UINT dwViewIndex, RECT rcDest, UINT32 unInterlaceMode, IMFSample** ppVideoOutFrame);
                            void        ReduceToLowestTerms(int NumeratorIn, int DenominatorIn, int* pNumeratorOut, int* pDenominatorOut);
                            HRESULT     SetMonitor(UINT adapterID);
                            HRESULT     SetVideoMonitor(HWND hwndVideo);

                            solids::lib::mf::critical_section                   _lock;                  // critical section for thread safety
                            BOOL                                                _is_shutdown;               // Flag to indicate if shutdown() method was called.
                            
                            solids::lib::video::sink::d3d11::plain::renderer::context_t _d3d11_renderer_ctx;
                            solids::lib::video::sink::d3d11::plain::renderer *  _d3d11_renderer;

                            IMFDXGIDeviceManager *                              m_pDXGIManager;
                            IDXGIOutput1 *                                      m_pDXGIOutput1;
                            IMFVideoSampleAllocatorEx *                         m_pSampleAllocatorEx;
                            HWND                                                m_hwndVideo;
                            solids::lib::mf::sink::monitors *                   m_pMonitors;
                            solids::lib::mf::sink::monitor_t *                  m_lpCurrMon;
                            UINT                                                m_DeviceResetToken;
                            UINT                                                m_ConnectionGUID;
                            UINT                                                m_DXSWSwitch;
                            UINT                                                m_useDebugLayer;
                            BOOL                                                _can_process_next_sample;
                        };
					};
				};
			};
		};
	};
};






#endif