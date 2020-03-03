#include "mf_d3d11_renderer.h"

solids::lib::mf::sink::video::plain::renderer::renderer(void)
    : _lock()
    , _is_shutdown(FALSE)
    , m_pDXGIManager(NULL)
    , m_pDXGIOutput1(NULL)
    , m_pSampleAllocatorEx(NULL)
    , m_hwndVideo(NULL)
    , m_pMonitors(NULL)
    , m_lpCurrMon(NULL)
    , m_DeviceResetToken(0)
    , m_DXSWSwitch(0)
    , m_useDebugLayer(D3D11_CREATE_DEVICE_VIDEO_SUPPORT)
    , _can_process_next_sample(TRUE)
{
    _d3d11_renderer = new solids::lib::video::sink::d3d11::plain::renderer();
}

solids::lib::mf::sink::video::plain::renderer::~renderer(void)
{
    solids::lib::mf::safe_delete(m_pMonitors);
    if (_d3d11_renderer)
        delete _d3d11_renderer;
    _d3d11_renderer = NULL;
}

ULONG solids::lib::mf::sink::video::plain::renderer::AddRef(void)
{
    return solids::lib::mf::refcount_object::AddRef();
}

HRESULT solids::lib::mf::sink::video::plain::renderer::QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void** ppv)
{
    if (!ppv)
    {
        return E_POINTER;
    }
    if (iid == IID_IUnknown)
    {
        *ppv = static_cast<IUnknown*>(static_cast<IMFVideoDisplayControl*>(this));
    }
    else if (iid == __uuidof(IMFVideoDisplayControl))
    {
        *ppv = static_cast<IMFVideoDisplayControl*>(this);
    }
    else if (iid == __uuidof(IMFGetService))
    {
        *ppv = static_cast<IMFGetService*>(this);
    }
    else
    {
        *ppv = NULL;
        return E_NOINTERFACE;
    }
    AddRef();
    return S_OK;
}

ULONG  solids::lib::mf::sink::video::plain::renderer::Release(void)
{
    return solids::lib::mf::refcount_object::Release();
}

// IMFVideoDisplayControl
HRESULT solids::lib::mf::sink::video::plain::renderer::GetFullscreen(__RPC__out BOOL* pfFullscreen)
{
    solids::lib::mf::auto_lock lock(&_lock);

    HRESULT hr = check_shutdown();
    if (FAILED(hr))
    {
        return hr;
    }

    if (pfFullscreen == NULL)
    {
        return E_POINTER;
    }

    *pfFullscreen = _d3d11_renderer->get_fullscreen();

    return S_OK;
}

// IMFVideoDisplayControl
HRESULT solids::lib::mf::sink::video::plain::renderer::SetFullscreen(BOOL fs)
{
    solids::lib::mf::auto_lock lock(&_lock);

    HRESULT hr = check_shutdown();
    if (SUCCEEDED(hr))
    {
        _d3d11_renderer->set_fullscreen(fs);

        _d3d11_renderer->release_d3d11_video_dev();
        _d3d11_renderer->release_d3d11_video_processor_enum();
        _d3d11_renderer->release_d3d11_video_processor();
    }
    return hr;
}

// IMFVideoDisplayControl
HRESULT solids::lib::mf::sink::video::plain::renderer::SetVideoWindow(__RPC__in HWND hwndVideo)
{
    HRESULT hr = S_OK;

    solids::lib::mf::auto_lock lock(&_lock);

    do
    {
        hr = check_shutdown();
        if (FAILED(hr))
            break;

        if (!IsWindow(hwndVideo))
        {
            hr = E_INVALIDARG;
            break;
        }

        m_pMonitors = new solids::lib::mf::sink::monitors();
        if (!m_pMonitors)
        {
            hr = E_OUTOFMEMORY;
            break;
        }

        hr = SetVideoMonitor(hwndVideo);
        if (FAILED(hr))
            break;

        CheckDecodeSwitchRegKey();

        m_hwndVideo = hwndVideo;

        hr = create_dxgi_manager_and_device();
        if (FAILED(hr))
            break;

    } while (FALSE);

    return hr;
}

//-------------------------------------------------------------------------
// Name: GetService
// Description: IMFGetService
//-------------------------------------------------------------------------
HRESULT solids::lib::mf::sink::video::plain::renderer::GetService(__RPC__in REFGUID guidService, __RPC__in REFIID riid, __RPC__deref_out_opt LPVOID* ppvObject)
{
    HRESULT hr = S_OK;

    if (guidService == MR_VIDEO_ACCELERATION_SERVICE)
    {
        if (riid == __uuidof(IMFDXGIDeviceManager))
        {
            if (NULL != m_pDXGIManager)
            {
                *ppvObject = (void*) static_cast<IUnknown*>(m_pDXGIManager);
                ((IUnknown*)*ppvObject)->AddRef();
            }
            else
            {
                hr = E_NOINTERFACE;
            }
        }
        else if (riid == __uuidof(IMFVideoSampleAllocatorEx))
        {
            if (NULL == m_pSampleAllocatorEx)
            {
                hr = MFCreateVideoSampleAllocatorEx(IID_IMFVideoSampleAllocatorEx, (LPVOID*)&m_pSampleAllocatorEx);
                if (SUCCEEDED(hr) && NULL != m_pDXGIManager)
                {
                    hr = m_pSampleAllocatorEx->SetDirectXManager(m_pDXGIManager);
                }
            }
            if (SUCCEEDED(hr))
            {
                hr = m_pSampleAllocatorEx->QueryInterface(riid, ppvObject);
            }
        }
        else
        {
            hr = E_NOINTERFACE;
        }
    }
    else if (guidService == MR_VIDEO_RENDER_SERVICE)
    {
        hr = QueryInterface(riid, ppvObject);
    }
    else
    {
        hr = MF_E_UNSUPPORTED_SERVICE;
    }

    return hr;
}

BOOL solids::lib::mf::sink::video::plain::renderer::can_process_next_sample(void)
{
    return _can_process_next_sample;
}

HRESULT solids::lib::mf::sink::video::plain::renderer::flush(void)
{
    solids::lib::mf::auto_lock lock(&_lock);
    HRESULT hr = check_shutdown();
    _can_process_next_sample = TRUE;
    return hr;
}

HRESULT solids::lib::mf::sink::video::plain::renderer::get_monitor_refresh_rate(DWORD* pdwRefreshRate)
{
    if (pdwRefreshRate == NULL)
    {
        return E_POINTER;
    }

    if (m_lpCurrMon == NULL)
    {
        return MF_E_INVALIDREQUEST;
    }

    *pdwRefreshRate = m_lpCurrMon->refresh_rate;

    return S_OK;
}

HRESULT solids::lib::mf::sink::video::plain::renderer::is_media_type_supported(IMFMediaType* pMediaType, DXGI_FORMAT dxgiFormat)
{
    HRESULT hr = S_OK;
    UINT32 uiNumerator = 30000, uiDenominator = 1001;
    UINT32 uimageWidthInPixels, uimageHeightInPixels = 0;

    do
    {
        hr = check_shutdown();
        if (FAILED(hr))
        {
            break;
        }

        if (pMediaType == NULL)
        {
            hr = E_POINTER;
            break;
        }

        hr = MFGetAttributeSize(pMediaType, MF_MT_FRAME_SIZE, &uimageWidthInPixels, &uimageHeightInPixels);

        if (FAILED(hr))
        {
            break;
        }

        MFGetAttributeRatio(pMediaType, MF_MT_FRAME_RATE, &uiNumerator, &uiDenominator);

        BOOL supported = _d3d11_renderer->is_media_type_supported(uimageWidthInPixels, uimageHeightInPixels, uimageWidthInPixels, uimageHeightInPixels, uiNumerator, uiDenominator, uiNumerator, uiDenominator, dxgiFormat);
        if (!supported)
        {
            hr = MF_E_UNSUPPORTED_D3D_TYPE;
            break;
        }
    } while (FALSE);

    return hr;
}

HRESULT solids::lib::mf::sink::video::plain::renderer::present(void)
{
    HRESULT hr = S_OK;

    solids::lib::mf::auto_lock lock(&_lock);

    do
    {
        hr = check_shutdown();
        if (FAILED(hr))
            break;

        if (!_d3d11_renderer->check_swap_chain())
            break;

        RECT rcDest;
        ZeroMemory(&rcDest, sizeof(rcDest));
        if (check_empty_rect(&rcDest))
        {
            hr = S_OK;
            break;
        }

        if (!_d3d11_renderer->present())
        {
            hr = E_FAIL;
            break;
        }

        _can_process_next_sample = TRUE;
    } while (FALSE);

    return hr;
}

HRESULT solids::lib::mf::sink::video::plain::renderer::process_sample(IMFMediaType * pCurrentType, IMFSample * pSample, UINT32 * punInterlaceMode, BOOL * pbDeviceChanged, BOOL * pbProcessAgain, IMFSample ** ppOutputSample)
{
    HRESULT             hr              = S_OK;
    BYTE *              pData           = NULL;
    DWORD               dwSampleSize    = 0;
    IMFMediaBuffer *    pBuffer         = NULL;
    DWORD               cBuffers        = 0;
    ID3D11Texture2D *   pTexture2D      = NULL;
    IMFDXGIBuffer *     pDXGIBuffer     = NULL;
    ID3D11Device *      pDeviceInput    = NULL;
    UINT                dwViewIndex     = 0;

    solids::lib::mf::auto_lock lock(&_lock);

    do
    {
        hr = check_shutdown();
        if (FAILED(hr))
            break;

        if (punInterlaceMode == NULL || pCurrentType == NULL || pSample == NULL || pbDeviceChanged == NULL || pbProcessAgain == NULL)
        {
            hr = E_POINTER;
            break;
        }

        *pbProcessAgain = FALSE;
        *pbDeviceChanged = FALSE;

        hr = pSample->GetBufferCount(&cBuffers);
        if (FAILED(hr))
            break;

        if (cBuffers == 1)
            hr = pSample->GetBufferByIndex(0, &pBuffer);
        else
            hr = pSample->ConvertToContiguousBuffer(&pBuffer);

        if (FAILED(hr))
            break;

        hr = CheckDeviceState(pbDeviceChanged);
        if (FAILED(hr))
            break;

        RECT rcDest;
        ZeroMemory(&rcDest, sizeof(rcDest));
        if (check_empty_rect(&rcDest))
        {
            hr = S_OK;
            break;
        }

        MFVideoInterlaceMode unInterlaceMode = (MFVideoInterlaceMode)MFGetAttributeUINT32(pCurrentType, MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
        if (MFVideoInterlace_MixedInterlaceOrProgressive == unInterlaceMode)
        {
            BOOL fInterlaced = MFGetAttributeUINT32(pSample, MFSampleExtension_Interlaced, FALSE);
            if (!fInterlaced)
            {
                // Progressive sample
                *punInterlaceMode = MFVideoInterlace_Progressive;
            }
            else
            {
                BOOL fBottomFirst = MFGetAttributeUINT32(pSample, MFSampleExtension_BottomFieldFirst, FALSE);
                if (fBottomFirst)
                {
                    *punInterlaceMode = MFVideoInterlace_FieldInterleavedLowerFirst;
                }
                else
                {
                    *punInterlaceMode = MFVideoInterlace_FieldInterleavedUpperFirst;
                }
            }
        }

        hr = pBuffer->QueryInterface(__uuidof(IMFDXGIBuffer), (LPVOID*)&pDXGIBuffer);
        if (FAILED(hr))
            break;

        hr = pDXGIBuffer->GetResource(__uuidof(ID3D11Texture2D), (LPVOID*)&pTexture2D);
        if (FAILED(hr))
            break;

        hr = pDXGIBuffer->GetSubresourceIndex(&dwViewIndex);
        if (FAILED(hr))
            break;

        pTexture2D->GetDevice(&pDeviceInput);
        
        ID3D11Device * d3d11_dev = NULL;
        d3d11_dev = _d3d11_renderer->d3d11_dev();
        if ((NULL == pDeviceInput) || (pDeviceInput != d3d11_dev))
            break;

        hr = process_sample(pTexture2D, dwViewIndex, rcDest, *punInterlaceMode, ppOutputSample);
        
        LONGLONG hnsDuration = 0;
        LONGLONG hnsTime = 0;
        DWORD dwSampleFlags = 0;
        if (ppOutputSample != NULL && *ppOutputSample != NULL)
        {
            if (SUCCEEDED(pSample->GetSampleDuration(&hnsDuration)))
            {
                (*ppOutputSample)->SetSampleDuration(hnsDuration);
            }

            if (SUCCEEDED(pSample->GetSampleTime(&hnsTime)))
            {
                (*ppOutputSample)->SetSampleTime(hnsTime);
            }

            if (SUCCEEDED(pSample->GetSampleFlags(&dwSampleFlags)))
            {
                (*ppOutputSample)->SetSampleFlags(dwSampleFlags);
            }
        }
    } while (FALSE);

    solids::lib::mf::safe_release(pTexture2D);
    solids::lib::mf::safe_release(pDXGIBuffer);
    solids::lib::mf::safe_release(pDeviceInput);
    solids::lib::mf::safe_release(pBuffer);

    return hr;
}

HRESULT solids::lib::mf::sink::video::plain::renderer::set_current_media_type(IMFMediaType* pMediaType)
{
    HRESULT hr = S_OK;
    IMFAttributes* pAttributes = NULL;

    solids::lib::mf::auto_lock lock(&_lock);

    do
    {
        hr = check_shutdown();
        if (FAILED(hr))
        {
            break;
        }

        hr = pMediaType->QueryInterface(IID_IMFAttributes, reinterpret_cast<void**>(&pAttributes));
        if (FAILED(hr))
        {
            break;
        }

        //Now Determine Correct Display Resolution
        if (SUCCEEDED(hr))
        {
            UINT32 parX = 0, parY = 0;
            int PARWidth = 0, PARHeight = 0;
            MFVideoArea videoArea = { 0 };

            if (FAILED(MFGetAttributeSize(pMediaType, MF_MT_PIXEL_ASPECT_RATIO, &parX, &parY)))
            {
                parX = 1;
                parY = 1;
            }

            hr = GetVideoDisplayArea(pMediaType, &videoArea);
            if (FAILED(hr))
            {
                break;
            }

            _d3d11_renderer->set_display_rect(solids::lib::mf::mf_video_area_to_rect(videoArea));

            PixelAspectToPictureAspect(videoArea.Area.cx, videoArea.Area.cy, parX, parY, &PARWidth, &PARHeight);

            SIZE szVideo = videoArea.Area;
            SIZE szPARVideo = { PARWidth, PARHeight };
            AspectRatioCorrectSize(&szVideo, szPARVideo, videoArea.Area, FALSE);
            _d3d11_renderer->set_real_display_resolution(int32_t(szVideo.cx), int32_t(szVideo.cy));
        }

    } while (FALSE);

    solids::lib::mf::safe_release(pAttributes);
    return hr;
}

HRESULT solids::lib::mf::sink::video::plain::renderer::shutdown(void)
{
    solids::lib::mf::auto_lock lock(&_lock);

    HRESULT hr = MF_E_SHUTDOWN;

    _is_shutdown = TRUE;

    solids::lib::mf::safe_release(m_pDXGIManager);
    solids::lib::mf::safe_release(m_pDXGIOutput1);
    solids::lib::mf::safe_release(m_pSampleAllocatorEx);

    if (_d3d11_renderer->is_initialized())
        _d3d11_renderer->release();

    return hr;
}

/// Private methods

//+-------------------------------------------------------------------------
//
//  Function:   AspectRatioCorrectSize
//
//  Synopsis:   Corrects the supplied size structure so that it becomes the same shape
//              as the specified aspect ratio, the correction is always applied in the
//              horizontal axis
//
//--------------------------------------------------------------------------

void solids::lib::mf::sink::video::plain::renderer::AspectRatioCorrectSize(
    LPSIZE lpSizeImage,     // size to be aspect ratio corrected
    const SIZE& sizeAr,     // aspect ratio of image
    const SIZE& sizeOrig,   // original image size
    BOOL ScaleXorY          // axis to correct in
)
{
    int cxAR = sizeAr.cx;
    int cyAR = sizeAr.cy;
    int cxOr = sizeOrig.cx;
    int cyOr = sizeOrig.cy;
    int sx = lpSizeImage->cx;
    int sy = lpSizeImage->cy;

    // MulDiv rounds correctly.
    lpSizeImage->cx = MulDiv((sx * cyOr), cxAR, (cyAR * cxOr));

    if (ScaleXorY && lpSizeImage->cx < cxOr)
    {
        lpSizeImage->cx = cxOr;
        lpSizeImage->cy = MulDiv((sy * cxOr), cyAR, (cxAR * cyOr));
    }
}

void solids::lib::mf::sink::video::plain::renderer::CheckDecodeSwitchRegKey(void)
{
    const TCHAR* lpcszDXSW = TEXT("DXSWSwitch");
    const TCHAR* lpcszInVP = TEXT("XVP");
    const TCHAR* lpcszDComp = TEXT("DComp");
    const TCHAR* lpcszDebugLayer = TEXT("Dbglayer");
    const TCHAR* lpcszREGKEY = TEXT("SOFTWARE\\Microsoft\\Scrunch\\CodecPack\\MSDVD");
    HKEY hk = NULL;
    DWORD dwData;
    DWORD cbData = sizeof(DWORD);
    DWORD cbType;

    if (0 == RegOpenKeyEx(HKEY_CURRENT_USER, lpcszREGKEY, 0, KEY_READ, &hk))
    {
        if (0 == RegQueryValueEx(hk, lpcszDXSW, 0, &cbType, (LPBYTE)&dwData, &cbData))
            m_DXSWSwitch = dwData;

        dwData = 0;
        cbData = sizeof(DWORD);
        if (0 == RegQueryValueEx(hk, lpcszDebugLayer, 0, &cbType, (LPBYTE)&dwData, &cbData))
            m_useDebugLayer = dwData;
    }

    if (NULL != hk)
        RegCloseKey(hk);

    return;
}

HRESULT solids::lib::mf::sink::video::plain::renderer::CheckDeviceState(BOOL* pbDeviceChanged)
{
    if (pbDeviceChanged == NULL)
        return E_POINTER;

    static int deviceStateChecks = 0;
    static D3D_DRIVER_TYPE driverType = D3D_DRIVER_TYPE_HARDWARE;

    HRESULT hr = SetVideoMonitor(m_hwndVideo);
    if (FAILED(hr))
        return hr;

    ID3D11Device * d3d11_dev = _d3d11_renderer->d3d11_dev();
    if (d3d11_dev != NULL)
    {
        // Lost/hung device. Destroy the device and create a new one.
        if (S_FALSE == hr || (m_DXSWSwitch > 0 && deviceStateChecks == m_DXSWSwitch))
        {
            if (m_DXSWSwitch > 0 && deviceStateChecks == m_DXSWSwitch)
            {
                (driverType == D3D_DRIVER_TYPE_HARDWARE) ? driverType = D3D_DRIVER_TYPE_WARP : driverType = D3D_DRIVER_TYPE_HARDWARE;
            }

            hr = create_dxgi_manager_and_device(driverType);
            if (FAILED(hr))
                return hr;

            *pbDeviceChanged = TRUE;

            _d3d11_renderer->release_d3d11_video_dev();
            _d3d11_renderer->release_d3d11_video_processor_enum();
            _d3d11_renderer->release_d3d11_video_processor();
            _d3d11_renderer->release_swap_chain();

            deviceStateChecks = 0;
        }
        deviceStateChecks++;
    }
    return hr;
}

BOOL solids::lib::mf::sink::video::plain::renderer::check_empty_rect(RECT* pDst)
{
    GetClientRect(m_hwndVideo, pDst);

    return IsRectEmpty(pDst);
}

HRESULT solids::lib::mf::sink::video::plain::renderer::check_shutdown(void) const
{
    if (_is_shutdown)
    {
        return MF_E_SHUTDOWN;
    }
    else
    {
        return S_OK;
    }
}

HRESULT solids::lib::mf::sink::video::plain::renderer::create_dxgi_manager_and_device(D3D_DRIVER_TYPE DriverType)
{
    HRESULT hr = S_OK;
    UINT    resetToken;
    do
    {
        hr = _d3d11_renderer->create_d3d11_dev(m_useDebugLayer);

        if (FAILED(hr))
            break;

        if (m_pDXGIManager == NULL)
        {
            hr = MFCreateDXGIDeviceManager(&resetToken, &m_pDXGIManager);
            if (FAILED(hr))
                break;

            m_DeviceResetToken = resetToken;
        }

        hr = m_pDXGIManager->ResetDevice(_d3d11_renderer->d3d11_dev(), m_DeviceResetToken);
        if (FAILED(hr))
            break;

    } while (FALSE);

    return hr;
}

HRESULT solids::lib::mf::sink::video::plain::renderer::GetVideoDisplayArea(IMFMediaType* pType, MFVideoArea* pArea)
{
    HRESULT hr = S_OK;
    BOOL    bPanScan = FALSE;
    UINT32  uimageWidthInPixels = 0;
    UINT32  uimageHeightInPixels = 0;

    hr = MFGetAttributeSize(pType, MF_MT_FRAME_SIZE, &uimageWidthInPixels, &uimageHeightInPixels);
    if (FAILED(hr))
        return hr;

    int32_t image_width = 0, image_height = 0;
    _d3d11_renderer->get_image_resolution(image_width, image_height);
    if (int32_t(uimageWidthInPixels) != image_width || int32_t(uimageHeightInPixels) != image_height)
    {
        _d3d11_renderer->release_d3d11_video_processor_enum();
        _d3d11_renderer->release_d3d11_video_processor();
        _d3d11_renderer->release_swap_chain();
    }

    image_width = int32_t(uimageWidthInPixels);
    image_height = int32_t(uimageHeightInPixels);
    _d3d11_renderer->set_image_resolution(image_width, image_height);

    bPanScan        = MFGetAttributeUINT32(pType, MF_MT_PAN_SCAN_ENABLED, FALSE);

    // In pan/scan mode, try to get the pan/scan region.
    if (bPanScan)
        hr = pType->GetBlob(MF_MT_PAN_SCAN_APERTURE, (UINT8*)pArea, sizeof(MFVideoArea), NULL);

    // If not in pan/scan mode, or the pan/scan region is not set,
    // get the minimimum display aperture.
    if (!bPanScan || hr == MF_E_ATTRIBUTENOTFOUND)
    {
        hr = pType->GetBlob(MF_MT_MINIMUM_DISPLAY_APERTURE, (UINT8*)pArea, sizeof(MFVideoArea), NULL);
        if (hr == MF_E_ATTRIBUTENOTFOUND)
        {
            // Minimum display aperture is not set.
            // For backward compatibility with some components,
            // check for a geometric aperture.
            hr = pType->GetBlob(MF_MT_GEOMETRIC_APERTURE, (UINT8*)pArea, sizeof(MFVideoArea), NULL);
        }

        // Default: Use the entire video area.
        if (hr == MF_E_ATTRIBUTENOTFOUND)
        {
            *pArea = solids::lib::mf::make_area(0.0, 0.0, image_width, image_height);
            hr = S_OK;
        }
    }
    return hr;
}

void solids::lib::mf::sink::video::plain::renderer::PixelAspectToPictureAspect(
    int Width,
    int Height,
    int PixelAspectX,
    int PixelAspectY,
    int* pPictureAspectX,
    int* pPictureAspectY
)
{
    //
    // sanity check - if any inputs are 0, return 0
    //
    if (PixelAspectX == 0 || PixelAspectY == 0 || Width == 0 || Height == 0)
    {
        *pPictureAspectX = 0;
        *pPictureAspectY = 0;
        return;
    }

    //
    // start by reducing both ratios to lowest terms
    //
    ReduceToLowestTerms(Width, Height, &Width, &Height);
    ReduceToLowestTerms(PixelAspectX, PixelAspectY, &PixelAspectX, &PixelAspectY);

    //
    // Make sure that none of the values are larger than 2^16, so we don't
    // overflow on the last operation.   This reduces the accuracy somewhat,
    // but it's a "hail mary" for incredibly strange aspect ratios that don't
    // exist in practical usage.
    //
    while (Width > 0xFFFF || Height > 0xFFFF)
    {
        Width >>= 1;
        Height >>= 1;
    }

    while (PixelAspectX > 0xFFFF || PixelAspectY > 0xFFFF)
    {
        PixelAspectX >>= 1;
        PixelAspectY >>= 1;
    }

    ReduceToLowestTerms(
        PixelAspectX * Width,
        PixelAspectY * Height,
        pPictureAspectX,
        pPictureAspectY
    );
}

HRESULT solids::lib::mf::sink::video::plain::renderer::process_sample(ID3D11Texture2D * pTexture2D, UINT dwViewIndex, RECT rcDest, UINT32 unInterlaceMode, IMFSample** ppVideoOutFrame)
{
    HRESULT hr = S_OK;
    ID3D11Texture2D *   pOutTexture = NULL;
    IMFSample *         pRTSample   = NULL;
    IMFMediaBuffer *    pBuffer     = NULL;

   enum D3D11_VIDEO_FRAME_FORMAT frameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
    if (MFVideoInterlace_FieldInterleavedUpperFirst == unInterlaceMode || MFVideoInterlace_FieldSingleUpper == unInterlaceMode || MFVideoInterlace_MixedInterlaceOrProgressive == unInterlaceMode)
        frameFormat = D3D11_VIDEO_FRAME_FORMAT_INTERLACED_TOP_FIELD_FIRST;
    else if (MFVideoInterlace_FieldInterleavedLowerFirst == unInterlaceMode || MFVideoInterlace_FieldSingleLower == unInterlaceMode) 
        frameFormat = D3D11_VIDEO_FRAME_FORMAT_INTERLACED_BOTTOM_FIELD_FIRST;

    do
    {
        hr = _d3d11_renderer->process_sample(m_hwndVideo, pTexture2D, dwViewIndex, rcDest, frameFormat, &pOutTexture);
        if (FAILED(hr))
            break;

        _can_process_next_sample = FALSE;

        // create the output media sample
        hr = MFCreateSample(&pRTSample);
        if (FAILED(hr))
            break;

        hr = MFCreateDXGISurfaceBuffer(__uuidof(ID3D11Texture2D), pOutTexture, 0, FALSE, &pBuffer);
        if (FAILED(hr))
            break;

        hr = pRTSample->AddBuffer(pBuffer);
        if (FAILED(hr))
            break;

        if (ppVideoOutFrame != NULL)
        {
            *ppVideoOutFrame = pRTSample;
            (*ppVideoOutFrame)->AddRef();
        }

    } while (0);

    solids::lib::mf::safe_release(pBuffer);
    solids::lib::mf::safe_release(pRTSample);
    solids::lib::mf::safe_release(pOutTexture);

    return hr;
}

//+-------------------------------------------------------------------------
//
//  Function:   ReduceToLowestTerms
//
//  Synopsis:   reduces a numerator and denominator pair to their lowest terms
//
//--------------------------------------------------------------------------

void solids::lib::mf::sink::video::plain::renderer::ReduceToLowestTerms(int NumeratorIn, int DenominatorIn, int * pNumeratorOut, int * pDenominatorOut)
{
    int GCD = gcd(NumeratorIn, DenominatorIn);

    *pNumeratorOut = NumeratorIn / GCD;
    *pDenominatorOut = DenominatorIn / GCD;
}

HRESULT solids::lib::mf::sink::video::plain::renderer::SetMonitor(UINT adapterID)
{
    HRESULT hr = S_OK;
    DWORD dwMatchID = 0;

    solids::lib::mf::auto_lock lock(&_lock);

    do
    {
        hr = m_pMonitors->match_guid(adapterID, &dwMatchID);
        if (FAILED(hr))
        {
            break;
        }

        if (hr == S_FALSE)
        {
            hr = E_INVALIDARG;
            break;
        }

        m_lpCurrMon = &(*m_pMonitors)[dwMatchID];
        m_ConnectionGUID = adapterID;
    } while (FALSE);

    return hr;
}


HRESULT solids::lib::mf::sink::video::plain::renderer::SetVideoMonitor(HWND hwndVideo)
{
    HRESULT hr = S_OK;
    solids::lib::mf::sink::monitor_t * pMonInfo = NULL;
    HMONITOR hMon = NULL;

    if (!m_pMonitors)
    {
        return E_UNEXPECTED;
    }

    hMon = MonitorFromWindow(hwndVideo, MONITOR_DEFAULTTONULL);

    do
    {
        if (NULL != hMon)
        {
            m_pMonitors->terminate_display_system();
            m_lpCurrMon = NULL;

            hr = m_pMonitors->initialize_display_system(hwndVideo);
            if (FAILED(hr))
            {
                break;
            }

            pMonInfo = m_pMonitors->find_monitor(hMon);
            if (NULL != pMonInfo && pMonInfo->id != m_ConnectionGUID)
            {
                hr = SetMonitor(pMonInfo->id);
                if (FAILED(hr))
                {
                    break;
                }
                hr = S_FALSE;
            }
        }
        else
        {
            hr = E_POINTER;
            break;
        }
    } while (FALSE);

    return hr;
}