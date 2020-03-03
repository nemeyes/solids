#include "d3d11_renderer.h"

solids::lib::video::sink::d3d11::plain::renderer::core::core(void)
    : _image_width(0)
    , _image_height(0)
    , _rc_src()
    , _rc_dst()
    , _display_rect()
    , _real_display_width(0)
    , _real_display_height(0)
{
    ::memset(&_rc_src, 0x00, sizeof(_rc_src));
    ::memset(&_rc_dst, 0x00, sizeof(_rc_dst));
}

solids::lib::video::sink::d3d11::plain::renderer::core::~core(void)
{

}

BOOL solids::lib::video::sink::d3d11::plain::renderer::core::is_initialized(void)
{
	return TRUE;
}

int32_t solids::lib::video::sink::d3d11::plain::renderer::core::initialize(void)
{


	return solids::lib::video::sink::d3d11::plain::renderer::err_code_t::success;
}

int32_t solids::lib::video::sink::d3d11::plain::renderer::core::release(void)
{

	return solids::lib::video::sink::d3d11::plain::renderer::err_code_t::success;
}

BOOL solids::lib::video::sink::d3d11::plain::renderer::core::present(void)
{
    HRESULT hr = _swap_chain->Present(0, 0);
    if (SUCCEEDED(hr))
        return TRUE;
    else
        return FALSE;
}

ID3D11Device * solids::lib::video::sink::d3d11::plain::renderer::core::d3d11_dev(void)
{
    return _d3d11_dev;
}

void solids::lib::video::sink::d3d11::plain::renderer::core::set_fullscreen(BOOL fs)
{
    _fullscreen = fs;
}

BOOL solids::lib::video::sink::d3d11::plain::renderer::core::get_fullscreen(void)
{
    return _fullscreen;
}

void solids::lib::video::sink::d3d11::plain::renderer::core::set_image_resolution(int32_t width, int32_t height)
{
    _image_width = width;
    _image_height = height;
}

void solids::lib::video::sink::d3d11::plain::renderer::core::get_image_resolution(int32_t& width, int32_t& height)
{
    width = _image_width;
    height = _image_height;
}

void solids::lib::video::sink::d3d11::plain::renderer::core::set_display_rect(RECT display_rect)
{
    _display_rect = display_rect;
}

void solids::lib::video::sink::d3d11::plain::renderer::core::set_real_display_resolution(int32_t width, int32_t height)
{
    _real_display_width = width;
    _real_display_height = height;
}

HRESULT solids::lib::video::sink::d3d11::plain::renderer::core::create_d3d11_dev(int32_t useDebugLayer)
{
    HRESULT hr = S_OK;
    IDXGIOutput *       pDXGIOutput = NULL;
    IDXGIAdapter *      pTempAdapter = NULL;
    ID3D10Multithread * pMultiThread = NULL;
    IDXGIDevice1 *      pDXGIDev = NULL;
    IDXGIAdapter1 *     pAdapter = NULL;

    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    D3D_FEATURE_LEVEL featureLevel;

    solids::lib::safe_release(_d3d11_dev);
    
    for (DWORD dwCount = 0; dwCount < ARRAYSIZE(featureLevels); dwCount++)
    {
        hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, useDebugLayer, featureLevels, ARRAYSIZE(featureLevels), D3D11_SDK_VERSION, &_d3d11_dev, &featureLevel, NULL);
        if (SUCCEEDED(hr))
        {
            ID3D11VideoDevice* pD3D11VideoDevice = NULL;
            hr = _d3d11_dev->QueryInterface(__uuidof(ID3D11VideoDevice), (void**)&pD3D11VideoDevice);
            solids::lib::safe_release(pD3D11VideoDevice);
            if (SUCCEEDED(hr))
            {
                solids::lib::safe_release(_d3d11_dev_ctx);
                break;
            }
            solids::lib::safe_release(_d3d11_dev);
        }
    }

    do
    {
        if (!_d3d11_dev)
            break;

        _d3d11_dev->GetImmediateContext(&_d3d11_dev_ctx);
        hr = _d3d11_dev_ctx->QueryInterface(__uuidof(ID3D10Multithread), (void**)&pMultiThread);
        if (FAILED(hr))
        {
            solids::lib::safe_release(_d3d11_dev_ctx);
            solids::lib::safe_release(_d3d11_dev);
            break;
        }

        pMultiThread->SetMultithreadProtected(TRUE);
        hr = _d3d11_dev->QueryInterface(__uuidof(IDXGIDevice1), (LPVOID*)&pDXGIDev);
        if (FAILED(hr))
        {
            solids::lib::safe_release(_d3d11_dev_ctx);
            solids::lib::safe_release(_d3d11_dev);
            break;
        }

        hr = pDXGIDev->GetAdapter(&pTempAdapter);
        if (FAILED(hr))
        {
            solids::lib::safe_release(_d3d11_dev_ctx);
            solids::lib::safe_release(_d3d11_dev);
            break;
        }

        hr = pTempAdapter->QueryInterface(__uuidof(IDXGIAdapter1), (LPVOID*)&pAdapter);
        if (FAILED(hr))
        {
            solids::lib::safe_release(_d3d11_dev_ctx);
            solids::lib::safe_release(_d3d11_dev);
            break;
        }

        solids::lib::safe_release(_dxgi_factory);
        hr = pAdapter->GetParent(__uuidof(IDXGIFactory2), (LPVOID*)&_dxgi_factory);
        if (FAILED(hr))
        {
            solids::lib::safe_release(_d3d11_dev_ctx);
            solids::lib::safe_release(_d3d11_dev);
            break;
        }

        hr = pAdapter->EnumOutputs(0, &pDXGIOutput);
        if (FAILED(hr))
        {
            solids::lib::safe_release(_dxgi_factory);
            solids::lib::safe_release(_d3d11_dev_ctx);
            solids::lib::safe_release(_d3d11_dev);
            break;
        }

        solids::lib::safe_release(_dxgi_output);
        hr = pDXGIOutput->QueryInterface(__uuidof(IDXGIOutput1), (LPVOID*)&_dxgi_output);
        if (FAILED(hr))
        {
            solids::lib::safe_release(_dxgi_factory);
            solids::lib::safe_release(_d3d11_dev_ctx);
            solids::lib::safe_release(_d3d11_dev);
            break;
        }

    } while (0);

    solids::lib::safe_release(pTempAdapter);
    solids::lib::safe_release(pMultiThread);
    solids::lib::safe_release(pDXGIDev);
    solids::lib::safe_release(pAdapter);
    solids::lib::safe_release(pDXGIOutput);

    return hr;
}

HRESULT	solids::lib::video::sink::d3d11::plain::renderer::core::process_sample(HWND hwnd, ID3D11Texture2D * input, int32_t vi, RECT rcDst, D3D11_VIDEO_FRAME_FORMAT interlace, ID3D11Texture2D ** output)
{
    HRESULT hr = S_OK;
    ID3D11VideoContext *                pVideoContext   = NULL;
    ID3D11VideoProcessorInputView *     pInputView      = NULL;
    ID3D11VideoProcessorOutputView *    pOutputView     = NULL;
    ID3D11Texture2D *                   pDXGIBackBuffer = NULL;
    D3D11_VIDEO_PROCESSOR_CAPS          vpCaps          = { 0 };

    do
    {
        if (!_d3d11_video_dev)
        {
            hr = _d3d11_dev->QueryInterface(__uuidof(ID3D11VideoDevice), (void**)&_d3d11_video_dev);
            if (FAILED(hr))
                break;
        }

        hr = _d3d11_dev_ctx->QueryInterface(__uuidof(ID3D11VideoContext), (void**)&pVideoContext);
        if (FAILED(hr))
            break;

        RECT targetRectOld = _rc_dst;
        RECT sourceRectOld = _rc_src;

        update_rectangles(&targetRectOld, &sourceRectOld);

        _rc_dst = rcDst;

        D3D11_TEXTURE2D_DESC textureDesc;
        input->GetDesc(&textureDesc);

        if (!_d3d11_video_processor_enum || !_d3d11_video_processor || _image_width != textureDesc.Width || _image_height != textureDesc.Height)
        {
            solids::lib::safe_release(_d3d11_video_processor_enum);
            solids::lib::safe_release(_d3d11_video_processor);

            _image_width    = textureDesc.Width;
            _image_height   = textureDesc.Height;

            D3D11_VIDEO_PROCESSOR_CONTENT_DESC contentDesc;
            ::memset(&contentDesc, 0x00, sizeof(contentDesc));
            contentDesc.InputFrameFormat    = interlace;// D3D11_VIDEO_FRAME_FORMAT_INTERLACED_TOP_FIELD_FIRST;
            contentDesc.InputWidth          = textureDesc.Width;
            contentDesc.InputHeight         = textureDesc.Height;
            contentDesc.OutputWidth         = textureDesc.Width;
            contentDesc.OutputHeight        = textureDesc.Height;
            contentDesc.Usage               = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;

            hr = _d3d11_video_dev->CreateVideoProcessorEnumerator(&contentDesc, &_d3d11_video_processor_enum); 
            if (FAILED(hr))
                break;

            UINT flags;
            DXGI_FORMAT vp_output_format = DXGI_FORMAT_B8G8R8A8_UNORM;
            hr = _d3d11_video_processor_enum->CheckVideoProcessorFormat(vp_output_format, &flags);
            if (FAILED(hr) || (flags & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT) == 0)
            {
                hr = E_FAIL;
                break;
            }

            _rc_src.left    = 0;
            _rc_src.top     = 0;
            _rc_src.right   = _real_display_width;
            _rc_src.bottom  = _real_display_height;

            DWORD index;
            hr = find_bob_processor_index(&index);
            if (FAILED(hr))
                break;

            hr = _d3d11_video_dev->CreateVideoProcessor(_d3d11_video_processor_enum, index, &_d3d11_video_processor);
            if (FAILED(hr))
                break;
        }

        RECT targetRect = _rc_dst;
        RECT sourceRect = _rc_src;
        update_rectangles(&targetRect, &sourceRect);

        const BOOL bDstRectChanged = !::EqualRect(&targetRect, &targetRectOld);
        if (!_swap_chain || bDstRectChanged)
        {
            hr = update_swap_chain(_rc_dst.right, _rc_dst.bottom, hwnd);
            if (FAILED(hr))
                break;
        }

        hr = _swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&pDXGIBackBuffer);
        if (FAILED(hr))
            break;

        // Create Output View of Output Surfaces.
        D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC OutputViewDesc;
        ZeroMemory(&OutputViewDesc, sizeof(OutputViewDesc));
        OutputViewDesc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
        OutputViewDesc.Texture2D.MipSlice = 0;
        OutputViewDesc.Texture2DArray.MipSlice = 0;
        OutputViewDesc.Texture2DArray.FirstArraySlice = 0;

        hr = _d3d11_video_dev->CreateVideoProcessorOutputView(pDXGIBackBuffer, _d3d11_video_processor_enum, &OutputViewDesc, &pOutputView);
        if (FAILED(hr))
        {
            solids::lib::safe_release(pDXGIBackBuffer);
            break;
        }

        D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC inputViewDesc;
        ZeroMemory(&inputViewDesc, sizeof(inputViewDesc));
        inputViewDesc.FourCC = 0;
        inputViewDesc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
        inputViewDesc.Texture2D.MipSlice = 0;
        inputViewDesc.Texture2D.ArraySlice = vi;
        hr = _d3d11_video_dev->CreateVideoProcessorInputView(input, _d3d11_video_processor_enum, &inputViewDesc, &pInputView);
        if (FAILED(hr))
        {
            solids::lib::safe_release(pDXGIBackBuffer);
            break;
        }

        set_video_context_parameters(pVideoContext, &sourceRect, &targetRect, &_rc_dst, interlace);

        D3D11_VIDEO_PROCESSOR_STREAM sd;
        ZeroMemory(&sd, sizeof(sd));
        sd.Enable = TRUE;
        sd.OutputIndex = 0;
        sd.InputFrameOrField = 0;
        sd.PastFrames = 0;
        sd.FutureFrames = 0;
        sd.ppPastSurfaces = NULL;
        sd.ppFutureSurfaces = NULL;
        sd.pInputSurface = pInputView;
        sd.ppPastSurfacesRight = NULL;
        sd.ppFutureSurfacesRight = NULL;
        hr = pVideoContext->VideoProcessorBlt(_d3d11_video_processor, pOutputView, 0, 1, &sd);
        if (FAILED(hr))
        {
            solids::lib::safe_release(pDXGIBackBuffer);
            break;
        }

        *output = pDXGIBackBuffer;


    } while (0);

    solids::lib::safe_release(pOutputView);
    solids::lib::safe_release(pInputView);
    solids::lib::safe_release(pVideoContext);

    return hr;
}

void solids::lib::video::sink::d3d11::plain::renderer::core::release_d3d11_dev(void)
{
    solids::lib::safe_release(_d3d11_dev);
}

void solids::lib::video::sink::d3d11::plain::renderer::core::release_d3d11_video_dev(void)
{
	solids::lib::safe_release(_d3d11_video_dev);
}

void solids::lib::video::sink::d3d11::plain::renderer::core::release_d3d11_video_processor_enum(void)
{
    solids::lib::safe_release(_d3d11_video_processor_enum);
}

void solids::lib::video::sink::d3d11::plain::renderer::core::release_d3d11_video_processor(void)
{
    solids::lib::safe_release(_d3d11_video_processor);
}

void solids::lib::video::sink::d3d11::plain::renderer::core::release_swap_chain(void)
{
    solids::lib::safe_release(_swap_chain);
}

BOOL solids::lib::video::sink::d3d11::plain::renderer::core::is_media_type_supported(int32_t input_width, int32_t input_height, int32_t output_width, int32_t output_height, int32_t input_num_fps, int32_t input_den_fps, int32_t output_num_fps, int32_t output_den_fps, DXGI_FORMAT dxgi_format)
{
    //Check if the format is supported
    HRESULT hr = E_FAIL;
    do
    {
        if (!_d3d11_video_dev)
        {
            hr = _d3d11_dev->QueryInterface(__uuidof(ID3D11VideoDevice), (void**)&_d3d11_video_dev);
            if (FAILED(hr))
            {
                break;
            }
        }

        D3D11_VIDEO_PROCESSOR_CONTENT_DESC ContentDesc;
        ZeroMemory(&ContentDesc, sizeof(ContentDesc));
        ContentDesc.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_INTERLACED_TOP_FIELD_FIRST;
        ContentDesc.InputWidth = (DWORD)input_width;
        ContentDesc.InputHeight = (DWORD)input_height;
        ContentDesc.OutputWidth = (DWORD)output_width;
        ContentDesc.OutputHeight = (DWORD)output_height;
        ContentDesc.InputFrameRate.Numerator = input_num_fps;
        ContentDesc.InputFrameRate.Denominator = input_den_fps;
        ContentDesc.OutputFrameRate.Numerator = output_num_fps;
        ContentDesc.OutputFrameRate.Denominator = output_den_fps;
        ContentDesc.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;

        solids::lib::safe_release(_d3d11_video_processor_enum);
        hr = _d3d11_video_dev->CreateVideoProcessorEnumerator(&ContentDesc, &_d3d11_video_processor_enum);
        if (FAILED(hr))
        {
            break;
        }

        UINT uiFlags;
        hr = _d3d11_video_processor_enum->CheckVideoProcessorFormat(dxgi_format, &uiFlags);
        if (FAILED(hr) || 0 == (uiFlags & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_INPUT))
        {
            break;
        }
    } while (FALSE);

    if (SUCCEEDED(hr))
        return TRUE;
    else
        return FALSE;
}

BOOL solids::lib::video::sink::d3d11::plain::renderer::core::check_swap_chain(void)
{
    if (_swap_chain == NULL)
        return FALSE;
    else
        return TRUE;
}


/////////////private////////////////
HRESULT solids::lib::video::sink::d3d11::plain::renderer::core::update_swap_chain(int32_t width, int32_t height, HWND hwnd)
{
    HRESULT hr = S_OK;

    // Get the DXGISwapChain1
    DXGI_SWAP_CHAIN_DESC1 scd;
    ZeroMemory(&scd, sizeof(scd));
    scd.SampleDesc.Count = 1;
    scd.SampleDesc.Quality = 0;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    scd.Scaling = DXGI_SCALING_STRETCH;
    scd.Width = width;
    scd.Height = height;
    scd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    scd.Stereo = FALSE;
    scd.BufferUsage = DXGI_USAGE_BACK_BUFFER | DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.Flags = 0; //opt in to do direct flip;
    scd.BufferCount = 4;

    do
    {
        if (_swap_chain)
        {
            // Resize our back buffers for the desired format.
            hr = _swap_chain->ResizeBuffers(4, width, height, scd.Format, scd.Flags);
            break;
        }

        hr = _dxgi_factory->CreateSwapChainForHwnd(_d3d11_dev, hwnd, &scd, NULL, NULL, &_swap_chain);
        if (FAILED(hr))
        {
            break;
        }

        if (_fullscreen)
        {
            hr = _swap_chain->SetFullscreenState(TRUE, NULL);
            if (FAILED(hr))
                break;
        }
        else
        {
            hr = _swap_chain->SetFullscreenState(FALSE, NULL);
            if (FAILED(hr))
                break;
        }
    } while (FALSE);

    return hr;
}

void solids::lib::video::sink::d3d11::plain::renderer::core::update_rectangles(RECT* dst, RECT* src)
{
    RECT rcSrc = *src;

    src->left = _display_rect.left + MulDiv(src->left, (_display_rect.right - _display_rect.left), _real_display_width);
    src->right = _display_rect.left + MulDiv(src->right, (_display_rect.right - _display_rect.left), _real_display_width);

    src->top = _display_rect.top + MulDiv(src->top, (_display_rect.bottom - _display_rect.top), _real_display_height);
    src->bottom = _display_rect.top + MulDiv(src->bottom, (_display_rect.bottom - _display_rect.top), _real_display_height);

    letter_box_dst_rect(dst, rcSrc, _rc_dst);
}

void solids::lib::video::sink::d3d11::plain::renderer::core::letter_box_dst_rect(LPRECT lprcLBDst, const RECT& src, const RECT& dst)
{
    int iSrcWidth = src.right - src.left;
    int iSrcHeight = src.bottom - src.top;

    int iDstWidth = dst.right - dst.left;
    int iDstHeight = dst.bottom - dst.top;

    int iDstLBWidth = 0;
    int iDstLBHeight = 0;

    if (MulDiv(iSrcWidth, iDstHeight, iSrcHeight) <= iDstWidth)
    {
        iDstLBWidth = MulDiv(iDstHeight, iSrcWidth, iSrcHeight);
        iDstLBHeight = iDstHeight;
    }
    else
    {
        iDstLBWidth = iDstWidth;
        iDstLBHeight = MulDiv(iDstWidth, iSrcHeight, iSrcWidth);
    }

    lprcLBDst->left = dst.left + ((iDstWidth - iDstLBWidth) / 2);
    lprcLBDst->right = lprcLBDst->left + iDstLBWidth;

    lprcLBDst->top = dst.top + ((iDstHeight - iDstLBHeight) / 2);
    lprcLBDst->bottom = lprcLBDst->top + iDstLBHeight;
}

HRESULT	solids::lib::video::sink::d3d11::plain::renderer::core::find_bob_processor_index(DWORD* index)
{
    HRESULT hr = S_OK;
    D3D11_VIDEO_PROCESSOR_CAPS caps = {};
    D3D11_VIDEO_PROCESSOR_RATE_CONVERSION_CAPS convCaps = {};

    *index = 0;
    hr = _d3d11_video_processor_enum->GetVideoProcessorCaps(&caps);
    if (FAILED(hr))
        return hr;

    for (DWORD i = 0; i < caps.RateConversionCapsCount; i++)
    {
        hr = _d3d11_video_processor_enum->GetVideoProcessorRateConversionCaps(i, &convCaps);
        if (FAILED(hr))
            return hr;

        // Check the caps to see which deinterlacer is supported
        if ((convCaps.ProcessorCaps & D3D11_VIDEO_PROCESSOR_PROCESSOR_CAPS_DEINTERLACE_BOB) != 0)
        {
            *index = i;
            return hr;
        }
    }
    return E_FAIL;
}

void solids::lib::video::sink::d3d11::plain::renderer::core::set_video_context_parameters(ID3D11VideoContext* vc, const RECT* src, const RECT* dst, const RECT* target, D3D11_VIDEO_FRAME_FORMAT interlace)
{
    D3D11_VIDEO_FRAME_FORMAT FrameFormat = interlace;

    // input format
    vc->VideoProcessorSetStreamFrameFormat(_d3d11_video_processor, 0, FrameFormat);
    // Output rate (repeat frames)
    vc->VideoProcessorSetStreamOutputRate(_d3d11_video_processor, 0, D3D11_VIDEO_PROCESSOR_OUTPUT_RATE_NORMAL, TRUE, NULL);
    // Source rect
    vc->VideoProcessorSetStreamSourceRect(_d3d11_video_processor, 0, TRUE, src);
    // Stream dest rect
    vc->VideoProcessorSetStreamDestRect(_d3d11_video_processor, 0, TRUE, dst);
    vc->VideoProcessorSetOutputTargetRect(_d3d11_video_processor, TRUE, target);

    // Stream color space
    D3D11_VIDEO_PROCESSOR_COLOR_SPACE colorSpace = {};
    colorSpace.YCbCr_xvYCC = 1;
    vc->VideoProcessorSetStreamColorSpace(_d3d11_video_processor, 0, &colorSpace);
    // Output color space
    vc->VideoProcessorSetOutputColorSpace(_d3d11_video_processor, &colorSpace);

    // Output background color (black)
    D3D11_VIDEO_COLOR backgroundColor = {};
    backgroundColor.RGBA.A = 1.0F;
    backgroundColor.RGBA.R = 1.0F * static_cast<float>(GetRValue(0)) / 255.0F;
    backgroundColor.RGBA.G = 1.0F * static_cast<float>(GetGValue(0)) / 255.0F;
    backgroundColor.RGBA.B = 1.0F * static_cast<float>(GetBValue(0)) / 255.0F;

    vc->VideoProcessorSetOutputBackgroundColor(_d3d11_video_processor, FALSE, &backgroundColor);
}