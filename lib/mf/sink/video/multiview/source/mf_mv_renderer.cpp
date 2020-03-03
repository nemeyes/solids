#include "mf_mv_renderer.h"
#include "mf_mv_media_sink.h"
//-------------------------------------------------------------------
// CPresenter constructor.
//-------------------------------------------------------------------
sld::lib::mf::sink::video::multiview::renderer::renderer(sld::lib::mf::sink::video::multiview::media * media_sink)
	: _lock()
	, _is_shutdown(FALSE)
	, _dxgi_factory2(NULL)
	, _d3d11_device(NULL)
	, _d3d11_immediate_context(NULL)
	, _dxgi_manager(NULL)
	, _dxgi_output1(NULL)
	, _sample_allocator_ex(NULL)
	, _dcomp_device(NULL)
	, _hwnd_target(NULL)
	, _root_visual(NULL)
	, _hwnd(NULL)
	, _monitors(NULL)
	, _current_monitor(NULL)
	, _device_reset_token(0)
	, _dx_sw_switch(0)
	, _use_dcomp_visual(0)
	, _use_debug_layer(D3D11_CREATE_DEVICE_VIDEO_SUPPORT | D3D11_CREATE_DEVICE_BGRA_SUPPORT /*| D3D11_CREATE_DEVICE_DEBUG*/)
	, _video_device(NULL)
	, _swap_chain1(NULL)
	, _is_full_screen(FALSE)
	, _can_process_next_sample(TRUE)
	, _ms(media_sink)
	, _current_time(0)
	, _d3d11_video_context(NULL)
	, _view_info(NULL)
	, _d3d_renderer(NULL)
	, _enable_coordinated_cs_converter(FALSE)
	, _view_count(4)
	, _first_sample(TRUE)
	, _selected(0)
	, _maximize(false)
	, _render_type(0)		// 0 : stretch, 1 : original
{
	ZeroMemory(&_output_view_desc, sizeof(_output_view_desc));
	_output_view_desc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
	_output_view_desc.Texture2D.MipSlice = 0;
	_output_view_desc.Texture2DArray.MipSlice = 0;
	_output_view_desc.Texture2DArray.FirstArraySlice = 0;

	ZeroMemory(&_input_view_desc, sizeof(_input_view_desc));
	_input_view_desc.FourCC = 0;
	_input_view_desc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
	_input_view_desc.Texture2D.MipSlice = 0;
	_input_view_desc.Texture2D.ArraySlice = 0;

	ZeroMemory(&_stream_data, sizeof(_stream_data));
	_stream_data.Enable = TRUE;
	_stream_data.OutputIndex = 0;
	_stream_data.InputFrameOrField = 0;
	_stream_data.PastFrames = 0;
	_stream_data.FutureFrames = 0;
	_stream_data.ppPastSurfaces = NULL;
	_stream_data.ppFutureSurfaces = NULL;
	_stream_data.ppPastSurfacesRight = NULL;
	_stream_data.ppFutureSurfacesRight = NULL;

}

sld::lib::mf::sink::video::multiview::renderer::~renderer(void)
{
	if (_view_info)
	{
		for (int i = 0; i < _view_count; i++)
		{
			sld::lib::mf::safe_release(_view_info[i].buffer);
			sld::lib::mf::safe_release(_view_info[i].video_processor_enum);
			sld::lib::mf::safe_release(_view_info[i].video_processor);
			sld::lib::mf::safe_release(_view_info[i].shader_resource_view);
		}
		delete[] _view_info;
		_view_info = NULL;
	}
	if (_d3d_renderer)
	{
		delete _d3d_renderer;
		_d3d_renderer = NULL;
	}
	sld::lib::mf::safe_release(_d3d11_video_context);
    safe_delete(_monitors);
}

//IUnknown
ULONG sld::lib::mf::sink::video::multiview::renderer::AddRef(void)
{
	return sld::lib::mf::refcount_object::AddRef();
}

//IUnknown
ULONG sld::lib::mf::sink::video::multiview::renderer::Release(void)
{
	return sld::lib::mf::refcount_object::Release();
}
//IUnknown
HRESULT sld::lib::mf::sink::video::multiview::renderer::QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void** ppv)
{
    if (!ppv)
        return E_POINTER;
    if (iid == IID_IUnknown)
        *ppv = static_cast<IUnknown*>(static_cast<IMFVideoDisplayControl*>(this));
    else if (iid == __uuidof(IMFVideoDisplayControl))
        *ppv = static_cast<IMFVideoDisplayControl*>(this);
    else if (iid == __uuidof(IMFGetService))
        *ppv = static_cast<IMFGetService*>(this);
    else
    {
        *ppv = NULL;
        return E_NOINTERFACE;
    }
    AddRef();
    return S_OK;
}


// IMFVideoDisplayControl
HRESULT sld::lib::mf::sink::video::multiview::renderer::GetFullscreen(__RPC__out BOOL* pfFullscreen)
{
	sld::lib::mf::auto_lock lock(&_lock);

    HRESULT hr = check_shutdown();
    if (FAILED(hr))
        return hr;

    if (pfFullscreen == NULL)
        return E_POINTER;

    *pfFullscreen = _is_full_screen;

    return S_OK;
}

// IMFVideoDisplayControl
HRESULT sld::lib::mf::sink::video::multiview::renderer::SetFullscreen(BOOL fFullscreen)
{
	sld::lib::mf::auto_lock lock(&_lock);

    HRESULT hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        _is_full_screen = fFullscreen;
		sld::lib::mf::safe_release(_video_device);
    }
    return hr;
}

// IMFVideoDisplayControl
HRESULT sld::lib::mf::sink::video::multiview::renderer::SetVideoWindow(__RPC__in HWND hwndVideo)
{
    HRESULT hr = S_OK;
	sld::lib::mf::auto_lock lock(&_lock);

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

        _monitors = new CMonitorArray();
        if (!_monitors)
        {
            hr = E_OUTOFMEMORY;
            break;
        }

        hr = set_video_monitor(hwndVideo);
        if (FAILED(hr)) break;

		check_decode_switch_regkey();
        _hwnd = hwndVideo;
		
        hr = create_dxgi_manager_and_device();
		if (FAILED(hr)) break;
    }
    while(FALSE);

    return hr;
}

HRESULT sld::lib::mf::sink::video::multiview::renderer::GetService(__RPC__in REFGUID guidService, __RPC__in REFIID riid, __RPC__deref_out_opt LPVOID* ppvObject)
{
    HRESULT hr = S_OK;

    if (guidService == MR_VIDEO_ACCELERATION_SERVICE)
    {
        if (riid == __uuidof(IMFDXGIDeviceManager))
        {
            if (NULL != _dxgi_manager)
            {
                *ppvObject = (void*) static_cast<IUnknown*>(_dxgi_manager);
                ((IUnknown*) *ppvObject)->AddRef();
            }
            else
            {
                hr = E_NOINTERFACE;
            }
        }
        else if (riid == __uuidof(IMFVideoSampleAllocatorEx))
        {
            if (NULL == _sample_allocator_ex)
            {
                hr = MFCreateVideoSampleAllocatorEx(IID_IMFVideoSampleAllocatorEx, (LPVOID*)&_sample_allocator_ex);
                if (SUCCEEDED(hr) && NULL != _dxgi_manager)
                {
                    hr = _sample_allocator_ex->SetDirectXManager(_dxgi_manager);
                }
            }
            if (SUCCEEDED(hr))
            {
                hr = _sample_allocator_ex->QueryInterface(riid, ppvObject);
            }
        }
        else
        {
            hr = E_NOINTERFACE;
        }
    }
    else if (guidService == MR_VIDEO_RENDER_SERVICE)
        hr = QueryInterface(riid, ppvObject);
    else
        hr = MF_E_UNSUPPORTED_SERVICE;

    return hr;
}

BOOL sld::lib::mf::sink::video::multiview::renderer::can_process_next_sample(void)
{
    return _can_process_next_sample;
}

HRESULT sld::lib::mf::sink::video::multiview::renderer::flush(void)
{
	sld::lib::mf::auto_lock lock(&_lock);

    HRESULT hr = check_shutdown();

    _can_process_next_sample = TRUE;

    return hr;
}

HRESULT sld::lib::mf::sink::video::multiview::renderer::get_monitor_refresh_rate(DWORD * refresh_rate)
{
	if (!refresh_rate)
		return E_POINTER;

	if (!_current_monitor)
		return MF_E_INVALIDREQUEST;

	*refresh_rate = _current_monitor->dwRefreshRate;
	return S_OK;
}

HRESULT sld::lib::mf::sink::video::multiview::renderer::is_media_type_supported(IMFMediaType* pMediaType, DXGI_FORMAT dxgiFormat)
{
    HRESULT hr = S_OK;
    do
    {
        hr = check_shutdown();
		if (FAILED(hr)) break;

        if (pMediaType == NULL)
        {
            hr = E_POINTER;
            break;
        }

        if (!_video_device)
        {
            hr = _d3d11_device->QueryInterface(__uuidof(ID3D11VideoDevice), (void**)&_video_device);
            if (FAILED(hr)) break;
        }
    } while (FALSE);

    return hr;
}

HRESULT sld::lib::mf::sink::video::multiview::renderer::render_samples(IMFSample* sample)
{
    HRESULT hr = S_OK;
	sld::lib::mf::auto_lock lock(&_lock);

    do
    {
        hr = check_shutdown();
        if (FAILED(hr)) break;

        if (NULL == _swap_chain1) break;

        RECT rcDest;
        ZeroMemory(&rcDest, sizeof(rcDest));
        if (check_empty_rect(&rcDest))
        {
            hr = S_OK;
            break;
        }

		hr = _swap_chain1->Present(0, 0);
		if (FAILED(hr)) 
			break;
        _can_process_next_sample = TRUE;

    } while (FALSE);
    return hr;
}

HRESULT sld::lib::mf::sink::video::multiview::renderer::process_samples(IMFMediaType * mt, IMFSample * sample, UINT32 * unInterlace_mode, BOOL * device_changed, BOOL * process_again, IMFSample ** osample)
{
	*process_again = FALSE;
	HRESULT hr = S_OK;

	IMFMediaBuffer * mf_buffer = NULL;
	ID3D11Texture2D * backBuffer = NULL;

	ID3D11Texture2D * texture_2d = NULL;
	IMFDXGIBuffer * dxgi_buffer = NULL;
	ID3D11Device * device_input = NULL;
	UINT resource_index = 0;

	sld::lib::mf::auto_lock lock(&_lock);
	do
	{
		_can_process_next_sample = FALSE;

		hr = check_shutdown();
		if (FAILED(hr)) break;

		if (unInterlace_mode == NULL || mt == NULL || sample == NULL || device_changed == NULL || process_again == NULL)
		{
			hr = E_POINTER;
			break;
		}

		*device_changed = FALSE;

		hr = check_device_state(device_changed);
		if (FAILED(hr)) break;

		MFVideoInterlaceMode unInterlaceMode = (MFVideoInterlaceMode)MFGetAttributeUINT32(mt, MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
		if (MFVideoInterlace_MixedInterlaceOrProgressive == unInterlaceMode)
		{
			BOOL fInterlaced = MFGetAttributeUINT32(sample, MFSampleExtension_Interlaced, FALSE);
			if (!fInterlaced)
			{
				*unInterlace_mode = MFVideoInterlace_Progressive;                 // Progressive sample
			}
			else
			{
				BOOL fBottomFirst = MFGetAttributeUINT32(sample, MFSampleExtension_BottomFieldFirst, FALSE);
				if (fBottomFirst)
				{
					*unInterlace_mode = MFVideoInterlace_FieldInterleavedLowerFirst;
				}
				else
				{
					*unInterlace_mode = MFVideoInterlace_FieldInterleavedUpperFirst;
				}
			}
		}

		if (_first_sample == TRUE)
		{
			_first_sample = FALSE;

			update_dxgi_swap_chain();
		
			double * ratio_array = NULL;
			UINT32 size;
			HRESULT check = sample->GetAllocatedBlob(MF_MT_ACTIVE_IMAGE_RATIO, (UINT8**)&ratio_array, /*sizeof(double) * 2 * _view_count*/&size);
			
			if (SUCCEEDED(check))
			{
				for (INT view_index = 0; view_index < _view_count; view_index++)
				{
					hr = sample->GetBufferByIndex(view_index, &mf_buffer);
					if (FAILED(hr)) break;
					if (mf_buffer)
					{
						hr = mf_buffer->QueryInterface(__uuidof(IMFDXGIBuffer), (LPVOID*)&dxgi_buffer);
						if (FAILED(hr)) break;

						hr = dxgi_buffer->GetResource(__uuidof(ID3D11Texture2D), (LPVOID*)&texture_2d);
						if (FAILED(hr)) break;


						D3D11_TEXTURE2D_DESC src_desc;
						texture_2d->GetDesc(&src_desc);

						_view_info[view_index].src_width = src_desc.Width;
						_view_info[view_index].src_height = src_desc.Height;

						if (_enable_coordinated_cs_converter)
						{
							_view_info[view_index].dst_width = _vcodec_width / 2;
							_view_info[view_index].dst_height = _vcodec_height / 2;
						}
						else
						{
							_view_info[view_index].dst_width = _view_info[view_index].src_width;
							_view_info[view_index].dst_height = _view_info[view_index].src_height;
						}
						//if (view_index == 3)
						//	*unInterlace_mode = MFVideoInterlace_FieldInterleavedUpperFirst;
						create_video_processor(view_index, texture_2d, *unInterlace_mode);
						if (FAILED(hr)) break;

						sld::lib::mf::safe_release(mf_buffer);
						sld::lib::mf::safe_release(dxgi_buffer);
						sld::lib::mf::safe_release(texture_2d);
					}

					_view_info[view_index].active_video_ratio[0] = ratio_array[2 * view_index];
					_view_info[view_index].active_video_ratio[1] = ratio_array[2 * view_index + 1];
				}
				if (FAILED(hr)) break;
			}

			if (_d3d_renderer)
			{
				_d3d_renderer->initialize(&_d3d_render_ctx);
			}
			CoTaskMemFree(ratio_array);
		}

		
		DWORD count = 0;
		sample->GetBufferCount(&count);
		/*if ((_maximize == false) && (count != _view_count))
		{
			return S_OK;
		}*/
		MFCreateSample(osample);
		if ((_maximize == false) && count == _view_count )
		{
			for (INT view_index = 0; view_index < _view_count; view_index++)
			{
				hr = sample->GetBufferByIndex(view_index, &mf_buffer);
				if (FAILED(hr)) break;
				if (mf_buffer)
				{
					hr = mf_buffer->QueryInterface(__uuidof(IMFDXGIBuffer), (LPVOID*)&dxgi_buffer);
					if (FAILED(hr)) break;

					hr = dxgi_buffer->GetResource(__uuidof(ID3D11Texture2D), (LPVOID*)&texture_2d);
					if (FAILED(hr)) break;

					hr = dxgi_buffer->GetSubresourceIndex(&resource_index);
					if (FAILED(hr)) break;

					hr = colorspace_convert(view_index, texture_2d, resource_index, *unInterlace_mode);
					if (FAILED(hr)) break;

					sld::lib::mf::safe_release(mf_buffer);
					sld::lib::mf::safe_release(dxgi_buffer);
					sld::lib::mf::safe_release(texture_2d);
				}
			}
			if (FAILED(hr)) break;
		}
		else
		{
			hr = sample->GetBufferByIndex(0, &mf_buffer);
			if (FAILED(hr)) break;
			if (mf_buffer)
			{
				hr = mf_buffer->QueryInterface(__uuidof(IMFDXGIBuffer), (LPVOID*)&dxgi_buffer);
				if (FAILED(hr)) break;

				hr = dxgi_buffer->GetResource(__uuidof(ID3D11Texture2D), (LPVOID*)&texture_2d);
				if (FAILED(hr)) break;

				hr = dxgi_buffer->GetSubresourceIndex(&resource_index);
				if (FAILED(hr)) break;

				hr = colorspace_convert(_selected, texture_2d, resource_index, *unInterlace_mode);
				if (FAILED(hr)) break;

				sld::lib::mf::safe_release(mf_buffer);
				sld::lib::mf::safe_release(dxgi_buffer);
				sld::lib::mf::safe_release(texture_2d);
			}
		}
		


#if 1 // Copy Whole videos.
		_d3d_renderer->render();
#else // Copy partial video
		{
			INT offset = 5;
			D3D11_BOX region;
			region.left = 0;
			region.right = 300;
			region.top = 0;
			region.bottom = 150;
			region.front = 0;
			region.back = 1;
			for (int i = 0; i < _view_count; i++)
			{
				_d3d_immediate_context->CopySubresourceRegion(backBuffer,
					0, offset + (offset + region.right)* (i % 2), offset + (region.bottom + offset) * (i / 2), 0,
					_view_info[i].buffer, 0, &region);
			}
		}
#endif
		//Record timestamp for this sample & Notify to client
		if (osample != NULL && *osample != NULL)
		{
			LONGLONG hnsTime = 0;
			if (SUCCEEDED(sample->GetSampleTime(&hnsTime)))
			{
				(*osample)->SetSampleTime(hnsTime);

				int32_t time = (int32_t)(hnsTime / 10 / 1000 / 1000);
				if (_current_time != time)
				{
					_current_time = time;
					{//notify to client
						PROPVARIANT var;
						PropVariantInit(&var);
						var.vt = VT_I4;
						var.intVal = time;
						_ms->QueueEvent(MEExtendedType, CLSID_MF_CURRENT_TIMEUPDATE, S_OK, &var);
						PropVariantClear(&var);
					}
				}

			}
		}
	} while (FALSE);

	sld::lib::mf::safe_release(device_input);
	sld::lib::mf::safe_release(mf_buffer);

	return hr;
}

HRESULT sld::lib::mf::sink::video::multiview::renderer::colorspace_convert(UINT view_index, ID3D11Texture2D* pSrcTexture2D, UINT resource_index, UINT32 unInterlaceMode)
{
	HRESULT hr = S_OK;
	ID3D11VideoProcessorOutputView* pOutputView = NULL;
	ID3D11VideoProcessorInputView* pInputView = NULL;
	do
	{
		//create_video_processor(view_index, srcDesc.Width, srcDesc.Height, dst_width, dst_height, unInterlaceMode);

		hr = _video_device->CreateVideoProcessorOutputView(_view_info[view_index].buffer, _view_info[view_index].video_processor_enum, &_output_view_desc, &pOutputView);
		if (FAILED(hr)) break;

		_input_view_desc.Texture2D.ArraySlice = resource_index;
		hr = _video_device->CreateVideoProcessorInputView(pSrcTexture2D, _view_info[view_index].video_processor_enum, &_input_view_desc, &pInputView);
		if (FAILED(hr)) break;

		_stream_data.pInputSurface = pInputView;
		hr = _d3d11_video_context->VideoProcessorBlt(_view_info[view_index].video_processor, pOutputView, 0, 1, &_stream_data);
		if (FAILED(hr))  break;
	} while (FALSE);
	sld::lib::mf::safe_release(pOutputView);
	sld::lib::mf::safe_release(pInputView);
	return hr;
}

HRESULT sld::lib::mf::sink::video::multiview::renderer::set_current_media_type(IMFMediaType* pMediaType)
{
	//hr = pMediaType->QueryInterface(IID_IMFAttributes, reinterpret_cast<void**>(&pAttributes));
	return S_OK;
}
//-------------------------------------------------------------------
// Name: Shutdown
// Description: Releases resources	 held by the presenter.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::multiview::renderer::release(void)
{
	HRESULT hr = MF_E_SHUTDOWN;
	sld::lib::mf::auto_lock lock(&_lock);
    _is_shutdown = TRUE;

	sld::lib::mf::safe_release(_dxgi_manager);
	sld::lib::mf::safe_release(_dxgi_factory2);
	sld::lib::mf::safe_release(_d3d11_device);
	sld::lib::mf::safe_release(_d3d11_immediate_context);
	sld::lib::mf::safe_release(_dxgi_output1);
	sld::lib::mf::safe_release(_sample_allocator_ex);
	sld::lib::mf::safe_release(_dcomp_device);
	sld::lib::mf::safe_release(_hwnd_target);
	sld::lib::mf::safe_release(_root_visual);

	sld::lib::mf::safe_release(_video_device);
	sld::lib::mf::safe_release(_swap_chain1);
	return hr;
}

void sld::lib::mf::sink::video::multiview::renderer::check_decode_switch_regkey(void)
{
	const TCHAR* lpcszDXSW = TEXT("DXSWSwitch");
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
			_dx_sw_switch = dwData;

		dwData = 0;
		cbData = sizeof(DWORD);
		if (0 == RegQueryValueEx(hk, lpcszDComp, 0, &cbType, (LPBYTE)&dwData, &cbData))
			_use_dcomp_visual = dwData;

		dwData = 0;
		cbData = sizeof(DWORD);
		if (0 == RegQueryValueEx(hk, lpcszDebugLayer, 0, &cbType, (LPBYTE)&dwData, &cbData))
			_use_debug_layer = dwData;
	}

	if (NULL != hk)
		RegCloseKey(hk);

	return;
}
        
HRESULT sld::lib::mf::sink::video::multiview::renderer::check_device_state(BOOL * device_changed)
{
	if (!device_changed)
		return E_POINTER;

	static int device_state_checks = 0;
	static D3D_DRIVER_TYPE driver_type = D3D_DRIVER_TYPE_HARDWARE;

	HRESULT hr = set_video_monitor(_hwnd);
	if (FAILED(hr))
		return hr;

	if (_d3d11_device)
	{
		if (S_FALSE == hr || (_dx_sw_switch > 0 && device_state_checks == _dx_sw_switch))
		{
			if (_dx_sw_switch > 0 && device_state_checks == _dx_sw_switch)
			{
				(driver_type == D3D_DRIVER_TYPE_HARDWARE) ? driver_type = D3D_DRIVER_TYPE_WARP : driver_type = D3D_DRIVER_TYPE_HARDWARE;
			}

			hr = create_dxgi_manager_and_device(driver_type);
			if (FAILED(hr))
				return hr;

			*device_changed = TRUE;
			sld::lib::mf::safe_release(_video_device);
			sld::lib::mf::safe_release(_swap_chain1);

			device_state_checks = 0;
		}
		device_state_checks++;
	}
	return hr;
}

BOOL sld::lib::mf::sink::video::multiview::renderer::check_empty_rect(RECT * dst)
{
	GetClientRect(_hwnd, dst);
	return IsRectEmpty(dst);
}

HRESULT sld::lib::mf::sink::video::multiview::renderer::check_shutdown(void) const
{
    if (_is_shutdown)
        return MF_E_SHUTDOWN;
    else
        return S_OK;
}

HRESULT sld::lib::mf::sink::video::multiview::renderer::create_dxgi_manager_and_device(D3D_DRIVER_TYPE driver_type)
{
	HRESULT hr = S_OK;
	IDXGIAdapter * tmpAdapter = NULL;
	ID3D10Multithread * d3d10MultiThread = NULL;
	IDXGIDevice1 * dxgiDevice = NULL;
	IDXGIAdapter1 * adapter = NULL;

	D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
	D3D_FEATURE_LEVEL featureLevel;
	UINT resetToken;

	do
	{
		sld::lib::mf::safe_release(_d3d11_device);
		if (D3D_DRIVER_TYPE_HARDWARE == driver_type)
		{
			hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, _use_debug_layer, featureLevels, ARRAYSIZE(featureLevels), D3D11_SDK_VERSION, &_d3d11_device, &featureLevel, NULL);
			if (SUCCEEDED(hr))
			{
				ID3D11VideoDevice* pDX11VideoDevice = NULL;
				hr = _d3d11_device->QueryInterface(__uuidof(ID3D11VideoDevice), (void**)&pDX11VideoDevice);
				sld::lib::mf::safe_release(pDX11VideoDevice);

				if (SUCCEEDED(hr))
					break;
				sld::lib::mf::safe_release(_d3d11_device);
			}
		}

		if (FAILED(hr))
			break;

		if (NULL == _dxgi_manager)
		{
			hr = MFCreateDXGIDeviceManager(&resetToken, &_dxgi_manager);
			if (FAILED(hr))
				break;
			_device_reset_token = resetToken;
		}

		hr = _dxgi_manager->ResetDevice(_d3d11_device, _device_reset_token);
		if (FAILED(hr))
			break;

		sld::lib::mf::safe_release(_d3d11_immediate_context);
		_d3d11_device->GetImmediateContext(&_d3d11_immediate_context);

		hr = _d3d11_immediate_context->QueryInterface(__uuidof(ID3D11VideoContext), (void**)&_d3d11_video_context);
		if (FAILED(hr)) 
			break;

		hr = _d3d11_immediate_context->QueryInterface(__uuidof(ID3D10Multithread), (void**)&d3d10MultiThread);
		if (FAILED(hr))
			break;
		d3d10MultiThread->SetMultithreadProtected(TRUE);

		if (_d3d_renderer == NULL)
		{
			_d3d_render_ctx.dev = _d3d11_device;
			_d3d_render_ctx.devctx = _d3d11_immediate_context;
			_d3d_renderer = new sld::lib::video::sink::d3d11::multiview::renderer();
		}

		hr = _d3d11_device->QueryInterface(__uuidof(IDXGIDevice1), (LPVOID*)&dxgiDevice);
		if (FAILED(hr))
			break;

		hr = dxgiDevice->GetAdapter(&tmpAdapter);
		if (FAILED(hr))
			break;

		hr = tmpAdapter->QueryInterface(__uuidof(IDXGIAdapter1), (LPVOID*)&adapter);
		if (FAILED(hr))
			break;

		sld::lib::mf::safe_release(_dxgi_factory2);
		hr = adapter->GetParent(__uuidof(IDXGIFactory2), (LPVOID*)&_dxgi_factory2);

	} while (FALSE);

	sld::lib::mf::safe_release(tmpAdapter);
	sld::lib::mf::safe_release(d3d10MultiThread);
	sld::lib::mf::safe_release(dxgiDevice);
	sld::lib::mf::safe_release(adapter);
	return hr;
}

HRESULT sld::lib::mf::sink::video::multiview::renderer::find_bob_processor_index(DWORD* pIndex, UINT index)
{
    HRESULT hr = S_OK;
    D3D11_VIDEO_PROCESSOR_CAPS caps = {};
    D3D11_VIDEO_PROCESSOR_RATE_CONVERSION_CAPS convCaps = {};

    *pIndex = 0;
    hr = _view_info[index].video_processor_enum->GetVideoProcessorCaps(&caps);
	//hr = _video_processor_enum->GetVideoProcessorCaps(&caps);
    if (FAILED(hr))  return hr;

    for (DWORD i = 0; i < caps.RateConversionCapsCount; i++)
    {
        hr = _view_info[index].video_processor_enum->GetVideoProcessorRateConversionCaps(i, &convCaps);
		//hr = _video_processor_enum->GetVideoProcessorRateConversionCaps(i, &convCaps);
        if (FAILED(hr))  return hr;

         //Check the caps to see which deinterlacer is supported
        if ((convCaps.ProcessorCaps & D3D11_VIDEO_PROCESSOR_PROCESSOR_CAPS_DEINTERLACE_BOB) != 0)
        {
            *pIndex = i;
            return hr;
        }
    }

	for (DWORD i = 0; i < caps.RateConversionCapsCount; i++)
	{
		hr = _view_info[index].video_processor_enum->GetVideoProcessorRateConversionCaps(i, &convCaps);
		//hr = _video_processor_enum->GetVideoProcessorRateConversionCaps(i, &convCaps);
		if (FAILED(hr)) return hr;

		if ((convCaps.ProcessorCaps & D3D11_VIDEO_PROCESSOR_PROCESSOR_CAPS_FRAME_RATE_CONVERSION) != 0)
		{
			*pIndex = i;
			return hr;
		}
	}

    return E_FAIL;
}

HRESULT sld::lib::mf::sink::video::multiview::renderer::create_video_processor(UINT index, ID3D11Texture2D* pSrcTexture2D, UINT32 unInterlaceMode)
{
	HRESULT hr = S_OK;
	do
	{
		//if (_view_info &&_view_info[index].dst_width != dst_width)
		{
			ID3D11Texture2D * backBuffer = NULL;
			hr = _swap_chain1->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&backBuffer);
			_d3d_renderer->set_shader_resource_view(index, backBuffer);
			sld::lib::mf::safe_release(backBuffer);
		}
			
		//if (!_view_info[index].video_processor_enum || !_view_info[index].video_processor
		//	|| _view_info[index].src_width != src_width || _view_info[index].src_height != src_height)
		if (!_view_info[index].video_processor_enum || ! _view_info[index].video_processor)
		{
			sld::lib::mf::safe_release(_view_info[index].video_processor_enum);
			sld::lib::mf::safe_release(_view_info[index].video_processor);
			//_view_info[index].src_width = src_width;
			//_view_info[index].src_height = src_height;

			D3D11_VIDEO_PROCESSOR_CONTENT_DESC content_desc;
			ZeroMemory(&content_desc, sizeof(content_desc));
			content_desc.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
			content_desc.InputWidth = _view_info[index].src_width;
			content_desc.InputHeight = _view_info[index].src_height;
			content_desc.OutputWidth = _view_info[index].dst_width;
			content_desc.OutputHeight = _view_info[index].dst_height;
			content_desc.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;

			hr = _video_device->CreateVideoProcessorEnumerator(&content_desc, &_view_info[index].video_processor_enum);
			if (FAILED(hr)) 
				break;

			UINT uiFlags;
			DXGI_FORMAT output_format = DXGI_FORMAT_B8G8R8A8_UNORM;
			hr = _view_info[index].video_processor_enum->CheckVideoProcessorFormat(output_format, &uiFlags);
			if (FAILED(hr) || 0 == (uiFlags & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT))
			{
				hr = MF_E_UNSUPPORTED_D3D_TYPE;
				break;
			}

			DWORD idx;
			hr = find_bob_processor_index(&idx, index);
			if (FAILED(hr)) break;

			hr = _video_device->CreateVideoProcessor(_view_info[index].video_processor_enum, idx, &_view_info[index].video_processor);
			if (FAILED(hr)) break;

			{
				D3D11_VIDEO_FRAME_FORMAT frame_format = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
				if (MFVideoInterlace_FieldInterleavedUpperFirst == unInterlaceMode || MFVideoInterlace_FieldSingleUpper == unInterlaceMode || MFVideoInterlace_MixedInterlaceOrProgressive == unInterlaceMode)
				{
					frame_format = D3D11_VIDEO_FRAME_FORMAT_INTERLACED_TOP_FIELD_FIRST;
				}
				else if (MFVideoInterlace_FieldInterleavedLowerFirst == unInterlaceMode || MFVideoInterlace_FieldSingleLower == unInterlaceMode)
				{
					frame_format = D3D11_VIDEO_FRAME_FORMAT_INTERLACED_BOTTOM_FIELD_FIRST;
				}

				// input format
				_d3d11_video_context->VideoProcessorSetStreamFrameFormat(_view_info[index].video_processor, 0, frame_format);

				// Output rate (repeat frames)
				_d3d11_video_context->VideoProcessorSetStreamOutputRate(_view_info[index].video_processor, 0, D3D11_VIDEO_PROCESSOR_OUTPUT_RATE_NORMAL, TRUE, NULL);

				// Source rect
				RECT srcRect;
				srcRect.left = 0;
				srcRect.top = 0;
				srcRect.right = _view_info[index].src_width;
				srcRect.bottom = _view_info[index].src_height;
				_d3d11_video_context->VideoProcessorSetStreamSourceRect(_view_info[index].video_processor, 0, TRUE, &srcRect);

				// Stream dest rect
				RECT dstRect;
				dstRect.left = 0;
				dstRect.top = 0;
				dstRect.right = _view_info[index].dst_width;
				dstRect.bottom = _view_info[index].dst_height;
				_d3d11_video_context->VideoProcessorSetStreamDestRect(_view_info[index].video_processor, 0, TRUE, &dstRect);

				//_pD3D_video_context->VideoProcessorSetOutputTargetRect(_view_info[index].video_processor, TRUE, &_back_buffer);

				// Stream color space
				D3D11_VIDEO_PROCESSOR_COLOR_SPACE colorSpace = {};
				colorSpace.YCbCr_xvYCC = 1;
				
				_d3d11_video_context->VideoProcessorSetStreamColorSpace(_view_info[index].video_processor, 0, &colorSpace);

				// Output color space
				_d3d11_video_context->VideoProcessorSetOutputColorSpace(_view_info[index].video_processor, &colorSpace);

				// Output background color (black)
				D3D11_VIDEO_COLOR backgroundColor = {};
				backgroundColor.RGBA.A = 1.0F;
				backgroundColor.RGBA.R = 1.0F * static_cast<float>(GetRValue(0)) / 255.0F;
				backgroundColor.RGBA.G = 1.0F * static_cast<float>(GetGValue(0)) / 255.0F;
				backgroundColor.RGBA.B = 1.0F * static_cast<float>(GetBValue(0)) / 255.0F;

				_d3d11_video_context->VideoProcessorSetOutputBackgroundColor(_view_info[index].video_processor, FALSE, &backgroundColor);
			}
		}
	} while (0);
	return hr;
}

HRESULT sld::lib::mf::sink::video::multiview::renderer::set_monitor(UINT adapterID)
{
    HRESULT hr = S_OK;
    DWORD dwMatchID = 0;

	sld::lib::mf::auto_lock lock(&_lock);

    do
    {
        hr = _monitors->MatchGUID(adapterID, &dwMatchID);
        if (FAILED(hr))
        {
            break;
        }

        if (hr == S_FALSE)
        {
            hr = E_INVALIDARG;
            break;
        }

        _current_monitor = &(*_monitors)[dwMatchID];
        _connection_guid = adapterID;
    }
    while (FALSE);

    return hr;
}

HRESULT sld::lib::mf::sink::video::multiview::renderer::set_video_monitor(HWND hwndVideo)
{
    HRESULT hr = S_OK;
    CAMDDrawMonitorInfo* pMonInfo = NULL;
    HMONITOR hMon = NULL;

    if (!_monitors)
    {
        return E_UNEXPECTED;
    }

    hMon = MonitorFromWindow(hwndVideo, MONITOR_DEFAULTTONULL);

    do
    {
        if (NULL != hMon)
        {
            _monitors->TerminateDisplaySystem();
            _current_monitor = NULL;

            hr = _monitors->InitializeDisplaySystem(hwndVideo);
            if (FAILED(hr))
            {
                break;
            }

            pMonInfo = _monitors->FindMonitor(hMon);
            if (NULL != pMonInfo && pMonInfo->uDevID != _connection_guid)
            {
                hr = set_monitor(pMonInfo->uDevID);
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
    }
    while(FALSE);

    return hr;
}

_Post_satisfies_(this->_swap_chain1 != NULL)
HRESULT sld::lib::mf::sink::video::multiview::renderer::update_dxgi_swap_chain(void)
{
    HRESULT hr = S_OK;
    // Get the DXGISwapChain1
    DXGI_SWAP_CHAIN_DESC1 scd;
    ZeroMemory(&scd, sizeof(scd));
    scd.SampleDesc.Count = 1;
    scd.SampleDesc.Quality = 0; 
	scd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;// DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL; 
    scd.Scaling = DXGI_SCALING_STRETCH;
    scd.Width = _back_buffer.right;
    scd.Height = _back_buffer.bottom;
    scd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	scd.Stereo = FALSE;
	scd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    scd.BufferUsage = DXGI_USAGE_BACK_BUFFER | DXGI_USAGE_RENDER_TARGET_OUTPUT;
    
    scd.BufferCount = 2;

    do
    {
        if (_swap_chain1)
        {
            hr = _swap_chain1->ResizeBuffers(4, _back_buffer.right, _back_buffer.bottom, scd.Format, scd.Flags);
            break;
        }
        if (!_use_dcomp_visual)
        {
            hr = _dxgi_factory2->CreateSwapChainForHwnd(_d3d11_device, _hwnd, &scd, NULL, NULL, &_swap_chain1);
			if (FAILED(hr)) break;

            if (_is_full_screen)
            {
                hr = _swap_chain1->SetFullscreenState(TRUE, NULL);
				if (FAILED(hr)) break;
            }
            else
            {
                hr = _swap_chain1->SetFullscreenState(FALSE, NULL);
				if (FAILED(hr)) break;
            }
        }
        else
        {
            // Create a swap chain for composition
            hr = _dxgi_factory2->CreateSwapChainForComposition(_d3d11_device, &scd, NULL, &_swap_chain1);
			if (FAILED(hr)) break;
			
            hr = _root_visual->SetContent(_swap_chain1);
            if (FAILED(hr)) break;

            hr = _dcomp_device->Commit();
			if (FAILED(hr)) break;
        }
		if (_d3d_renderer)
		{
			_d3d_render_ctx.sw = _swap_chain1;
			_d3d_render_ctx.vc = _view_count;
			_d3d_render_ctx.vi = _view_info;
			_d3d_render_ctx.width = _vcodec_width;
			_d3d_render_ctx.height = _vcodec_height;
			//_d3d_renderer->initialize_config(_swap_chain1, _view_count, _view_info, _vcodec_width, _vcodec_height);
		}
    }
    while (FALSE);

    return hr;
}

void sld::lib::mf::sink::video::multiview::renderer::set_view_count(INT count)
{
	bool changed = (_view_count != count);
	_view_count = count;
	if (_view_info == NULL)
		_view_info = new sld::lib::video::sink::d3d11::multiview::renderer::view_session_t[_view_count];
	if (_view_info && changed)
	{
		delete _view_info;
		_view_info = new sld::lib::video::sink::d3d11::multiview::renderer::view_session_t[_view_count];
	}
}

void sld::lib::mf::sink::video::multiview::renderer::enable_coordinated_cs_converter(BOOL enable)
{
	_enable_coordinated_cs_converter = enable;
}

void sld::lib::mf::sink::video::multiview::renderer::set_view_position(INT index, FLOAT* position)
{
	if (_view_info)
	{
		_view_info[index].position[0] = position[0];
		_view_info[index].position[1] = position[1];
		_view_info[index].position[2] = position[2];
		_view_info[index].position[3] = position[3];
	}
}

void sld::lib::mf::sink::video::multiview::renderer::set_selected(INT index)
{
	_selected = index;
	if (_d3d_renderer)
		_d3d_renderer->select(index);
}

void sld::lib::mf::sink::video::multiview::renderer::maximize(void)
{
	_maximize = !_maximize;
	if (_d3d_renderer)
		_d3d_renderer->maximize();
}

void sld::lib::mf::sink::video::multiview::renderer::change_render_type(void)
{
	/*
	if (_d3d_renderer)
		_d3d_renderer->change_render_type();
	*/
}
