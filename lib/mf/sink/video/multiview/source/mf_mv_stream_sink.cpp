#include "mf_mv_stream_sink.h"

GUID const * const solids::lib::mf::sink::video::multiview::stream::_video_formats[] =
{
	&MFVideoFormat_NV12,
	&MFVideoFormat_IYUV,
	&MFVideoFormat_YUY2,
	&MFVideoFormat_YV12,
	&MFVideoFormat_RGB32,
	&MFVideoFormat_RGB32,
	&MFVideoFormat_RGB24,
	&MFVideoFormat_RGB555,
	&MFVideoFormat_RGB565,
	&MFVideoFormat_RGB8,
	&MFVideoFormat_AYUV,
	&MFVideoFormat_UYVY,
	&MFVideoFormat_YVYU,
	&MFVideoFormat_YVU9,
	&MEDIASUBTYPE_V216,
	&MFVideoFormat_v410,
	&MFVideoFormat_I420,
	&MFVideoFormat_NV11,
	&MFVideoFormat_420O
};

const DWORD solids::lib::mf::sink::video::multiview::stream::_nvideo_formats = sizeof(solids::lib::mf::sink::video::multiview::stream::_video_formats) / sizeof(solids::lib::mf::sink::video::multiview::stream::_video_formats[0]);
const MFRatio solids::lib::mf::sink::video::multiview::stream::_default_fps = { 30, 1 };
const solids::lib::mf::sink::video::multiview::stream::format_entry_t solids::lib::mf::sink::video::multiview::stream::_dxgi_format_mapping[] =
{
	{ MFVideoFormat_RGB32,      DXGI_FORMAT_B8G8R8X8_UNORM },
	{ MFVideoFormat_ARGB32,     DXGI_FORMAT_R8G8B8A8_UNORM },
	{ MFVideoFormat_AYUV,      DXGI_FORMAT_AYUV },
	{ MFVideoFormat_YUY2,      DXGI_FORMAT_YUY2 },
	{ MFVideoFormat_NV12,      DXGI_FORMAT_NV12 },
	{ MFVideoFormat_NV11,      DXGI_FORMAT_NV11 },
	{ MFVideoFormat_AI44,      DXGI_FORMAT_AI44 },
	{ MFVideoFormat_P010,      DXGI_FORMAT_P010 },
	{ MFVideoFormat_P016,      DXGI_FORMAT_P016 },
	{ MFVideoFormat_Y210,      DXGI_FORMAT_Y210 },
	{ MFVideoFormat_Y216,      DXGI_FORMAT_Y216 },
	{ MFVideoFormat_Y410,      DXGI_FORMAT_Y410 },
	{ MFVideoFormat_Y416,      DXGI_FORMAT_Y416 },
	{ MFVideoFormat_420O,      DXGI_FORMAT_420_OPAQUE }
};

#define SAMPLE_QUEUE_HIWATER_THRESHOLD 3
#define MAX_PAST_FRAMES         3


BOOL solids::lib::mf::sink::video::multiview::stream::_valid_state_mat[solids::lib::mf::sink::video::multiview::stream::state_t::count][solids::lib::mf::sink::async_operation::type_t::count] =
{
    // States:    Operations:
    //            SetType   Start     Restart   Pause     Stop      Sample    Marker
    /* NotSet */  TRUE,     FALSE,    FALSE,    FALSE,    FALSE,    FALSE,    FALSE,

    /* Ready */   TRUE,     TRUE,     TRUE,     TRUE,     TRUE,     FALSE,    TRUE,

    /* Start */   TRUE,     TRUE,     FALSE,    TRUE,     TRUE,     TRUE,     TRUE,

    /* Pause */   TRUE,     TRUE,     TRUE,     TRUE,     TRUE,     TRUE,     TRUE,

    /* Stop */    TRUE,     TRUE,     FALSE,    FALSE,    TRUE,     FALSE,    TRUE

    // Note about states:
    // 1. OnClockRestart should only be called from paused state.
    // 2. While paused, the sink accepts samples but does not process them.

};

#pragma warning( push )
#pragma warning( disable : 4355 )  // 'this' used in base member initializer list

solids::lib::mf::sink::video::multiview::stream::stream(DWORD id, solids::lib::mf::critical_section & lock, solids::lib::mf::scheduler * sched)
	: _stream_id(id)
    , _lock(lock)
	, _state(state_t::type_not_set)
    , _is_shutdown(FALSE)
    , _work_queue_id(0)
    , _work_queue_cb(this, &solids::lib::mf::sink::video::multiview::stream::dispatch_workitem_callback)
	, _consume_data(solids::lib::mf::sink::video::multiview::stream::consume_state_t::process_samples)
    , _stime(0)
    , _nwritten(0)
    , _noutstanding_sample_requests(0)
    , _sink(NULL)
    , _event_queue(NULL)
    , _renderer(NULL)
    , _scheduler(sched)
    , _current_type(NULL)
    , _prerolling(FALSE)
    , _waiting_for_clock_start(FALSE)
    , _samples_to_process() // default ctor
    , _uninterlace_mode(MFVideoInterlace_Progressive)
    , _image_bytes_pp() // default ctor
    , _dxgi_format(DXGI_FORMAT_UNKNOWN)
{
    _image_bytes_pp.numerator = 1;
    _image_bytes_pp.denominator = 1;
}

#pragma warning( pop )

//-------------------------------------------------------------------
// CStreamSink destructor
//-------------------------------------------------------------------

solids::lib::mf::sink::video::multiview::stream::~stream(void)
{
}

// IUnknown methods

ULONG solids::lib::mf::sink::video::multiview::stream::AddRef(void)
{
	return solids::lib::mf::refcount_object::AddRef();
}

HRESULT solids::lib::mf::sink::video::multiview::stream::QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void** ppv)
{
    if (!ppv)
    {
        return E_POINTER;
    }
    if (iid == IID_IUnknown)
    {
        *ppv = static_cast<IUnknown*>(static_cast<IMFStreamSink*>(this));
    }
    else if (iid == __uuidof(IMFStreamSink ))
    {
        *ppv = static_cast<IMFStreamSink*>(this);
    }
    else if (iid == __uuidof(IMFMediaEventGenerator))
    {
        *ppv = static_cast<IMFMediaEventGenerator*>(this);
    }
    else if (iid == __uuidof(IMFMediaTypeHandler))
    {
        *ppv = static_cast<IMFMediaTypeHandler*>(this);
    }
    else if (iid == __uuidof(IMFGetService))
    {
        *ppv = static_cast<IMFGetService*>(this);
    }
    else if (iid == __uuidof(IMFAttributes))
    {
        *ppv = static_cast<IMFAttributes*>(this);
    }
    else
    {
        *ppv = NULL;
        return E_NOINTERFACE;
    }
    AddRef();
    return S_OK;
}

ULONG solids::lib::mf::sink::video::multiview::stream::Release(void)
{
	return solids::lib::mf::refcount_object::Release();
}

/// IMFStreamSink methods

//-------------------------------------------------------------------
// Name: Flush
// Description: Discards all samples that were not processed yet.
//-------------------------------------------------------------------

HRESULT solids::lib::mf::sink::video::multiview::stream::Flush(void)
{
	HRESULT hr = check_shutdown();
	if (SUCCEEDED(hr))
	{
		_consume_data = consume_state_t::drop_samples;
		// Note: Even though we are flushing data, we still need to send
		// any marker events that were queued.
		hr = process_samples_from_queue(_consume_data);
	}

	if (SUCCEEDED(hr))
	{
		// This call blocks until the scheduler threads discards all scheduled samples.
		_scheduler->flush();
		hr = _renderer->flush();
	}
	_consume_data = consume_state_t::process_samples;
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::GetIdentifier(__RPC__out DWORD* id)
{
	solids::lib::mf::auto_lock lock(&_lock);

	if (!id)
		return E_POINTER;

	HRESULT hr = check_shutdown();
	if (SUCCEEDED(hr))
		*id = _stream_id;

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::GetMediaSink(__RPC__deref_out_opt IMFMediaSink ** ppms)
{
	solids::lib::mf::auto_lock lock(&_lock);

	if (!ppms)
		return E_POINTER;

	HRESULT hr = check_shutdown();
	if (SUCCEEDED(hr))
	{
		*ppms = _sink;
		(*ppms)->AddRef();
	}
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::GetMediaTypeHandler(__RPC__deref_out_opt IMFMediaTypeHandler** handler)
{
    solids::lib::mf::auto_lock lock(&_lock);

    if (handler == NULL)
        return E_POINTER;

    HRESULT hr = check_shutdown();
    if (SUCCEEDED(hr))
        hr = this->QueryInterface(IID_IMFMediaTypeHandler, (void**)handler);

    return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::PlaceMarker(MFSTREAMSINK_MARKER_TYPE marker_type, __RPC__in const PROPVARIANT * marker_value, __RPC__in const PROPVARIANT * context_value)
{
	solids::lib::mf::auto_lock lock(&_lock);

	HRESULT hr = S_OK;
	IMarker * marker = NULL;
	hr = check_shutdown();
	if (SUCCEEDED(hr))
		hr = validate_operation(async_operation::type_t::place_marker);

	if (SUCCEEDED(hr))
		hr = marker::create(marker_type, marker_value, context_value, &marker);

	if (SUCCEEDED(hr))
		hr = _samples_to_process.queue(marker);

	if (SUCCEEDED(hr))
	{
		if (_state != state_t::paused)
			hr = queue_async_operation(async_operation::type_t::place_marker); 
	}
	safe_release(marker);

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::ProcessSample(__RPC__in_opt IMFSample* pSample)
{
    if (_noutstanding_sample_requests == 0)  
		return MF_E_INVALIDREQUEST;

    HRESULT hr = S_OK;
    do
    {
        hr = check_shutdown();
        if (FAILED(hr)) break;

		_noutstanding_sample_requests--;

        if (!_prerolling && !_waiting_for_clock_start)
        {
            hr = validate_operation(async_operation::type_t::process_sample);
			if (FAILED(hr)) break;
        }

		if (pSample)
		{
			hr = _samples_to_process.queue(pSample);
			if (FAILED(hr)) break;
		}
			
		if (_prerolling)
		{
			_prerolling = FALSE;
			return QueueEvent(MEStreamSinkPrerolled, GUID_NULL, S_OK, NULL);
		}

		if (_state != state_t::paused && _state != state_t::stopped)
			hr = queue_async_operation(async_operation::type_t::process_sample);
    }
    while (FALSE);

    return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::BeginGetEvent(IMFAsyncCallback * callback, IUnknown * unk_state)
{
    HRESULT hr = S_OK;

    solids::lib::mf::auto_lock lock(&_lock);
    hr = check_shutdown();

	if (SUCCEEDED(hr))
		hr = _event_queue->BeginGetEvent(callback, unk_state);

    return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::EndGetEvent(IMFAsyncResult * result, _Out_  IMFMediaEvent ** ppevent)
{
	solids::lib::mf::auto_lock lock(&_lock);

	HRESULT hr = S_OK;
	hr = check_shutdown();
	if (SUCCEEDED(hr))
		hr = _event_queue->EndGetEvent(result, ppevent);

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::GetEvent(DWORD flags, __RPC__deref_out_opt IMFMediaEvent ** ppevent)
{
	HRESULT hr = S_OK;
	IMFMediaEventQueue * queue = NULL;

	{ // scope for lock
		solids::lib::mf::auto_lock lock(&_lock);

		hr = check_shutdown();
		if (SUCCEEDED(hr))
		{
			queue = _event_queue;
			queue->AddRef();
		}
	}

	if (SUCCEEDED(hr))
		hr = queue->GetEvent(flags, ppevent);

	safe_release(queue);
	return hr;

    return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::QueueEvent(MediaEventType met, __RPC__in REFGUID guid_extended_type, HRESULT status, __RPC__in_opt const PROPVARIANT * value)
{
	solids::lib::mf::auto_lock lock(&_lock);

	HRESULT hr = S_OK;
	hr = check_shutdown();
	if (SUCCEEDED(hr))
		hr = _event_queue->QueueEventParamVar(met, guid_extended_type, status, value);

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::GetCurrentMediaType(_Outptr_ IMFMediaType ** ppmt)
{
	solids::lib::mf::auto_lock lock(&_lock);

	if (!ppmt)
		return E_POINTER;

	HRESULT hr = check_shutdown();
	if (SUCCEEDED(hr))
	{
		if (!_current_type)
			hr = MF_E_NOT_INITIALIZED;
	}

	if (SUCCEEDED(hr))
	{
		*ppmt = _current_type;
		(*ppmt)->AddRef();
	}

	return hr;
}

//-------------------------------------------------------------------
// Name: GetMajorType
// Description: Return the major type GUID.
//-------------------------------------------------------------------
HRESULT solids::lib::mf::sink::video::multiview::stream::GetMajorType(__RPC__out GUID * guid_major_type)
{
	if (!guid_major_type)
		return E_POINTER;

	HRESULT hr = check_shutdown();
	if (FAILED(hr))
		return hr;

	if (!_current_type)
		return MF_E_NOT_INITIALIZED;

	return _current_type->GetGUID(MF_MT_MAJOR_TYPE, guid_major_type);
}

HRESULT solids::lib::mf::sink::video::multiview::stream::GetMediaTypeByIndex(DWORD index, _Outptr_ IMFMediaType ** ppmt)
{
	HRESULT hr = S_OK;
	do
	{
		if (!ppmt)
		{
			hr = E_POINTER;
			break;
		}

		hr = check_shutdown();
		if (FAILED(hr))
			break;

		if (index >= _nvideo_formats)
		{
			hr = MF_E_NO_MORE_TYPES;
			break;
		}

		IMFMediaType * vmt = NULL;
		do
		{
			hr = MFCreateMediaType(&vmt);
			if (FAILED(hr))
				break;

			hr = vmt->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
			if (FAILED(hr))
				break;

			hr = vmt->SetGUID(MF_MT_SUBTYPE, *(_video_formats[index]));
			if (FAILED(hr))
				break;

			vmt->AddRef();
			*ppmt = vmt;

		} while (FALSE);

		safe_release(vmt);

		if (FAILED(hr))
			break;

	} while (FALSE);

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::GetMediaTypeCount(__RPC__out DWORD * tc)
{
	HRESULT hr = S_OK;
	do
	{
		if (!tc)
		{
			hr = E_POINTER;
			break;
		}

		hr = check_shutdown();
		if (FAILED(hr))
			break;

		*tc = _nvideo_formats;

	} while (FALSE);

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::IsMediaTypeSupported(IMFMediaType * pmt, _Outptr_opt_result_maybenull_ IMFMediaType ** ppmt)
{
	HRESULT hr = S_OK;
	GUID sub_type = GUID_NULL;
	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		if (pmt == NULL)
		{
			hr = E_POINTER;
			break;
		}

		hr = pmt->GetGUID(MF_MT_SUBTYPE, &sub_type);
		if (FAILED(hr))
			break;

		hr = MF_E_INVALIDMEDIATYPE;
		for (DWORD i = 0; i < _nvideo_formats; i++)
		{
			if (sub_type == (*_video_formats[i]))
			{
				hr = S_OK;
				break;
			}
		}

		if (FAILED(hr))
			break;

		for (DWORD i = 0; i < ARRAYSIZE(_dxgi_format_mapping); i++)
		{
			const format_entry_t & e = _dxgi_format_mapping[i];
			if (e.sub_type == sub_type)
			{
				_dxgi_format = e.dxgi_format;
				break;
			}
		}

		hr = _renderer->is_media_type_supported(pmt, _dxgi_format);
		if (FAILED(hr))
			break;

	} while (FALSE);

	if (ppmt)
		*ppmt = NULL;

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::SetCurrentMediaType(IMFMediaType * pmt)
{
	if (!pmt)
		return E_POINTER;

	HRESULT hr = S_OK;
	MFRatio fps = { 0, 0 };
	GUID guid_sub_type = GUID_NULL;

	solids::lib::mf::auto_lock lock(&_lock);
	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		hr = validate_operation(async_operation::type_t::set_media_type);
		if (FAILED(hr))
			break;

		hr = IsMediaTypeSupported(pmt, NULL);
		if (FAILED(hr))
			break;

		safe_release(_current_type);
		_current_type = pmt;
		_current_type->AddRef();

		pmt->GetGUID(MF_MT_SUBTYPE, &guid_sub_type);
		if ((guid_sub_type == MFVideoFormat_NV12) || (guid_sub_type == MFVideoFormat_YV12) || (guid_sub_type == MFVideoFormat_IYUV) ||
			(guid_sub_type == MFVideoFormat_YVU9) || (guid_sub_type == MFVideoFormat_I420))
		{
			_image_bytes_pp.numerator = 3;
			_image_bytes_pp.denominator = 2;
		}
		else if ((guid_sub_type == MFVideoFormat_YUY2) || (guid_sub_type == MFVideoFormat_RGB555) || (guid_sub_type == MFVideoFormat_RGB565) ||
			(guid_sub_type == MFVideoFormat_UYVY) || (guid_sub_type == MFVideoFormat_YVYU) || (guid_sub_type == MEDIASUBTYPE_V216))
		{
			_image_bytes_pp.numerator = 2;
			_image_bytes_pp.denominator = 1;
		}
		else if (guid_sub_type == MFVideoFormat_RGB24)
		{
			_image_bytes_pp.numerator = 3;
			_image_bytes_pp.denominator = 1;
		}
		else if (guid_sub_type == MFVideoFormat_RGB32)
		{
			_image_bytes_pp.numerator = 4;
			_image_bytes_pp.denominator = 1;
		}
		else if (guid_sub_type == MFVideoFormat_v410)
		{
			_image_bytes_pp.numerator = 5;
			_image_bytes_pp.denominator = 4;
		}
		else
		{
			// This is just a fail-safe
			_image_bytes_pp.numerator = 1;
			_image_bytes_pp.denominator = 1;
		}

		pmt->GetUINT32(MF_MT_INTERLACE_MODE, &_uninterlace_mode);
		if (SUCCEEDED(get_fps(pmt, &fps)) && (fps.Numerator != 0) && (fps.Denominator != 0))
		{
			if (MFVideoInterlace_FieldInterleavedUpperFirst == _uninterlace_mode ||
				MFVideoInterlace_FieldInterleavedLowerFirst == _uninterlace_mode ||
				MFVideoInterlace_FieldSingleUpper == _uninterlace_mode ||
				MFVideoInterlace_FieldSingleLower == _uninterlace_mode ||
				MFVideoInterlace_MixedInterlaceOrProgressive == _uninterlace_mode)
			{
				fps.Numerator *= 2;
			}

			UINT64 avg_time_per_frame = 0;
			MFFrameRateToAverageTimePerFrame(fps.Numerator, fps.Denominator, &avg_time_per_frame);
			MFTIME per_frame_interval = (MFTIME)avg_time_per_frame;
			LONGLONG duration = per_frame_interval / 4;
			_scheduler->set_duration(duration);
		}
		else
		{
			UINT64 avg_time_per_frame = 0;
			MFFrameRateToAverageTimePerFrame(_default_fps.Numerator, _default_fps.Denominator, &avg_time_per_frame);
			MFTIME per_frame_interval = (MFTIME)avg_time_per_frame;
			LONGLONG duration = per_frame_interval / 4;
			_scheduler->set_duration(duration);
		}

		// Update the required sample count based on the media type (progressive vs. interlaced)
		if (_uninterlace_mode == MFVideoInterlace_Progressive)
		{
			// XVP will hold on to 1 sample but that's the same sample we will internally hold on to
			hr = SetUINT32(MF_SA_REQUIRED_SAMPLE_COUNT, SAMPLE_QUEUE_HIWATER_THRESHOLD);
		}
		else
		{
			// Assume we will need a maximum of 3 backward reference frames for deinterlacing
			// However, one of the frames is "shared" with SVR
			hr = SetUINT32(MF_SA_REQUIRED_SAMPLE_COUNT, SAMPLE_QUEUE_HIWATER_THRESHOLD + MAX_PAST_FRAMES - 1);
		}

		if (SUCCEEDED(hr))
		{
			hr = _renderer->set_current_media_type(pmt);
			if (FAILED(hr))
				break;
		}

		if ((_state != state_t::started) && (_state != state_t::paused))
			_state = state_t::ready;
		else
		{
			//Flush all current samples in the Queue as this is a format change
			hr = Flush();
			return hr;
		}
	} while (FALSE);

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::GetService(__RPC__in REFGUID guid_service, __RPC__in REFIID iid, __RPC__deref_out_opt LPVOID * ppv)
{
	IMFGetService * get_service = NULL;
	HRESULT hr = _sink->QueryInterface(IID_PPV_ARGS(&get_service));
	if (SUCCEEDED(hr))
		hr = get_service->GetService(guid_service, iid, ppv);

	safe_release(get_service);
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::set_uuid(LPCTSTR uuid)
{
	wcsncpy_s(_uuid, uuid, wcslen(uuid));
	return S_OK;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::process(IMFSample * sample)
{
	HRESULT hr = S_OK;
	if (_consume_data == solids::lib::mf::sink::video::multiview::stream::consume_state_t::drop_samples)
		return hr;

	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		hr = _renderer->render_samples(sample);
		if (FAILED(hr))
			break;

	} while (FALSE);

	if (SUCCEEDED(hr))
	{
		// Unless we are paused/stopped, start an async operation to dispatch the next sample.
		if (_state != state_t::paused && _state != state_t::stopped)
		{
			// Queue the operation.
			hr = queue_async_operation(async_operation::type_t::process_sample);
		}
	}
	else
	{
		// We are in the middle of an asynchronous operation, so if something failed, send an error.
		hr = QueueEvent(MEError, GUID_NULL, hr, NULL);
	}

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::get_max_rate(BOOL thin, float * rate)
{
	HRESULT hr = S_OK;
	DWORD monitor_refresh_rate = 0;
	UINT32 numerator = 0;
	UINT32 denominator = 0;
	do
	{
		hr = _renderer->get_monitor_refresh_rate(&monitor_refresh_rate);
		if (FAILED(hr))
			break;

		if (!_current_type)
		{
			hr = MF_E_INVALIDREQUEST;
			break;
		}

		if (thin)
		{
			*rate = FLT_MAX;
			break;
		}

		MFGetAttributeRatio(_current_type, MF_MT_FRAME_RATE, &numerator, &denominator);
		if (numerator == 0 || denominator == 0)
		{
			// We support anything.
			*rate = FLT_MAX;
		}
		else
		{
			if (MFVideoInterlace_FieldInterleavedUpperFirst == _uninterlace_mode ||
				MFVideoInterlace_FieldInterleavedLowerFirst == _uninterlace_mode ||
				MFVideoInterlace_FieldSingleUpper == _uninterlace_mode ||
				MFVideoInterlace_FieldSingleLower == _uninterlace_mode ||
				MFVideoInterlace_MixedInterlaceOrProgressive == _uninterlace_mode)
			{
				numerator *= 2;
			}
			*rate = (float)MulDiv(monitor_refresh_rate, denominator, numerator);
		}

	} while (FALSE);

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::initialize(IMFMediaSink * parent, solids::lib::mf::sink::video::multiview::renderer * renderer)
{
	HRESULT hr = S_OK;
	if (SUCCEEDED(hr))
		hr = solids::lib::mf::attributes<IMFAttributes>::initialize();

	// Create the event queue helper.
	if (SUCCEEDED(hr))
		hr = MFCreateEventQueue(&_event_queue);

	if (SUCCEEDED(hr))
	{
		hr = MFAllocateWorkQueue(&_work_queue_id);	//MFASYNC_CALLBACK_QUEUE_STANDARD
													//hr = MFAllocateSerialWorkQueue(MFASYNC_CALLBACK_QUEUE_MULTITHREADED, &m_WorkQueueId);
													//hr = MFAllocateWorkQueue(&m_WorkQueueId);	
	}

	if (SUCCEEDED(hr))
	{
		_renderer = renderer;
		_renderer->AddRef();
	}

	if (SUCCEEDED(hr))
	{
		_sink = parent;
		_sink->AddRef();
	}
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::release(void)
{
	_is_shutdown = TRUE;
	if (_event_queue)
		_event_queue->Shutdown();

	MFUnlockWorkQueue(_work_queue_id);
	_samples_to_process.clear();

	solids::lib::mf::safe_release(_sink);
	solids::lib::mf::safe_release(_event_queue);
	solids::lib::mf::safe_release(_renderer);
	solids::lib::mf::safe_release(_current_type);

	return MF_E_SHUTDOWN;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::pause(void)
{
	HRESULT hr = validate_operation(solids::lib::mf::sink::async_operation::type_t::pause);
	if (SUCCEEDED(hr))
	{
		_state = state_t::paused;
		hr = queue_async_operation(solids::lib::mf::sink::async_operation::type_t::pause);
	}
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::preroll(void)
{
	HRESULT hr = check_shutdown();
	if (SUCCEEDED(hr))
	{
		_prerolling = TRUE;
		_waiting_for_clock_start = TRUE;

		// Kick things off by requesting a sample...
		//_noutstanding_sample_requests++;
		hr = QueueEvent(MEStreamSinkRequestSample, GUID_NULL, hr, NULL);
		//cap_log4cplus_logger::make_debug_log("amadeus.player", "[ Renderer ] SAMPLE REQUEST");
		if (SUCCEEDED(hr))
			_noutstanding_sample_requests++;
	}
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::restart(void)
{
	HRESULT hr = validate_operation(solids::lib::mf::sink::async_operation::type_t::restart);
	if (SUCCEEDED(hr))
	{
		_state = state_t::started;
		hr = queue_async_operation(solids::lib::mf::sink::async_operation::type_t::restart);
	}
	return hr;
}
//
//HRESULT amadeus::mf::sink::video::mv_stream_sink::Shutdown(void)
//{
//    //auto_lock lock(&_cs);
//
//    _is_shutdown = TRUE;
//
//    if (_event_queue)
//    {
//        _event_queue->Shutdown();
//    }
//
//    MFUnlockWorkQueue(_work_queue_id);
//
//    _samples_to_process.Clear();
//
//    safe_release(_sink);
//    safe_release(_event_queue);
//    //safe_release(m_pByteStream);
//    safe_release(_renderer);
//    safe_release(_current_type);
//
//    return MF_E_SHUTDOWN;
//}

HRESULT solids::lib::mf::sink::video::multiview::stream::start(MFTIME start)
{
	HRESULT hr = S_OK;
	do
	{
		hr = validate_operation(solids::lib::mf::sink::async_operation::type_t::start);
		if (FAILED(hr))
			break;

		if (start != PRESENTATION_CURRENT_POSITION)
			_stime = start;        // We're starting from a "new" position, Cache the start time.

		_state = state_t::started;
		hr = queue_async_operation(solids::lib::mf::sink::async_operation::type_t::start);

	} while (FALSE);

	_waiting_for_clock_start = FALSE;
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::stop(void)
{
	HRESULT hr = validate_operation(solids::lib::mf::sink::async_operation::type_t::stop);
	if (SUCCEEDED(hr))
	{
		_state = state_t::stopped;
		hr = queue_async_operation(solids::lib::mf::sink::async_operation::type_t::stop);
	}
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::dispatch_process_sample(solids::lib::mf::sink::async_operation * operation)
{
	assert(operation != NULL);

	HRESULT hr = check_shutdown();
	if (FAILED(hr))
		return hr;

	if (_renderer->can_process_next_sample())
	{
		hr = process_samples_from_queue(consume_state_t::process_samples);
		if (SUCCEEDED(hr))
		{
			if (operation->op() == async_operation::type_t::process_sample)
				hr = request_sample();
		}

		if (FAILED(hr))
			hr = QueueEvent(MEError, GUID_NULL, hr, NULL);
	}
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::check_shutdown(void) const
{
	if (_is_shutdown)
		return MF_E_SHUTDOWN;
	else
		return S_OK;
}

inline HRESULT solids::lib::mf::sink::video::multiview::stream::get_fps(IMFMediaType * type, MFRatio * ratio)
{
	return MFGetAttributeRatio(type, MF_MT_FRAME_RATE, (UINT32*)&ratio->Numerator, (UINT32*)&ratio->Denominator);
}

BOOL solids::lib::mf::sink::video::multiview::stream::need_more_samples(void)
{
	const DWORD samples_in_flight = _samples_to_process.get_count() + _noutstanding_sample_requests;
	return samples_in_flight < SAMPLE_QUEUE_HIWATER_THRESHOLD;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::dispatch_workitem_callback(IMFAsyncResult * ar)
{
	solids::lib::mf::auto_lock lock(&_lock);

	HRESULT hr = check_shutdown();
	if (FAILED(hr))
		return hr;

	IUnknown * state = NULL;

	hr = ar->GetState(&state);
	if (SUCCEEDED(hr))
	{
		solids::lib::mf::sink::async_operation * aop = (solids::lib::mf::sink::async_operation*)state;

		int32_t op = aop->op();
		switch (op)
		{
		case solids::lib::mf::sink::async_operation::type_t::start:
		case solids::lib::mf::sink::async_operation::type_t::restart:
			hr = QueueEvent(MEStreamSinkStarted, GUID_NULL, hr, NULL); // Send MEStreamSinkStarted.
			if (SUCCEEDED(hr))
			{
				hr = QueueEvent(MEStreamSinkRequestSample, GUID_NULL, hr, NULL);
				//cap_log4cplus_logger::make_debug_log("amadeus.player", "[ Renderer ] SAMPLE REQUEST");
				if (SUCCEEDED(hr))
					_noutstanding_sample_requests++;
			}
			if (SUCCEEDED(hr))
				hr = process_samples_from_queue(_consume_data);

			break;
		case solids::lib::mf::sink::async_operation::type_t::stop:
			_renderer->SetFullscreen(FALSE);
			Flush();

			_noutstanding_sample_requests = 0;
			hr = QueueEvent(MEStreamSinkStopped, GUID_NULL, hr, NULL);

			break;

		case solids::lib::mf::sink::async_operation::type_t::pause:
			hr = QueueEvent(MEStreamSinkPaused, GUID_NULL, hr, NULL);
			break;

		case solids::lib::mf::sink::async_operation::type_t::process_sample:
		case solids::lib::mf::sink::async_operation::type_t::place_marker:
			if (!(_waiting_for_clock_start))
				hr = dispatch_process_sample(aop);
			break;
		}
	}

	solids::lib::mf::safe_release(state);
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::process_samples_from_queue(int32_t consume_state)
{
	HRESULT hr = S_OK;
	IUnknown * unk = NULL;
	BOOL process_more_samples = TRUE;
	BOOL device_changed = FALSE;
	BOOL process_again = FALSE;
	while (_samples_to_process.dequeue(&unk) == S_OK)
	{
		process_more_samples = TRUE;
		IMarker * pmarker = NULL;
		IMFSample * sample = NULL;
		IMFSample * outsample = NULL;

		hr = unk->QueryInterface(__uuidof(IMarker), (void**)&pmarker);
		if (hr == E_NOINTERFACE)
		{
			hr = unk->QueryInterface(IID_IMFSample, (void**)&sample);
		}

		if (SUCCEEDED(hr))
		{
			if (pmarker)
				hr = send_marker_event(pmarker, consume_state);
			else
			{
				if (consume_state == consume_state_t::process_samples)
				{

					hr = _renderer->process_samples(_current_type, sample, &_uninterlace_mode, &device_changed, &process_again, &outsample);

					if (SUCCEEDED(hr))
					{
						if (device_changed)
							QueueEvent(MEStreamSinkDeviceChanged, GUID_NULL, S_OK, NULL);

						if (process_again)
							hr = _samples_to_process.push_back(sample);
					}

					if (SUCCEEDED(hr))
					{
						if (outsample)
						{
							hr = _scheduler->schedule_sample(outsample, (_state != state_t::started));
							process_more_samples = FALSE;
						}
					}
				}
			}
		}

		solids::lib::mf::safe_release(unk);
		solids::lib::mf::safe_release(pmarker);
		solids::lib::mf::safe_release(sample);
		solids::lib::mf::safe_release(outsample);

		if (!process_more_samples)
			break;
	}
	solids::lib::mf::safe_release(unk);
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::queue_async_operation(int32_t op)
{
	HRESULT hr = S_OK;
	solids::lib::mf::sink::async_operation * aop = new solids::lib::mf::sink::async_operation(op); // Created with ref count = 1
	if (!aop)
		hr = E_OUTOFMEMORY;

	if (SUCCEEDED(hr))
		hr = MFPutWorkItem(_work_queue_id, &_work_queue_cb, aop);

	solids::lib::mf::safe_release(aop);  // Releases ref count
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::request_sample(void)
{
	HRESULT hr = S_OK;
	while (need_more_samples())
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		hr = QueueEvent(MEStreamSinkRequestSample, GUID_NULL, S_OK, NULL);
		if (SUCCEEDED(hr))
			_noutstanding_sample_requests++;
	}
	return hr;
}
HRESULT solids::lib::mf::sink::video::multiview::stream::send_marker_event(IMarker* marker, int32_t consume_state)
{
	HRESULT hr = S_OK;
	HRESULT status = S_OK;  // Status code for marker event.

	PROPVARIANT var;
	PropVariantInit(&var);

	do
	{
		if (consume_state == consume_state_t::drop_samples)
			status = E_ABORT;

		hr = marker->GetContext(&var);
		if (SUCCEEDED(hr))
			hr = QueueEvent(MEStreamSinkMarker, GUID_NULL, status, &var);

	} while (FALSE);

	PropVariantClear(&var);
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::stream::validate_operation(int32_t op)
{
    HRESULT hr = S_OK;
    BOOL transition = _valid_state_mat[_state][op];
    if (transition)
        return S_OK;
    else
        return MF_E_INVALIDREQUEST;
}
