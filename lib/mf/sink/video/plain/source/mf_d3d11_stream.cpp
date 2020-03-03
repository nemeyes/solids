//
#include "mf_d3d11_stream.h"

GUID const* const sld::lib::mf::sink::video::plain::stream::_video_formats[] =
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
    &MFVideoFormat_v410,
    &MFVideoFormat_I420,
    &MFVideoFormat_NV11,
    &MFVideoFormat_420O
};

const DWORD sld::lib::mf::sink::video::plain::stream::_nvideo_formats = sizeof(sld::lib::mf::sink::video::plain::stream::_video_formats) / sizeof(sld::lib::mf::sink::video::plain::stream::_video_formats[0]);
const MFRatio sld::lib::mf::sink::video::plain::stream::_default_fps = { 60, 1 };
const sld::lib::mf::sink::video::plain::stream::format_entry_t sld::lib::mf::sink::video::plain::stream::_dxgi_format_mapping[] =
{
    { MFVideoFormat_RGB32,      DXGI_FORMAT_B8G8R8X8_UNORM },
    { MFVideoFormat_ARGB32,     DXGI_FORMAT_R8G8B8A8_UNORM },
    { MFVideoFormat_AYUV,      DXGI_FORMAT_AYUV            },
    { MFVideoFormat_YUY2,      DXGI_FORMAT_YUY2            },
    { MFVideoFormat_NV12,      DXGI_FORMAT_NV12            },
    { MFVideoFormat_NV11,      DXGI_FORMAT_NV11            },
    { MFVideoFormat_AI44,      DXGI_FORMAT_AI44            },
    { MFVideoFormat_P010,      DXGI_FORMAT_P010            },
    { MFVideoFormat_P016,      DXGI_FORMAT_P016            },
    { MFVideoFormat_Y210,      DXGI_FORMAT_Y210            },
    { MFVideoFormat_Y216,      DXGI_FORMAT_Y216            },
    { MFVideoFormat_Y410,      DXGI_FORMAT_Y410            },
    { MFVideoFormat_Y416,      DXGI_FORMAT_Y416            },
    { MFVideoFormat_420O,      DXGI_FORMAT_420_OPAQUE      }
};

#define SAMPLE_QUEUE_HIWATER_THRESHOLD 3
#define MAX_PAST_FRAMES         3

BOOL sld::lib::mf::sink::video::plain::stream::_valid_state_mat[sld::lib::mf::sink::video::plain::stream::state_t::count][sld::lib::mf::sink::async_operation::type_t::count] =
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

sld::lib::mf::sink::video::plain::stream::stream(DWORD dwStreamId, sld::lib::mf::critical_section * cs, sld::lib::mf::scheduler * sched)
    : _stream_id(dwStreamId)
    , _lock(cs)
    , _state(sld::lib::mf::sink::video::plain::stream::state_t::type_not_set)
    , _is_shutdown(FALSE)
    , _work_queue_id(0)
    , _work_queue_cb(this, &sld::lib::mf::sink::video::plain::stream::dispatch_workitem_cb)
    , _consume_data(sld::lib::mf::sink::video::plain::stream::consume_state_t::process_samples)
    , _start_time(0)
    , _nwritten(0)
    , _noutstanding_sample_requests(0)
    , _media(NULL)
    , _event_queue(NULL)
    , _byte_stream(NULL)
    , _renderer(NULL)
    , _scheduler(sched)
    , _current_type(NULL)
    , _prerolling(FALSE)
    , _waiting_for_on_clock_start(FALSE)
    , _samples_to_process()
    , _uninterlace_mode(MFVideoInterlace_Progressive)
    , _image_bytes_pp()
    , _dxgi_format(DXGI_FORMAT::DXGI_FORMAT_UNKNOWN)
{
    _image_bytes_pp.numerator = 1;
    _image_bytes_pp.denominator = 1;
}

#pragma warning( pop )

//-------------------------------------------------------------------
// sld::lib::mf::sink::video::plain::stream destructor
//-------------------------------------------------------------------

sld::lib::mf::sink::video::plain::stream::~stream(void)
{
}

// IUnknown methods

ULONG sld::lib::mf::sink::video::plain::stream::AddRef(void)
{
    return sld::lib::mf::refcount_object::AddRef();
}

HRESULT sld::lib::mf::sink::video::plain::stream::QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void** ppv)
{
    if (!ppv)
    {
        return E_POINTER;
    }
    if (iid == IID_IUnknown)
    {
        *ppv = static_cast<IUnknown*>(static_cast<IMFStreamSink*>(this));
    }
    else if (iid == __uuidof(IMFStreamSink))
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

ULONG  sld::lib::mf::sink::video::plain::stream::Release(void)
{
    return sld::lib::mf::refcount_object::Release();
}

/// IMFStreamSink methods
HRESULT sld::lib::mf::sink::video::plain::stream::Flush(void)
{
    sld::lib::mf::auto_lock lock(_lock);

    HRESULT hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        _consume_data = sld::lib::mf::sink::video::plain::stream::consume_state_t::drop_samples;
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

    _consume_data = sld::lib::mf::sink::video::plain::stream::consume_state_t::process_samples;

    return hr;
}

//-------------------------------------------------------------------
// Name: GetIdentifier
// Description: Returns the stream identifier.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::GetIdentifier(__RPC__out DWORD* pdwIdentifier)
{
    sld::lib::mf::auto_lock lock(_lock);

    if (pdwIdentifier == NULL)
    {
        return E_POINTER;
    }

    HRESULT hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        *pdwIdentifier = _stream_id;
    }

    return hr;
}

//-------------------------------------------------------------------
// Name: GetMediaSink
// Description: Returns the parent media sink.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::GetMediaSink(__RPC__deref_out_opt IMFMediaSink** ppMediaSink)
{
    sld::lib::mf::auto_lock lock(_lock);

    if (ppMediaSink == NULL)
    {
        return E_POINTER;
    }

    HRESULT hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        *ppMediaSink = _media;
        (*ppMediaSink)->AddRef();
    }

    return hr;

}

//-------------------------------------------------------------------
// Name: GetMediaTypeHandler
// Description: Returns a media type handler for this stream.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::GetMediaTypeHandler(__RPC__deref_out_opt IMFMediaTypeHandler** ppHandler)
{
    sld::lib::mf::auto_lock lock(_lock);

    if (ppHandler == NULL)
    {
        return E_POINTER;
    }

    HRESULT hr = check_shutdown();

    // This stream object acts as its own type handler, so we QI ourselves.
    if (SUCCEEDED(hr))
    {
        hr = this->QueryInterface(IID_IMFMediaTypeHandler, (void**)ppHandler);
    }

    return hr;
}

//-------------------------------------------------------------------
// Name: PlaceMarker
// Description: Receives a marker. [Asynchronous]
//
// Note: The client can call PlaceMarker at any time. In response,
//       we need to queue an MEStreamSinkMarker event, but not until
//       *after* we have processed all samples that we have received
//       up to this point.
//
//       Also, in general you might need to handle specific marker
//       types, although this sink does not.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::PlaceMarker(MFSTREAMSINK_MARKER_TYPE eMarkerType, __RPC__in const PROPVARIANT* pvarMarkerValue, __RPC__in const PROPVARIANT* pvarContextValue)
{

    sld::lib::mf::auto_lock lock(_lock);

    HRESULT hr = S_OK;
    IMarker* pMarker = NULL;
    hr = check_shutdown();
    if (SUCCEEDED(hr))
        hr = validate_operation(sld::lib::mf::sink::async_operation::type_t::place_marker);

    // Create a marker object and put it on the sample queue.
    if (SUCCEEDED(hr))
        hr = sld::lib::mf::marker::create(eMarkerType, pvarMarkerValue, pvarContextValue, &pMarker);

    if (SUCCEEDED(hr))
        hr = _samples_to_process.queue(pMarker);

    // Unless we are paused, start an async operation to dispatch the next sample/marker.
    if (SUCCEEDED(hr))
    {
        if (_state != State_Paused)
        {
            // Queue the operation.
            hr = queue_async_operation(sld::lib::mf::sink::async_operation::type_t::place_marker); // Increments ref count on pOp.
        }
    }
    sld::lib::mf::safe_release(pMarker);

    return hr;
}

//-------------------------------------------------------------------
// Name: ProcessSample
// Description: Receives an input sample. [Asynchronous]
//
// Note: The client should only give us a sample after we send an
//       MEStreamSinkRequestSample event.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::ProcessSample(__RPC__in_opt IMFSample* pSample)
{
    sld::lib::mf::auto_lock lock(_lock);
    
    if (pSample == NULL)
    {
        return E_POINTER;
    }

    if (0 == _noutstanding_sample_requests)
    {
        return MF_E_INVALIDREQUEST;
    }

    HRESULT hr = S_OK;

    do
    {
        hr = check_shutdown();
        if (FAILED(hr))
        {
            break;
        }

        _noutstanding_sample_requests--;

        if (!_prerolling && !_waiting_for_on_clock_start)
        {
            // Validate the operation.
            hr = validate_operation(sld::lib::mf::sink::async_operation::type_t::process_sample);
            if (FAILED(hr))
            {
                break;
            }
        }

        // Add the sample to the sample queue.
        hr = _samples_to_process.queue(pSample);
        if (FAILED(hr))
        {
            break;
        }

        if (_prerolling)
        {
            _prerolling = FALSE;
            return QueueEvent(MEStreamSinkPrerolled, GUID_NULL, S_OK, NULL);
        }

        // Unless we are paused/stopped, start an async operation to dispatch the next sample.
        if (_state != State_Paused && _state != State_Stopped)
        {
            // Queue the operation.
            hr = queue_async_operation(sld::lib::mf::sink::async_operation::type_t::process_sample);
        }
    } while (FALSE);

    return hr;
}

// IMFMediaEventGenerator methods.
// Note: These methods call through to the event queue helper object.

HRESULT sld::lib::mf::sink::video::plain::stream::BeginGetEvent(IMFAsyncCallback* pCallback, IUnknown* punkState)
{
    HRESULT hr = S_OK;

    sld::lib::mf::auto_lock lock(_lock);
    hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        hr = _event_queue->BeginGetEvent(pCallback, punkState);
    }

    return hr;
}

HRESULT sld::lib::mf::sink::video::plain::stream::EndGetEvent(IMFAsyncResult* pResult, _Out_  IMFMediaEvent** ppEvent)
{
    HRESULT hr = S_OK;

    sld::lib::mf::auto_lock lock(_lock);
    hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        hr = _event_queue->EndGetEvent(pResult, ppEvent);
    }

    return hr;
}

HRESULT sld::lib::mf::sink::video::plain::stream::GetEvent(DWORD dwFlags, __RPC__deref_out_opt IMFMediaEvent** ppEvent)
{
    // NOTE:
    // GetEvent can block indefinitely, so we don't hold the lock.
    // This requires some juggling with the event queue pointer.

    HRESULT hr = S_OK;

    IMFMediaEventQueue* pQueue = NULL;

    { // scope for lock

        sld::lib::mf::auto_lock lock(_lock);

        // Check shutdown
        hr = check_shutdown();

        // Get the pointer to the event queue.
        if (SUCCEEDED(hr))
        {
            pQueue = _event_queue;
            pQueue->AddRef();
        }

    }   // release lock

    // Now get the event.
    if (SUCCEEDED(hr))
    {
        hr = pQueue->GetEvent(dwFlags, ppEvent);
    }

    sld::lib::mf::safe_release(pQueue);

    return hr;
}

HRESULT sld::lib::mf::sink::video::plain::stream::QueueEvent(MediaEventType met, __RPC__in REFGUID guidExtendedType, HRESULT hrStatus, __RPC__in_opt const PROPVARIANT* pvValue)
{
    HRESULT hr = S_OK;

    sld::lib::mf::auto_lock lock(_lock);
    hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        hr = _event_queue->QueueEventParamVar(met, guidExtendedType, hrStatus, pvValue);
    }

    return hr;
}

/// IMFMediaTypeHandler methods

//-------------------------------------------------------------------
// Name: GetCurrentMediaType
// Description: Return the current media type, if any.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::GetCurrentMediaType(_Outptr_ IMFMediaType** ppMediaType)
{
    sld::lib::mf::auto_lock lock(_lock);

    if (ppMediaType == NULL)
    {
        return E_POINTER;
    }

    HRESULT hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        if (_current_type == NULL)
        {
            hr = MF_E_NOT_INITIALIZED;
        }
    }

    if (SUCCEEDED(hr))
    {
        *ppMediaType = _current_type;
        (*ppMediaType)->AddRef();
    }

    return hr;
}

//-------------------------------------------------------------------
// Name: GetMajorType
// Description: Return the major type GUID.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::GetMajorType(__RPC__out GUID* pguidMajorType)
{
    if (pguidMajorType == NULL)
    {
        return E_POINTER;
    }

    HRESULT hr = check_shutdown();
    if (FAILED(hr))
    {
        return hr;
    }

    if (_current_type == NULL)
    {
        return MF_E_NOT_INITIALIZED;
    }

    return _current_type->GetGUID(MF_MT_MAJOR_TYPE, pguidMajorType);
}

//-------------------------------------------------------------------
// Name: GetMediaTypeByIndex
// Description: Return a preferred media type by index.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::GetMediaTypeByIndex(DWORD dwIndex, _Outptr_ IMFMediaType** ppType)
{
    HRESULT hr = S_OK;

    do
    {
        if (ppType == NULL)
        {
            hr = E_POINTER;
            break;
        }

        hr = check_shutdown();
        if (FAILED(hr))
        {
            break;
        }

        if (dwIndex >= _nvideo_formats)
        {
            hr = MF_E_NO_MORE_TYPES;
            break;
        }

        IMFMediaType* pVideoMediaType = NULL;

        do
        {
            hr = MFCreateMediaType(&pVideoMediaType);
            if (FAILED(hr))
            {
                break;
            }

            hr = pVideoMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
            if (FAILED(hr))
            {
                break;
            }

            hr = pVideoMediaType->SetGUID(MF_MT_SUBTYPE, *(_video_formats[dwIndex]));
            if (FAILED(hr))
            {
                break;
            }

            pVideoMediaType->AddRef();
            *ppType = pVideoMediaType;
        } while (FALSE);

        sld::lib::mf::safe_release(pVideoMediaType);

        if (FAILED(hr))
        {
            break;
        }
    } while (FALSE);

    return hr;
}

//-------------------------------------------------------------------
// Name: GetMediaTypeCount
// Description: Return the number of preferred media types.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::GetMediaTypeCount(__RPC__out DWORD* pdwTypeCount)
{
    HRESULT hr = S_OK;

    do
    {
        if (pdwTypeCount == NULL)
        {
            hr = E_POINTER;
            break;
        }

        hr = check_shutdown();
        if (FAILED(hr))
        {
            break;
        }

        *pdwTypeCount = _nvideo_formats;
    } while (FALSE);

    return hr;
}

//-------------------------------------------------------------------
// Name: IsMediaTypeSupported
// Description: Check if a media type is supported.
//
// pMediaType: The media type to check.
// ppMediaType: Optionally, receives a "close match" media type.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::IsMediaTypeSupported(IMFMediaType* pMediaType, _Outptr_opt_result_maybenull_ IMFMediaType** ppMediaType)
{
    HRESULT hr = S_OK;
    GUID subType = GUID_NULL;

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

        hr = pMediaType->GetGUID(MF_MT_SUBTYPE, &subType);
        if (FAILED(hr))
        {
            break;
        }

        hr = MF_E_INVALIDMEDIATYPE; // This will be set to OK if we find the subtype is accepted

        for (DWORD i = 0; i < _nvideo_formats; i++)
        {
            if (subType == (*_video_formats[i]))
            {
                hr = S_OK;
                break;
            }
        }

        if (FAILED(hr))
        {
            break;
        }

        for (DWORD i = 0; i < ARRAYSIZE(_dxgi_format_mapping); i++)
        {
            const sld::lib::mf::sink::video::plain::stream::format_entry_t & e = _dxgi_format_mapping[i];
            if (e.sub_type == subType)
            {
                _dxgi_format = e.dxgi_format;
                break;
            }
        }

        hr = _renderer->is_media_type_supported(pMediaType, _dxgi_format);
        if (FAILED(hr))
        {
            break;
        }
    } while (FALSE);

    // We don't return any "close match" types.
    if (ppMediaType)
    {
        *ppMediaType = NULL;
    }

    return hr;
}

//-------------------------------------------------------------------
// Name: SetCurrentMediaType
// Description: Set the current media type.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::SetCurrentMediaType(IMFMediaType* pMediaType)
{
    if (pMediaType == NULL)
    {
        return E_POINTER;
    }

    HRESULT hr = S_OK;
    MFRatio fps = { 0, 0 };
    GUID guidSubtype = GUID_NULL;

    sld::lib::mf::auto_lock lock(_lock);

    do
    {
        hr = check_shutdown();
        if (FAILED(hr))
        {
            break;
        }

        hr = validate_operation(sld::lib::mf::sink::async_operation::type_t::set_media_type);
        if (FAILED(hr))
        {
            break;
        }

        hr = IsMediaTypeSupported(pMediaType, NULL);
        if (FAILED(hr))
        {
            break;
        }

        sld::lib::mf::safe_release(_current_type);
        _current_type = pMediaType;
        _current_type->AddRef();

        pMediaType->GetGUID(MF_MT_SUBTYPE, &guidSubtype);

        if ((guidSubtype == MFVideoFormat_NV12) ||
            (guidSubtype == MFVideoFormat_YV12) ||
            (guidSubtype == MFVideoFormat_IYUV) ||
            (guidSubtype == MFVideoFormat_YVU9) ||
            (guidSubtype == MFVideoFormat_I420))
        {
            _image_bytes_pp.numerator = 3;
            _image_bytes_pp.denominator = 2;
        }
        else if ((guidSubtype == MFVideoFormat_YUY2) ||
            (guidSubtype == MFVideoFormat_RGB555) ||
            (guidSubtype == MFVideoFormat_RGB565) ||
            (guidSubtype == MFVideoFormat_UYVY) ||
            (guidSubtype == MFVideoFormat_YVYU))
        {
            _image_bytes_pp.numerator = 2;
            _image_bytes_pp.denominator = 1;
        }
        else if (guidSubtype == MFVideoFormat_RGB24)
        {
            _image_bytes_pp.numerator = 3;
            _image_bytes_pp.denominator = 1;
        }
        else if (guidSubtype == MFVideoFormat_RGB32)
        {
            _image_bytes_pp.numerator = 4;
            _image_bytes_pp.denominator = 1;
        }
        else if (guidSubtype == MFVideoFormat_v410)
        {
            _image_bytes_pp.numerator = 5;
            _image_bytes_pp.denominator = 4;
        }
        else // includes:
            // MFVideoFormat_RGB8
            // MFVideoFormat_AYUV
            // MFVideoFormat_NV11
        {
            // This is just a fail-safe
            _image_bytes_pp.numerator = 1;
            _image_bytes_pp.denominator = 1;
        }

        pMediaType->GetUINT32(MF_MT_INTERLACE_MODE, &_uninterlace_mode);

        // Set the frame rate on the scheduler.
        if (SUCCEEDED(get_fps(pMediaType, &fps)) && (fps.Numerator != 0) && (fps.Denominator != 0))
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
            hr = _renderer->set_current_media_type(pMediaType);
            if (FAILED(hr))
                break;
        }

        if (sld::lib::mf::sink::video::plain::stream::state_t::started != _state && sld::lib::mf::sink::video::plain::stream::state_t::paused != _state)
        {
            _state = sld::lib::mf::sink::video::plain::stream::state_t::ready;
        }
        else
        {
            //Flush all current samples in the Queue as this is a format change
            hr = Flush();
        }
    } while (FALSE);

    return hr;
}

//-------------------------------------------------------------------------
// Name: GetService
// Description: IMFGetService
//-------------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::GetService(__RPC__in REFGUID guidService, __RPC__in REFIID riid, __RPC__deref_out_opt LPVOID* ppvObject)
{
    IMFGetService* pGetService = NULL;
    HRESULT hr = _media->QueryInterface(IID_PPV_ARGS(&pGetService));
    if (SUCCEEDED(hr))
    {
        hr = pGetService->GetService(guidService, riid, ppvObject);
    }
    sld::lib::mf::safe_release(pGetService);
    return hr;
}

//+-------------------------------------------------------------------------
//
//  Member:     PresentFrame
//
//  Synopsis:   Present the current outstanding frame in the DX queue
//
//--------------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::process(IMFSample* sample)
{
    HRESULT hr = S_OK;

    if (sld::lib::mf::sink::video::plain::stream::consume_state_t::drop_samples == _consume_data)
        return hr;

    sld::lib::mf::auto_lock lock(_lock);
    do
    {
        hr = check_shutdown();
        if (FAILED(hr))
        {
            break;
        }

        hr = _renderer->present();
        if (FAILED(hr))
        {
            break;
        }
    } while (FALSE);

    if (SUCCEEDED(hr))
    {
        // Unless we are paused/stopped, start an async operation to dispatch the next sample.
        if (_state != State_Paused && _state != State_Stopped)
        {
            // Queue the operation.
            hr = queue_async_operation(sld::lib::mf::sink::async_operation::type_t::process_sample);
        }
    }
    else
    {
        // We are in the middle of an asynchronous operation, so if something failed, send an error.
        hr = QueueEvent(MEError, GUID_NULL, hr, NULL);
    }

    return hr;
}

HRESULT sld::lib::mf::sink::video::plain::stream::get_max_rate(BOOL fThin, float* pflRate)
{
    HRESULT hr = S_OK;
    DWORD dwMonitorRefreshRate = 0;
    UINT32 unNumerator = 0;
    UINT32 unDenominator = 0;

    do
    {
        hr = _renderer->get_monitor_refresh_rate(&dwMonitorRefreshRate);
        if (FAILED(hr))
        {
            break;
        }

        if (_current_type == NULL)
        {
            hr = MF_E_INVALIDREQUEST;
            break;
        }

        if (fThin == TRUE)
        {
            *pflRate = FLT_MAX;
            break;
        }

        MFGetAttributeRatio(_current_type, MF_MT_FRAME_RATE, &unNumerator, &unDenominator);

        if (unNumerator == 0 || unDenominator == 0)
        {
            // We support anything.
            *pflRate = FLT_MAX;
        }
        else
        {
            if (MFVideoInterlace_FieldInterleavedUpperFirst == _uninterlace_mode ||
                MFVideoInterlace_FieldInterleavedLowerFirst == _uninterlace_mode ||
                MFVideoInterlace_FieldSingleUpper == _uninterlace_mode ||
                MFVideoInterlace_FieldSingleLower == _uninterlace_mode ||
                MFVideoInterlace_MixedInterlaceOrProgressive == _uninterlace_mode)
            {
                unNumerator *= 2;
            }

            //
            // Only support rates up to the refresh rate of the monitor.
            //
            *pflRate = (float)MulDiv(dwMonitorRefreshRate, unDenominator, unNumerator);
        }
    } while (FALSE);

    return hr;
}

//-------------------------------------------------------------------
// Name: initialize
// Description: Initializes the stream sink.
//
// Note: This method is called once when the media sink is first
//       initialized.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::initialize(IMFMediaSink* pParent, sld::lib::mf::sink::video::plain::renderer * presenter)
{
    HRESULT hr = S_OK;

    if (SUCCEEDED(hr))
    {
        hr = sld::lib::mf::attributes<IMFAttributes>::initialize();
    }

    // Create the event queue helper.
    if (SUCCEEDED(hr))
    {
        hr = MFCreateEventQueue(&_event_queue);
    }

    // Allocate a new work queue for async operations.
    if (SUCCEEDED(hr))
    {
        hr = MFAllocateWorkQueue(&_work_queue_id);
    }

    if (SUCCEEDED(hr))
    {
        _renderer = presenter;
        _renderer->AddRef();
    }

    if (SUCCEEDED(hr))
    {
        _media = pParent;
        _media->AddRef();
    }

    return hr;
}

//-------------------------------------------------------------------
// Name: release
// Description: Shuts down the stream sink.
//-------------------------------------------------------------------
HRESULT sld::lib::mf::sink::video::plain::stream::release(void)
{
    sld::lib::mf::auto_lock lock(_lock);

    _is_shutdown = TRUE;

    if (_event_queue)
    {
        _event_queue->Shutdown();
    }

    MFUnlockWorkQueue(_work_queue_id);

    _samples_to_process.clear();

    sld::lib::mf::safe_release(_media);
    sld::lib::mf::safe_release(_event_queue);
    sld::lib::mf::safe_release(_byte_stream);
    sld::lib::mf::safe_release(_renderer);
    sld::lib::mf::safe_release(_current_type);

    return MF_E_SHUTDOWN;
}

BOOL sld::lib::mf::sink::video::plain::stream::is_active(void) const
{
    return ((_state == sld::lib::mf::sink::video::plain::stream::state_t::started) || (_state == sld::lib::mf::sink::video::plain::stream::state_t::paused));
}

//-------------------------------------------------------------------
// Name: Pause
// Description: Called when the presentation clock pauses.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::pause(void)
{
    sld::lib::mf::auto_lock lock(_lock);
    HRESULT hr = validate_operation(sld::lib::mf::sink::async_operation::type_t::pause);
    if (SUCCEEDED(hr))
    {
        _state = State_Paused;
        hr = queue_async_operation(sld::lib::mf::sink::async_operation::type_t::pause);
    }

    return hr;
}

HRESULT sld::lib::mf::sink::video::plain::stream::preroll(void)
{
    HRESULT hr = S_OK;

    sld::lib::mf::auto_lock lock(_lock);

    hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        _prerolling = TRUE;
        _waiting_for_on_clock_start = TRUE;

        // Kick things off by requesting a sample...
        _noutstanding_sample_requests++;

        hr = QueueEvent(MEStreamSinkRequestSample, GUID_NULL, hr, NULL);
    }

    return hr;
}

//-------------------------------------------------------------------
// Name: Restart
// Description: Called when the presentation clock restarts.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::restart(void)
{
    sld::lib::mf::auto_lock lock(_lock);

    HRESULT hr = validate_operation(sld::lib::mf::sink::async_operation::type_t::restart);

    if (SUCCEEDED(hr))
    {
        _state = sld::lib::mf::sink::video::plain::stream::state_t::started;
        hr = queue_async_operation(sld::lib::mf::sink::async_operation::type_t::restart);
    }

    return hr;
}


//-------------------------------------------------------------------
// Name: Start
// Description: Called when the presentation clock starts.
// Note: Start time can be PRESENTATION_CURRENT_POSITION meaning
//       resume from the last current position.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::start(MFTIME start)
{
    sld::lib::mf::auto_lock lock(_lock);

    HRESULT hr = S_OK;

    do
    {
        hr = validate_operation(sld::lib::mf::sink::async_operation::type_t::start);
        if (FAILED(hr))
        {
            break;
        }

        if (start != PRESENTATION_CURRENT_POSITION)
        {
            // We're starting from a "new" position
            _start_time = start;        // Cache the start time.
        }

        _state = sld::lib::mf::sink::video::plain::stream::state_t::started;
        hr = queue_async_operation(sld::lib::mf::sink::async_operation::type_t::start);
    } while (FALSE);

    _waiting_for_on_clock_start = FALSE;

    return hr;
}

//-------------------------------------------------------------------
// Name: Stop
// Description: Called when the presentation clock stops.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::stop(void)
{
    sld::lib::mf::auto_lock lock(_lock);

    HRESULT hr = validate_operation(sld::lib::mf::sink::async_operation::type_t::stop);

    if (SUCCEEDED(hr))
    {
        _state = State_Stopped;
        hr = queue_async_operation(sld::lib::mf::sink::async_operation::type_t::stop);
    }

    return hr;
}

// private methods

//-------------------------------------------------------------------
// Name: DispatchProcessSample
// Description: Complete a ProcessSample or PlaceMarker request.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::dispatch_process_sample(sld::lib::mf::sink::async_operation* pOp)
{
    HRESULT hr = S_OK;
    assert(pOp != NULL);

    hr = check_shutdown();
    if (FAILED(hr))
    {
        return hr;
    }

    if (_renderer->can_process_next_sample())
    {
        hr = process_samples_from_queue(sld::lib::mf::sink::video::plain::stream::consume_state_t::process_samples);

        // Ask for another sample
        if (SUCCEEDED(hr))
        {
            if (pOp->op() == sld::lib::mf::sink::async_operation::type_t::process_sample)
            {
                hr = request_samples();
            }
        }

        // We are in the middle of an asynchronous operation, so if something failed, send an error.
        if (FAILED(hr))
        {
            hr = QueueEvent(MEError, GUID_NULL, hr, NULL);
        }
    }

    return hr;
}

HRESULT sld::lib::mf::sink::video::plain::stream::check_shutdown(void) const
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

inline HRESULT sld::lib::mf::sink::video::plain::stream::get_fps(IMFMediaType* pType, MFRatio* pRatio)
{
    return MFGetAttributeRatio(pType, MF_MT_FRAME_RATE, (UINT32*)&pRatio->Numerator, (UINT32*)&pRatio->Denominator);
}

//+-------------------------------------------------------------------------
//
//  Member:     NeedMoreSamples
//
//  Synopsis:   Returns true if the number of samples in flight
//              (queued + requested) is less than the max allowed
//
//--------------------------------------------------------------------------
BOOL sld::lib::mf::sink::video::plain::stream::need_more_samples(void)
{
    const DWORD cSamplesInFlight = _samples_to_process.get_count() + _noutstanding_sample_requests;

    return cSamplesInFlight < SAMPLE_QUEUE_HIWATER_THRESHOLD;
}

//-------------------------------------------------------------------
// Name: OnDispatchWorkItem
// Description: Callback for MFPutWorkItem.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::dispatch_workitem_cb(IMFAsyncResult* pAsyncResult)
{
    // Called by work queue thread. Need to hold the critical section.
    sld::lib::mf::auto_lock lock(_lock);

    HRESULT hr = check_shutdown();
    if (FAILED(hr))
    {
        return hr;
    }

    IUnknown* pState = NULL;

    hr = pAsyncResult->GetState(&pState);

    if (SUCCEEDED(hr))
    {
        // The state object is a CAsncOperation object.
        sld::lib::mf::sink::async_operation * pOp = (sld::lib::mf::sink::async_operation*)pState;

        int32_t op = pOp->op();
        switch (op)
        {
        case sld::lib::mf::sink::async_operation::type_t::start :
        case sld::lib::mf::sink::async_operation::type_t::restart:
            // Send MEStreamSinkStarted.
            hr = QueueEvent(MEStreamSinkStarted, GUID_NULL, hr, NULL);

            // Kick things off by requesting two samples...
            if (SUCCEEDED(hr))
            {
                _noutstanding_sample_requests++;
                hr = QueueEvent(MEStreamSinkRequestSample, GUID_NULL, hr, NULL);
            }

            // There might be samples queue from earlier (ie, while paused).
            if (SUCCEEDED(hr))
            {
                hr = process_samples_from_queue(_consume_data);
            }

            break;

        case sld::lib::mf::sink::async_operation::type_t::stop:

            _renderer->SetFullscreen(FALSE);

            // Drop samples from queue.
            Flush();

            _noutstanding_sample_requests = 0;

            // Send the event even if the previous call failed.
            hr = QueueEvent(MEStreamSinkStopped, GUID_NULL, hr, NULL);

            break;

        case sld::lib::mf::sink::async_operation::type_t::pause:
            hr = QueueEvent(MEStreamSinkPaused, GUID_NULL, hr, NULL);
            break;

        case sld::lib::mf::sink::async_operation::type_t::process_sample:
        case sld::lib::mf::sink::async_operation::type_t::place_marker:
            if (!(_waiting_for_on_clock_start))
            {
                hr = dispatch_process_sample(pOp);
            }
            break;
        }
    }
    sld::lib::mf::safe_release(pState);
    return hr;
}

//-------------------------------------------------------------------
// Name: process_samples_from_queue
// Description:
//
// Removes all of the samples and markers that are currently in the
// queue and processes them.
//
// If bConsumeData = DropFrames
//     For each marker, send an MEStreamSinkMarker event, with hr = E_ABORT.
//     For each sample, drop the sample.
//
// If bConsumeData = process_samples
//     For each marker, send an MEStreamSinkMarker event, with hr = S_OK.
//     For each sample, write the sample to the file.
//
// This method is called when we flush, stop, restart, receive a new
// sample, or receive a marker.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::process_samples_from_queue(int32_t consumeData)
{
    HRESULT hr = S_OK;
    IUnknown* pUnk = NULL;
    BOOL bProcessMoreSamples = TRUE;
    BOOL bDeviceChanged = FALSE;
    BOOL bProcessAgain = FALSE;

    // Enumerate all of the samples/markers in the queue.

    // Note: Dequeue returns S_FALSE when the queue is empty.
    while (_samples_to_process.dequeue(&pUnk) == S_OK)
    {
        bProcessMoreSamples = TRUE;
        IMarker* pMarker = NULL;
        IMFSample* pSample = NULL;
        IMFSample* pOutSample = NULL;

        // Figure out if this is a marker or a sample.

        hr = pUnk->QueryInterface(__uuidof(IMarker), (void**)&pMarker);
        if (hr == E_NOINTERFACE)
        {
            // If this is a sample, write it to the file.
            hr = pUnk->QueryInterface(IID_IMFSample, (void**)&pSample);
        }

        // Now handle the sample/marker appropriately.
        if (SUCCEEDED(hr))
        {
            if (pMarker)
            {
                hr = send_marker_event(pMarker, consumeData);
            }
            else
            {
                assert(pSample != NULL);    // Not a marker, must be a sample
                if (consumeData == sld::lib::mf::sink::video::plain::stream::consume_state_t::process_samples)
                {
                    hr = _renderer->process_sample(_current_type, pSample, &_uninterlace_mode, &bDeviceChanged, &bProcessAgain, &pOutSample);

                    if (SUCCEEDED(hr))
                    {
                        if (bDeviceChanged)
                        {
                            QueueEvent(MEStreamSinkDeviceChanged, GUID_NULL, S_OK, NULL);
                        }

                        if (bProcessAgain)
                        {
                            // The input sample is not used. Return it to the queue.
                            hr = _samples_to_process.push_back(pSample);
                        }
                    }

                    // If we are not actively playing, OR we are scrubbing (rate = 0) OR this is a
                    // repaint request, then we need to present the sample immediately. Otherwise,
                    // schedule it normally.
                    if (SUCCEEDED(hr))
                    {
                        if (pOutSample)
                        {
                            hr = _scheduler->schedule_sample(pOutSample, (sld::lib::mf::sink::video::plain::stream::state_t::started != _state));
                            bProcessMoreSamples = FALSE;
                        }
                    }
                }
            }
        }

        sld::lib::mf::safe_release(pUnk);
        sld::lib::mf::safe_release(pMarker);
        sld::lib::mf::safe_release(pSample);
        sld::lib::mf::safe_release(pOutSample);

        if (!bProcessMoreSamples)
        {
            break;
        }
    }       // while loop

    sld::lib::mf::safe_release(pUnk);

    return hr;
}

//-------------------------------------------------------------------
// Name: queue_async_operation
// Description: Puts an async operation on the work queue.
//-------------------------------------------------------------------
HRESULT sld::lib::mf::sink::video::plain::stream::queue_async_operation(int32_t op)
{
    HRESULT hr = S_OK;
    sld::lib::mf::sink::async_operation * pOp = new sld::lib::mf::sink::async_operation(op); // Created with ref count = 1
    if (pOp == NULL)
    {
        hr = E_OUTOFMEMORY;
    }

    if (SUCCEEDED(hr))
    {
        hr = MFPutWorkItem(_work_queue_id, &_work_queue_cb, pOp);
    }

    sld::lib::mf::safe_release(pOp);  // Releases ref count

    return hr;
}

//+-------------------------------------------------------------------------
//
//  Member:     RequestSamples
//
//  Synopsis:   Issue more sample requests if necessary.
//
//--------------------------------------------------------------------------
HRESULT sld::lib::mf::sink::video::plain::stream::request_samples(void)
{
    HRESULT hr = S_OK;

    while (need_more_samples())
    {
        hr = check_shutdown();
        if (FAILED(hr))
        {
            break;
        }

        _noutstanding_sample_requests++;

        hr = QueueEvent(MEStreamSinkRequestSample, GUID_NULL, S_OK, NULL);
    }

    return hr;
}

//-------------------------------------------------------------------
// Name: send_marker_event
// Description: Saned a marker event.
//
// pMarker: Pointer to our custom IMarker interface, which holds
//          the marker information.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::send_marker_event(IMarker* pMarker, int32_t consumeState)
{
    HRESULT hr = S_OK;
    HRESULT hrStatus = S_OK;  // Status code for marker event.

    PROPVARIANT var;
    PropVariantInit(&var);

    do
    {
        if (consumeState == sld::lib::mf::sink::video::plain::stream::consume_state_t::drop_samples)
        {
            hrStatus = E_ABORT;
        }

        // Get the context data.
        hr = pMarker->GetContext(&var);

        if (SUCCEEDED(hr))
        {
            hr = QueueEvent(MEStreamSinkMarker, GUID_NULL, hrStatus, &var);
        }
    } while (FALSE);

    PropVariantClear(&var);

    return hr;
}

//-------------------------------------------------------------------
// Name: validate_operation
// Description: Checks if an operation is valid in the current state.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::stream::validate_operation(int32_t op)
{
    HRESULT hr = S_OK;

    BOOL bTransitionAllowed = _valid_state_mat[_state][op];

    if (bTransitionAllowed)
    {
        return S_OK;
    }
    else
    {
        return MF_E_INVALIDREQUEST;
    }
}