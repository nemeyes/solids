#include "mf_d3d11_media.h"

sld::lib::mf::critical_section sld::lib::mf::sink::video::plain::media::_lock_stream_and_scheduler;

HRESULT sld::lib::mf::sink::video::plain::media::create_instance(_In_ REFIID iid, _COM_Outptr_ void** ppSink)
{
    if (ppSink == NULL)
    {
        return E_POINTER;
    }

    *ppSink = NULL;

    HRESULT hr = S_OK;
    sld::lib::mf::sink::video::plain::media * pSink = new sld::lib::mf::sink::video::plain::media();

    if (pSink == NULL)
    {
        hr = E_OUTOFMEMORY;
    }

    if (SUCCEEDED(hr))
    {
        hr = pSink->initialize();
    }

    if (SUCCEEDED(hr))
    {
        hr = pSink->QueryInterface(iid, ppSink);
    }

    sld::lib::mf::safe_release(pSink);

    return hr;
}

// IUnknown methods

ULONG sld::lib::mf::sink::video::plain::media::AddRef(void)
{
    return sld::lib::mf::refcount_object::AddRef();
}

HRESULT sld::lib::mf::sink::video::plain::media::QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void** ppv)
{
    if (!ppv)
    {
        return E_POINTER;
    }
    if (iid == IID_IUnknown)
    {
        *ppv = static_cast<IUnknown*>(static_cast<IMFMediaSink*>(this));
    }
    else if (iid == __uuidof(IMFMediaSink))
    {
        *ppv = static_cast<IMFMediaSink*>(this);
    }
    else if (iid == __uuidof(IMFClockStateSink))
    {
        *ppv = static_cast<IMFClockStateSink*>(this);
    }
    else if (iid == __uuidof(IMFGetService))
    {
        *ppv = static_cast<IMFGetService*>(this);
    }
    else if (iid == IID_IMFRateSupport)
    {
        *ppv = static_cast<IMFRateSupport*>(this);
    }
    else if (iid == IID_IMFMediaSinkPreroll)
    {
        *ppv = static_cast<IMFMediaSinkPreroll*>(this);
    }
    else
    {
        *ppv = NULL;
        return E_NOINTERFACE;
    }
    AddRef();
    return S_OK;
}

ULONG  sld::lib::mf::sink::video::plain::media::Release(void)
{
    return sld::lib::mf::refcount_object::Release();
}

///  IMFMediaSink methods.
HRESULT sld::lib::mf::sink::video::plain::media::AddStreamSink(DWORD dwStreamSinkIdentifier, __RPC__in_opt IMFMediaType* pMediaType, __RPC__deref_out_opt IMFStreamSink** ppStreamSink)
{
    return MF_E_STREAMSINKS_FIXED;
}

//-------------------------------------------------------------------
// Name: GetCharacteristics
// Description: Returns the characteristics flags.
//
// Note: This sink has a fixed number of streams.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::media::GetCharacteristics(__RPC__out DWORD* pdwCharacteristics)
{
    sld::lib::mf::auto_lock lock(&_lock);

    if (pdwCharacteristics == NULL)
    {
        return E_POINTER;
    }

    HRESULT hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        *pdwCharacteristics = MEDIASINK_FIXED_STREAMS | MEDIASINK_CAN_PREROLL;
    }

    return hr;
}

//-------------------------------------------------------------------
// Name: GetPresentationClock
// Description: Returns a pointer to the presentation clock.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::media::GetPresentationClock(__RPC__deref_out_opt IMFPresentationClock** ppPresentationClock)
{
    sld::lib::mf::auto_lock lock(&_lock);

    if (ppPresentationClock == NULL)
    {
        return E_POINTER;
    }

    HRESULT hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        if (_clock == NULL)
        {
            hr = MF_E_NO_CLOCK; // There is no presentation clock.
        }
        else
        {
            // Return the pointer to the caller.
            *ppPresentationClock = _clock;
            (*ppPresentationClock)->AddRef();
        }
    }

    return hr;
}

//-------------------------------------------------------------------
// Name: GetStreamSinkById
// Description: Retrieves a stream by ID.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::media::GetStreamSinkById(DWORD dwStreamSinkIdentifier, __RPC__deref_out_opt IMFStreamSink** ppStreamSink)
{
    sld::lib::mf::auto_lock lock(&_lock);

    if (ppStreamSink == NULL)
    {
        return E_POINTER;
    }

    // Fixed stream ID.
    if (dwStreamSinkIdentifier != _stream_id)
    {
        return MF_E_INVALIDSTREAMNUMBER;
    }

    HRESULT hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        *ppStreamSink = _stream;
        (*ppStreamSink)->AddRef();
    }

    return hr;
}

//-------------------------------------------------------------------
// Name: GetStreamSinkByIndex
// Description: Retrieves a stream by index.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::media::GetStreamSinkByIndex(DWORD dwIndex, __RPC__deref_out_opt IMFStreamSink** ppStreamSink)
{
    sld::lib::mf::auto_lock lock(&_lock);

    if (ppStreamSink == NULL)
    {
        return E_POINTER;
    }

    // Fixed stream: Index 0.
    if (dwIndex > 0)
    {
        return MF_E_INVALIDINDEX;
    }

    HRESULT hr = check_shutdown();
    if (SUCCEEDED(hr))
    {
        *ppStreamSink = _stream;
        (*ppStreamSink)->AddRef();
    }

    return hr;
}

//-------------------------------------------------------------------
// Name: GetStreamSinkCount
// Description: Returns the number of streams.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::media::GetStreamSinkCount(__RPC__out DWORD* pcStreamSinkCount)
{
    sld::lib::mf::auto_lock lock(&_lock);

    if (pcStreamSinkCount == NULL)
    {
        return E_POINTER;
    }

    HRESULT hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        *pcStreamSinkCount = 1;  // Fixed number of streams.
    }

    return hr;

}

//-------------------------------------------------------------------
// Name: RemoveStreamSink
// Description: Removes a stream from the sink.
//
// Note: This sink has a fixed number of streams, so this method
//       always returns MF_E_STREAMSINKS_FIXED.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::media::RemoveStreamSink(DWORD dwStreamSinkIdentifier)
{
    return MF_E_STREAMSINKS_FIXED;
}

//-------------------------------------------------------------------
// Name: SetPresentationClock
// Description: Sets the presentation clock.
//
// pPresentationClock: Pointer to the clock. Can be NULL.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::media::SetPresentationClock(__RPC__in_opt IMFPresentationClock* pPresentationClock)
{
    sld::lib::mf::auto_lock lock(&_lock);

    HRESULT hr = check_shutdown();

    // If we already have a clock, remove ourselves from that clock's
    // state notifications.
    if (SUCCEEDED(hr))
    {
        if (_clock)
        {
            hr = _clock->RemoveClockStateSink(this);
        }
    }

    // Register ourselves to get state notifications from the new clock.
    if (SUCCEEDED(hr))
    {
        if (pPresentationClock)
        {
            hr = pPresentationClock->AddClockStateSink(this);
        }
    }

    if (SUCCEEDED(hr))
    {
        // Release the pointer to the old clock.
        // Store the pointer to the new clock.

        sld::lib::mf::safe_release(_clock);
        _clock = pPresentationClock;
        if (_clock)
        {
            _clock->AddRef();
        }
    }

    return hr;
}

//-------------------------------------------------------------------
// Name: Shutdown
// Description: Releases resources held by the media sink.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::media::Shutdown(void)
{
    sld::lib::mf::auto_lock lock(&_lock);

    HRESULT hr = MF_E_SHUTDOWN;

    _is_shutdown = TRUE;

    if (_stream != NULL)
    {
        _stream->release();
    }

    if (_renderer != NULL)
    {
        _renderer->shutdown();
    }

    sld::lib::mf::safe_release(_clock);
    sld::lib::mf::safe_release(_stream);
    sld::lib::mf::safe_release(_renderer);

    if (_scheduler != NULL)
    {
        hr = _scheduler->stop();
    }

    sld::lib::mf::safe_release(_scheduler);

    return hr;
}

//-------------------------------------------------------------------
// Name: OnClockPause
// Description: Called when the presentation clock paused.
//
// Note: For an archive sink, the paused state is equivalent to the
//       running (started) state. We still accept data and archive it.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::media::OnClockPause(MFTIME hnsSystemTime)
{
    sld::lib::mf::auto_lock lock(&_lock);

    HRESULT hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        hr = _stream->pause();
    }

    return hr;
}

//-------------------------------------------------------------------
// Name: OnClockRestart
// Description: Called when the presentation clock restarts.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::media::OnClockRestart(MFTIME hnsSystemTime)
{
    sld::lib::mf::auto_lock lock(&_lock);

    HRESULT hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        hr = _stream->restart();
    }

    return hr;
}

//-------------------------------------------------------------------
// Name: OnClockSetRate
// Description: Called when the presentation clock's rate changes.
//
// Note: For a rateless sink, the clock rate is not important.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::media::OnClockSetRate(MFTIME hnsSystemTime, float flRate)
{
    if (_scheduler != NULL)
    {
        // Tell the scheduler about the new rate.
        _scheduler->set_clock_rate(flRate);
    }

    return S_OK;
}

//-------------------------------------------------------------------
// Name: OnClockStart
// Description: Called when the presentation clock starts.
//
// hnsSystemTime: System time when the clock started.
// llClockStartOffset: Starting presentatation time.
//
// Note: For an archive sink, we don't care about the system time.
//       But we need to cache the value of llClockStartOffset. This
//       gives us the earliest time stamp that we archive. If any
//       input samples have an earlier time stamp, we discard them.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::media::OnClockStart(MFTIME hnsSystemTime, LONGLONG llClockStartOffset)
{
    sld::lib::mf::auto_lock lock(&_lock);

    HRESULT hr = check_shutdown();
    if (FAILED(hr))
    {
        return hr;
    }

    // Check if the clock is already active (not stopped).
    // And if the clock position changes while the clock is active, it
    // is a seek request. We need to flush all pending samples.
    if (_stream->is_active() && llClockStartOffset != PRESENTATION_CURRENT_POSITION)
    {
        // This call blocks until the scheduler threads discards all scheduled samples.
        hr = _stream->Flush();
    }
    else
    {
        if (_scheduler != NULL)
        {
            // Start the scheduler thread.
            hr = _scheduler->start(_clock);
        }
    }

    if (SUCCEEDED(hr))
    {
        hr = _stream->start(llClockStartOffset);
    }

    return hr;
}

//-------------------------------------------------------------------
// Name: OnClockStop
// Description: Called when the presentation clock stops.
//
// Note: After this method is called, we stop accepting new data.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::media::OnClockStop(MFTIME hnsSystemTime)
{
    sld::lib::mf::auto_lock lock(&_lock);

    HRESULT hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        hr = _stream->stop();
    }

    if (SUCCEEDED(hr))
    {
        if (_scheduler != NULL)
        {
            // Stop the scheduler thread.
            hr = _scheduler->stop();
        }
    }

    return hr;
}

//-------------------------------------------------------------------------
// Name: GetService
// Description: IMFGetService
//-------------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::media::GetService(__RPC__in REFGUID guidService, __RPC__in REFIID riid, __RPC__deref_out_opt LPVOID* ppvObject)
{
    HRESULT hr = S_OK;

    if (guidService == MF_RATE_CONTROL_SERVICE)
    {
        hr = QueryInterface(riid, ppvObject);
    }
    else if (guidService == MR_VIDEO_RENDER_SERVICE)
    {
        hr = _renderer->QueryInterface(riid, ppvObject);
    }
    else if (guidService == MR_VIDEO_ACCELERATION_SERVICE)
    {
        hr = _renderer->GetService(guidService, riid, ppvObject);
    }
    else
    {
        hr = MF_E_UNSUPPORTED_SERVICE;
    }

    return hr;
}

STDMETHODIMP sld::lib::mf::sink::video::plain::media::GetFastestRate(
    MFRATE_DIRECTION eDirection,
    BOOL fThin,
    _Out_ float* pflRate
)
{
    HRESULT hr = S_OK;

    sld::lib::mf::auto_lock lock(&_lock);

    do
    {
        hr = check_shutdown();
        if (FAILED(hr))
        {
            break;
        }

        if (NULL == pflRate)
        {
            hr = E_POINTER;
            break;
        }

        float rate;

        hr = _stream->get_max_rate(fThin, &rate);
        if (FAILED(hr))
        {
            break;
        }

        if (MFRATE_FORWARD == eDirection)
        {
            *pflRate = rate;
        }
        else
        {
            *pflRate = -rate;
        }
    } while (FALSE);

    return hr;
}

//-------------------------------------------------------------------------
// Name: GetSlowestRate
// Description: IMFRateSupport
//-------------------------------------------------------------------------

STDMETHODIMP sld::lib::mf::sink::video::plain::media::GetSlowestRate(
    MFRATE_DIRECTION eDirection,
    BOOL fThin,
    _Out_ float* pflRate
)
{
    HRESULT hr = S_OK;

    sld::lib::mf::auto_lock lock(&_lock);

    do
    {
        hr = check_shutdown();
        if (FAILED(hr))
        {
            break;
        }

        if (NULL == pflRate)
        {
            hr = E_POINTER;
            break;
        }

        if (SUCCEEDED(hr))
        {
            //
            // We go as slow as you want!
            //
            *pflRate = 0;
        }
    } while (FALSE);

    return hr;
}

STDMETHODIMP sld::lib::mf::sink::video::plain::media::IsRateSupported(BOOL fThin, float flRate, __RPC__inout_opt float* pflNearestSupportedRate)
{
    HRESULT hr = S_OK;
    float flNearestSupportedRate = flRate;

    sld::lib::mf::auto_lock lock(&_lock);

    do
    {
        hr = check_shutdown();
        if (FAILED(hr))
        {
            break;
        }

        //
        // Only support rates up to the refresh rate of the monitor.
        // This check makes sense only if we're going to be receiving
        // all frames
        //
        if (!fThin)
        {
            float rate;

            hr = _stream->get_max_rate(fThin, &rate);
            if (FAILED(hr))
            {
                break;
            }

            if ((flRate > 0 && flRate > (float)rate) ||
                (flRate < 0 && flRate < -(float)rate))
            {
                hr = MF_E_UNSUPPORTED_RATE;
                flNearestSupportedRate = (flRate >= 0.0f) ? rate : -rate;

                break;
            }
        }
    } while (FALSE);

    if (NULL != pflNearestSupportedRate)
    {
        *pflNearestSupportedRate = flNearestSupportedRate;
    }

    return hr;
}

STDMETHODIMP sld::lib::mf::sink::video::plain::media::NotifyPreroll(MFTIME hnsUpcomingStartTime)
{
    HRESULT hr = S_OK;

    sld::lib::mf::auto_lock lock(&_lock);

    hr = check_shutdown();

    if (SUCCEEDED(hr))
    {
        hr = _stream->preroll();
    }

    return hr;
}

/// Private methods

//-------------------------------------------------------------------
// CMediaSink constructor.
//-------------------------------------------------------------------

sld::lib::mf::sink::video::plain::media::media(void)
    : _stream_id(1)
    , _lock()
    , _is_shutdown(FALSE)
    , _stream(NULL)
    , _clock(NULL)
    , _scheduler(NULL)
    , _renderer(NULL)
{
}

//-------------------------------------------------------------------
// CMediaSink destructor.
//-------------------------------------------------------------------

sld::lib::mf::sink::video::plain::media::~media(void)
{
}

HRESULT sld::lib::mf::sink::video::plain::media::check_shutdown(void) const
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

//-------------------------------------------------------------------
// Name: Initialize
// Description: Initializes the media sink.
//
// Note: This method is called once when the media sink is first
//       initialized.
//-------------------------------------------------------------------

HRESULT sld::lib::mf::sink::video::plain::media::initialize(void)
{
    HRESULT hr = S_OK;
    IMFMediaSink* pSink = NULL;

    do
    {
        _scheduler = new sld::lib::mf::scheduler(&_lock_stream_and_scheduler);
        if (_scheduler == NULL)
        {
            hr = E_OUTOFMEMORY;
            break;
        }

        _stream = new sld::lib::mf::sink::video::plain::stream(_stream_id, &_lock_stream_and_scheduler, _scheduler);
        if (_stream == NULL)
        {
            hr = E_OUTOFMEMORY;
            break;
        }

        _renderer = new sld::lib::mf::sink::video::plain::renderer();
        if (_renderer == NULL)
        {
            hr = E_OUTOFMEMORY;
            break;
        }

        hr = QueryInterface(IID_PPV_ARGS(&pSink));
        if (FAILED(hr))
        {
            break;
        }

        hr = _stream->initialize(pSink, _renderer);
        if (FAILED(hr))
        {
            break;
        }

        _scheduler->set_callback(static_cast<sld::lib::mf::scheduler_callback_t*>(_stream));
    } while (FALSE);

    if (FAILED(hr))
    {
        Shutdown();
    }

    sld::lib::mf::safe_release(pSink);

    return hr;
}
