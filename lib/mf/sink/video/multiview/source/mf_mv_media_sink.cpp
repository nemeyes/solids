#include <mf_base.h>
#include "mf_mv_media_sink.h"
#include "mf_mv_stream_sink.h"
#include "mf_mv_renderer.h"

const int32_t solids::lib::mf::sink::video::multiview::media::sink_stream_id = 1;
solids::lib::mf::critical_section solids::lib::mf::sink::video::multiview::media::_lock_streamsink_and_scheduler;

HRESULT solids::lib::mf::sink::video::multiview::media::create_instance(_In_ REFIID iid, _COM_Outptr_ void ** ppsink)
{
	if (!ppsink)
		return E_POINTER;

	*ppsink = NULL;

	HRESULT hr = S_OK;
	solids::lib::mf::sink::video::multiview::media * pmedia = new solids::lib::mf::sink::video::multiview::media();

	if (!pmedia)
		hr = E_OUTOFMEMORY;

	if (SUCCEEDED(hr))
		hr = pmedia->initialize();

	if (SUCCEEDED(hr))
		hr = pmedia->QueryInterface(iid, ppsink);

	solids::lib::mf::safe_release(pmedia);
	return hr;
}

ULONG solids::lib::mf::sink::video::multiview::media::AddRef(void)
{
	return solids::lib::mf::refcount_object::AddRef();
}

HRESULT solids::lib::mf::sink::video::multiview::media::QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void** ppv)
{
	if (!ppv)
		return E_POINTER;

	if (iid == IID_IUnknown)
		*ppv = static_cast<IUnknown*>(static_cast<IMFMediaSink*>(this));
	else if (iid == __uuidof(IMFMediaSink))
		*ppv = static_cast<IMFMediaSink*>(this);
	else if (iid == __uuidof(IMFClockStateSink))
		*ppv = static_cast<IMFClockStateSink*>(this);
	else if (iid == __uuidof(IMFGetService))
		*ppv = static_cast<IMFGetService*>(this);
	//else if (iid == IID_IMFRateSupport)
	//	*ppv = static_cast<IMFRateSupport*>(this);
	else if (iid == IID_IMFMediaSinkPreroll)
		*ppv = static_cast<IMFMediaSinkPreroll*>(this);
	else if (iid == __uuidof(IPresenter))
		*ppv = static_cast<IPresenter*>(this);
    else
    {
        *ppv = NULL;
        return E_NOINTERFACE;
    }
    AddRef();
    return S_OK;
}

ULONG solids::lib::mf::sink::video::multiview::media::Release(void)
{
	return solids::lib::mf::refcount_object::Release();
}

HRESULT solids::lib::mf::sink::video::multiview::media::AddStreamSink(DWORD dwStreamSinkIdentifier, __RPC__in_opt IMFMediaType* pMediaType, __RPC__deref_out_opt IMFStreamSink** ppStreamSink)
{
    return MF_E_STREAMSINKS_FIXED;
}

HRESULT solids::lib::mf::sink::video::multiview::media::GetCharacteristics(__RPC__out DWORD * characteristics)
{
	solids::lib::mf::auto_lock lock(&_lock);

	if (characteristics == NULL)
		return E_POINTER;

	HRESULT hr = check_shutdown();
	if (SUCCEEDED(hr))
		*characteristics = MEDIASINK_FIXED_STREAMS | MEDIASINK_CAN_PREROLL;

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::media::GetPresentationClock(__RPC__deref_out_opt IMFPresentationClock ** pppc)
{
	solids::lib::mf::auto_lock lock(&_lock);

	if (pppc == NULL)
		return E_POINTER;

	HRESULT hr = check_shutdown();
	if (SUCCEEDED(hr))
	{
		if (_clock == NULL)
		{
			hr = MF_E_NO_CLOCK;
		}
		else
		{
			*pppc = _clock;
			(*pppc)->AddRef();
		}
	}
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::media::GetStreamSinkById(DWORD ssid, __RPC__deref_out_opt IMFStreamSink ** ppss)
{
	solids::lib::mf::auto_lock lock(&_lock);

	if (ppss == NULL)
		return E_POINTER;

	if (ssid != _stream_id)
		return MF_E_INVALIDSTREAMNUMBER;

	HRESULT hr = check_shutdown();
	if (SUCCEEDED(hr))
	{
		*ppss = _stream;
		(*ppss)->AddRef();
	}
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::media::GetStreamSinkByIndex(DWORD index, __RPC__deref_out_opt IMFStreamSink ** ppss)
{
	solids::lib::mf::auto_lock lock(&_lock);

	if (ppss == NULL)
		return E_POINTER;

	if (index > 0)
		return MF_E_INVALIDINDEX;

	HRESULT hr = check_shutdown();
	if (SUCCEEDED(hr))
	{
		*ppss = _stream;
		(*ppss)->AddRef();
	}
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::media::GetStreamSinkCount(__RPC__out DWORD * ssc)
{
	solids::lib::mf::auto_lock lock(&_lock);

	if (!ssc)
		return E_POINTER;

	HRESULT hr = check_shutdown();
	if (SUCCEEDED(hr))
		*ssc = 1;  // Fixed number of streams.

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::media::RemoveStreamSink(DWORD dwStreamSinkIdentifier)
{
    return MF_E_STREAMSINKS_FIXED;
}

HRESULT solids::lib::mf::sink::video::multiview::media::SetPresentationClock(__RPC__in_opt IMFPresentationClock * pc)
{
	solids::lib::mf::auto_lock lock(&_lock);

	HRESULT hr = check_shutdown();
	if (SUCCEEDED(hr))
	{
		if (_clock)
		{
			hr = _clock->RemoveClockStateSink(this);
		}
	}

	if (SUCCEEDED(hr))
	{
		if (pc)
		{
			hr = pc->AddClockStateSink(this);
		}
	}

	if (SUCCEEDED(hr))
	{
		safe_release(_clock);
		_clock = pc;
		if (_clock)
		{
			_clock->AddRef();
		}
	}
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::media::Shutdown(void)
{
	solids::lib::mf::auto_lock lock(&_lock);

	HRESULT hr = MF_E_SHUTDOWN;
	_is_shutdown = TRUE;

	if (_stream != NULL)
		_stream->release();

	if (_renderer != NULL)
		_renderer->release();

	safe_release(_clock);
	safe_release(_stream);
	safe_release(_renderer);

	if (_scheduler != NULL)
		hr = _scheduler->stop();

	safe_release(_scheduler);
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::media::OnClockPause(MFTIME st)
{
	solids::lib::mf::auto_lock lock(&_lock);

	HRESULT hr = check_shutdown();
	if (SUCCEEDED(hr))
		hr = _stream->pause();

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::media::OnClockRestart(MFTIME st)
{
	solids::lib::mf::auto_lock lock(&_lock);

	HRESULT hr = check_shutdown();
	if (SUCCEEDED(hr))
		hr = _stream->restart();

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::media::OnClockSetRate(MFTIME st, float rate)
{
	if (_scheduler)
		_scheduler->set_clock_rate(rate);

	return S_OK;
}

HRESULT solids::lib::mf::sink::video::multiview::media::OnClockStart(MFTIME st, LONGLONG offset)
{
	solids::lib::mf::auto_lock lock(&_lock);

	HRESULT hr = check_shutdown();
	if (FAILED(hr))
		return hr;

	if (_stream->is_active() && offset != PRESENTATION_CURRENT_POSITION)
	{
		hr = _stream->Flush();
	}
	else
	{
		if (_scheduler)
		{
			hr = _scheduler->start(_clock);
		}
	}

	if (SUCCEEDED(hr))
		hr = _stream->start(offset);

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::media::OnClockStop(MFTIME st)
{
	solids::lib::mf::auto_lock lock(&_lock);

	HRESULT hr = check_shutdown();
	if (SUCCEEDED(hr))
		hr = _stream->stop();

	if (SUCCEEDED(hr))
	{
		if (_scheduler != NULL)
		{
			hr = _scheduler->stop();
		}
	}

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::media::GetService(__RPC__in REFGUID guid_service, __RPC__in REFIID iid, __RPC__deref_out_opt LPVOID * ppv)
{
	HRESULT hr = S_OK;

	if (guid_service == MF_RATE_CONTROL_SERVICE)
		hr = QueryInterface(iid, ppv);
	else if (guid_service == MR_VIDEO_RENDER_SERVICE)
		hr = _renderer->QueryInterface(iid, ppv);
	else if (guid_service == MR_VIDEO_ACCELERATION_SERVICE)
		hr = _renderer->GetService(guid_service, iid, ppv);
	else
		hr = MF_E_UNSUPPORTED_SERVICE;

	return hr;
}
STDMETHODIMP solids::lib::mf::sink::video::multiview::media::NotifyPreroll(MFTIME hnsUpcomingStartTime)
{
	solids::lib::mf::auto_lock lock(&_lock);

	HRESULT hr = S_OK;
	hr = check_shutdown();
	if (SUCCEEDED(hr))
		hr = _stream->preroll();

	return hr;
}

STDMETHODIMP solids::lib::mf::sink::video::multiview::media::SetViewCount(INT count)
{
	if (_renderer)
		_renderer->set_view_count(count);
	return NOERROR;
}

STDMETHODIMP solids::lib::mf::sink::video::multiview::media::EnableCoordinatedCSConverter(BOOL enable)
{
	if (_renderer)
		_renderer->enable_coordinated_cs_converter(enable);
	return NOERROR;
}
STDMETHODIMP solids::lib::mf::sink::video::multiview::media::SetViewPosition(INT index, FLOAT* position)
{
	if (_renderer)
		_renderer->set_view_position(index, position);
	return NOERROR;
}
STDMETHODIMP solids::lib::mf::sink::video::multiview::media::SetSelected(INT index)
{
	if (_renderer)
		_renderer->set_selected(index);
	return NOERROR;
}

STDMETHODIMP solids::lib::mf::sink::video::multiview::media::Maximize(void)
{
	if (_renderer)
		_renderer->maximize();
	return NOERROR;
}
STDMETHODIMP solids::lib::mf::sink::video::multiview::media::ChangeRenderType(void)
{
	if (_renderer)
		_renderer->change_render_type();
	return NOERROR;
}

//-------------------------------------------------------------------
// CMediaSink constructor.
//-------------------------------------------------------------------
solids::lib::mf::sink::video::multiview::media::media(void)
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

solids::lib::mf::sink::video::multiview::media::~media(void)
{
}

HRESULT solids::lib::mf::sink::video::multiview::media::check_shutdown(void) const
{
    if (_is_shutdown)
        return MF_E_SHUTDOWN;
    else
        return S_OK;
}

HRESULT solids::lib::mf::sink::video::multiview::media::initialize(void)
{
	HRESULT hr = S_OK;
	IMFMediaSink * sink = NULL;

	do
	{
		_scheduler = new scheduler(&_lock_streamsink_and_scheduler);
		if (!_scheduler)
		{
			hr = E_OUTOFMEMORY;
			break;
		}

		_stream = new solids::lib::mf::sink::video::multiview::stream(_stream_id, _lock_streamsink_and_scheduler, _scheduler);
		if (!_stream)
		{
			hr = E_OUTOFMEMORY;
			break;
		}

		_renderer = new solids::lib::mf::sink::video::multiview::renderer(this);
		if (!_renderer)
		{
			hr = E_OUTOFMEMORY;
			break;
		}

		hr = QueryInterface(IID_PPV_ARGS(&sink));
		if (FAILED(hr))
			break;

		hr = _stream->initialize(sink, _renderer);
		if (FAILED(hr))
			break;

		_scheduler->set_callback(static_cast<scheduler_callback_t*>(_stream));
	} while (FALSE);

	if (FAILED(hr))
		Shutdown();

	solids::lib::mf::safe_release(sink);
	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::media::QueueEvent(MediaEventType met, __RPC__in REFGUID guidExtendedType, HRESULT hrStatus, __RPC__in_opt const PROPVARIANT* pvValue)
{
	return _stream->QueueEvent(met, guidExtendedType, hrStatus, pvValue);
}
