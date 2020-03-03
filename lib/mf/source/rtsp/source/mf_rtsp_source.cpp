#include "mf_rtsp_source.h"
#include "mf_rtsp_stream.h"

#include <wmcodecdsp.h>
#pragma comment(lib, "wmcodecdspuuid")

#include <sld_stringhelper.h>
#include <sld_locks.h>

#define MIN_VIDEO_BUFFER_COUNT	60
#define MIN_AUDIO_BUFFER_COUNT	60

#define MAX_VIDEO_BUFFER_COUNT	MIN_VIDEO_BUFFER_COUNT*100
#define MAX_AUDIO_BUFFER_COUNT	MIN_AUDIO_BUFFER_COUNT*100

sld::lib::mf::source::rtsp::source::source(void)
	: _work_queue_id(MFASYNC_CALLBACK_QUEUE_STANDARD)
	, _work_queue_cb(this, &sld::lib::mf::source::rtsp::source::dispatch_workitem_cb)
	, _state(sld::lib::mf::source::rtsp::source::state_t::invalid)
#if defined(WITH_FFRTSPCLIENT)
	, _video_codec(sld::lib::net::rtsp::ff::client::video_codec_t::unknown)
#else
	, _video_codec(sld::lib::net::rtsp::client::video_codec_t::unknown)
#endif
	, _video_wait_idr(FALSE)
	, _video_start_time(-1)
	, _intialized_pd(FALSE)
	, _extradata_size(0)
{
	::InitializeSRWLock(&_samples_lock);
}

sld::lib::mf::source::rtsp::source::~source(void)
{
	Shutdown();
}

HRESULT sld::lib::mf::source::rtsp::source::create_instance(sld::lib::mf::source::rtsp::source ** source)
{
	HRESULT hr = E_FAIL;
	do
	{
		sld::lib::mf::source::rtsp::source * psource = new sld::lib::mf::source::rtsp::source();
		if (psource == NULL)
			break;
		hr = psource->initialize();
		if (FAILED(hr))
			break;

		*source = psource;
		(*source)->AddRef();

		sld::lib::mf::safe_release(psource);

		hr = S_OK;

	} while (FALSE);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::QueryInterface(REFIID iid, void ** ppv)
{
	if (!ppv)
		return E_POINTER;
	if (iid == IID_IUnknown)
		*ppv = static_cast<IUnknown*>(this);
	else if (iid == __uuidof(IMFMediaSource))
		*ppv = static_cast<IMFMediaSource*>(this);
	else if (iid == __uuidof(IMFMediaEventGenerator))
		*ppv = static_cast<IMFMediaEventGenerator*>(this);
	else
	{
		*ppv = NULL;
		return E_NOINTERFACE;
	}
	AddRef();
	return S_OK;
}

ULONG sld::lib::mf::source::rtsp::source::AddRef(void)
{ 
	return sld::lib::mf::refcount_object::AddRef();
}

ULONG sld::lib::mf::source::rtsp::source::Release(void)
{ 
	return sld::lib::mf::refcount_object::Release();
}

HRESULT sld::lib::mf::source::rtsp::source::BeginGetEvent(IMFAsyncCallback * callback, IUnknown * state)
{
	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);
	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		hr = _event_queue->BeginGetEvent(callback, state);
		if (FAILED(hr))
			break;

	} while (FALSE);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::EndGetEvent(IMFAsyncResult * result, IMFMediaEvent ** evt)
{
	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);
	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		hr = _event_queue->EndGetEvent(result, evt);
		if (FAILED(hr))
			break;

	} while (FALSE);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::GetEvent(DWORD flags, IMFMediaEvent ** evt)
{
	// NOTE: 
	// GetEvent can block indefinitely, so we don't hold the critical 
	// section. Therefore we need to use a local copy of the event queue 
	// pointer, to make sure the pointer remains valid.
	HRESULT hr = S_OK;
	IMFMediaEventQueue * meq = NULL;

	do
	{
		sld::lib::mf::auto_lock mutex(&_lock);
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		meq = _event_queue;
		meq->AddRef();

	} while (FALSE);

	if (SUCCEEDED(hr))
		hr = meq->GetEvent(flags, evt);

	safe_release(meq);
	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::QueueEvent(MediaEventType met, REFGUID guid_ext_type, HRESULT status, const PROPVARIANT * value)
{
	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);
	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		hr = _event_queue->QueueEventParamVar(met, guid_ext_type, status, value);
	} while (FALSE);

	return hr;
}

//-------------------------------------------------------------------
// IMFMediaSource methods
//-------------------------------------------------------------------
HRESULT sld::lib::mf::source::rtsp::source::CreatePresentationDescriptor(IMFPresentationDescriptor ** pd)
{
	if (!pd)
		return E_POINTER;

	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);

	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		hr = is_initialized();
		if (FAILED(hr))
			break;

		if (!_pd)
		{
			hr = MF_E_NOT_INITIALIZED;
			break;
		}

		hr = _pd->Clone(pd);

	} while (FALSE);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::GetCharacteristics(DWORD * characteristics)
{
	if (!characteristics)
		return E_POINTER;

	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);

	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		*characteristics = MFMEDIASOURCE_IS_LIVE;
	} while (FALSE);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::Pause(void)
{
	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);

	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		hr = queue_async_operation(sld::lib::mf::source::rtsp::async_operation::type_t::pause);

	} while (FALSE);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::Shutdown(void)
{
	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);
	
#if defined(WITH_FFRTSPCLIENT)
	sld::lib::net::rtsp::ff::client::stop();
#else
	sld::lib::net::rtsp::client::stop();
#endif
	ATL::CComPtr<sld::lib::mf::source::rtsp::stream> stream = NULL;
	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		std::map<int32_t, ATL::CAdapt<ATL::CComPtr<sld::lib::mf::source::rtsp::stream>>>::iterator iter;
		for (iter = _streams.begin(); iter != _streams.end(); iter++)
		{
			stream = iter->second;
			(void)stream->release();
		}

		if (_event_queue)
			(void)_event_queue->Shutdown();

	} while (FALSE);


	_streams.clear();

	safe_release(_event_queue);
	
	_pd.Release();

	release_samples();

	_state = sld::lib::mf::source::rtsp::source::state_t::shutdown;

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::Start(IMFPresentationDescriptor * pd, const GUID * guid_time_fmt, const PROPVARIANT * position)
{
	HRESULT hr = S_OK;

	if (!pd)
		return E_INVALIDARG;
	if (!position)
		return E_INVALIDARG;
	if (guid_time_fmt && ((*guid_time_fmt) != GUID_NULL))
		return MF_E_UNSUPPORTED_TIME_FORMAT;
	if ((position->vt != VT_I8) && (position->vt != VT_EMPTY))
		return MF_E_UNSUPPORTED_TIME_FORMAT;
	if (position->vt == VT_I8)
	{
		if ((_state != sld::lib::mf::source::rtsp::source::state_t::stopped) || (position->hVal.QuadPart!=0))
			return MF_E_INVALIDREQUEST;
	}

	sld::lib::mf::auto_lock mutex(&_lock);
	sld::lib::mf::source::rtsp::async_operation  * aop = NULL;
	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;
		
		hr = is_initialized();
		if (FAILED(hr))
			break;

		hr = validate_presentation_descriptor(pd);
		if (FAILED(hr))
			break;

		aop = new sld::lib::mf::source::rtsp::start_async_operation(pd);
		aop->data(*position);
		hr = queue_async_operation(aop);

	} while (FALSE);

	safe_release(aop);
	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::Stop(void)
{
	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);

	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		hr = is_initialized();
		if (FAILED(hr))
			break;

		hr = queue_async_operation(sld::lib::mf::source::rtsp::async_operation::type_t::stop);

	} while (FALSE);
	
	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::begin_open(LPCWSTR url, IMFAsyncCallback * callback, IUnknown * state) 
{
	sld::lib::mf::auto_lock mutex(&_lock);

	if (!url)
		return E_POINTER;
	if (!callback)
		return E_POINTER;
	if (_state != sld::lib::mf::source::rtsp::source::state_t::invalid)
		return MF_E_INVALIDREQUEST;

	HRESULT hr = E_FAIL;
	do
	{
		char * ascii_url = NULL;
		sld::lib::stringhelper::convert_wide2multibyte((LPWSTR)url, &ascii_url);

		if (ascii_url)
		{
#if defined(WITH_FFRTSPCLIENT)
			sld::lib::net::rtsp::ff::client::play(ascii_url, sld::lib::net::rtsp::ff::client::transport_t::rtp_over_tcp, 10);
#else
			sld::lib::net::rtsp::client::play(ascii_url, NULL, NULL, sld::lib::net::rtsp::client::transport_option_t::rtp_over_tcp, sld::lib::net::rtsp::client::media_type_t::video | sld::lib::net::rtsp::client::media_type_t::audio, 10);
#endif
			free(ascii_url);
			ascii_url = NULL;
		}
		else
			break;
		
		hr = MFCreateAsyncResult(NULL, callback, state, &_begin_open_result);

		//_disable_audio = false;
		_state = sld::lib::mf::source::rtsp::source::state_t::opening;
	} while (FALSE);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::end_open(IMFAsyncResult * result) 
{
	sld::lib::mf::auto_lock mutex(&_lock);

	HRESULT hr = S_OK;
	hr = result->GetStatus();

	if (FAILED(hr)) 
	{
		// The source is not designed to recover after failing to open. Switch to shut-down state.
#if defined(WITH_FFRTSPCLIENT)
		sld::lib::net::rtsp::ff::client::stop();
#else
		sld::lib::net::rtsp::client::stop();
#endif
		Shutdown();
	}
	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::queue_async_operation(sld::lib::mf::source::rtsp::async_operation * aop)
{
	HRESULT hr = S_OK;
	if (!aop)
		hr = E_OUTOFMEMORY;

	if (SUCCEEDED(hr))
		hr = MFPutWorkItem(_work_queue_id, &_work_queue_cb, aop);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::queue_async_operation(int32_t op)
{
	HRESULT hr = S_OK;
	sld::lib::mf::source::rtsp::async_operation * aop= new sld::lib::mf::source::rtsp::async_operation(op);

	if (!aop)
		hr = E_OUTOFMEMORY;

	if (SUCCEEDED(hr))
		hr = MFPutWorkItem(_work_queue_id, &_work_queue_cb, aop);

	safe_release(aop);
	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::initialize(void)
{
	HRESULT hr = E_FAIL;
	hr = MFCreateEventQueue(&_event_queue);

	return hr;
}

/*
HRESULT sld::lib::mf::source::rtsp::source::release(void)
{
	if (_event_queue)
		_event_queue->Shutdown();

	_event_queue = NULL;

	return S_OK;
}
*/

HRESULT sld::lib::mf::source::rtsp::source::dispatch_workitem_cb(IMFAsyncResult * ar)
{
	sld::lib::mf::auto_lock mutex(&_lock);

	HRESULT hr = check_shutdown();
	if (FAILED(hr))
		return hr;

	IUnknown * state = NULL;
	hr = ar->GetState(&state);
	if (SUCCEEDED(hr))
	{
		sld::lib::mf::source::rtsp::async_operation * aop = (sld::lib::mf::source::rtsp::async_operation*)state;
		int32_t op = aop->op();
		switch (op)
		{
			case sld::lib::mf::source::rtsp::async_operation::type_t::start:
			{
				HRESULT hr = S_OK;
				IMFPresentationDescriptor * pd = NULL;
				BOOL sent_events = FALSE;

				sld::lib::mf::source::rtsp::start_async_operation * saop = static_cast<sld::lib::mf::source::rtsp::start_async_operation*>(aop);
				saop->presentation_descriptor(&pd);

				BOOL was_selected = FALSE;
				BOOL selected = FALSE;
				DWORD stream_id = 0;
				IMFStreamDescriptor * sd = NULL;
				ATL::CComPtr<sld::lib::mf::source::rtsp::stream> stream = NULL;

				IMFMediaEvent* evt = NULL;
				do
				{
					std::map<int32_t, ATL::CAdapt<ATL::CComPtr<sld::lib::mf::source::rtsp::stream>>>::iterator iter;
					int32_t index = 0;
					for (iter = _streams.begin(); iter != _streams.end(); iter++, index++)
					{
						hr = pd->GetStreamDescriptorByIndex(index, &selected, &sd);
						if (FAILED(hr))
							break;
						hr = sd->GetStreamIdentifier(&stream_id);
						if (FAILED(hr))
							break;
						stream = iter->second;
						if (!stream)
						{
							hr = E_INVALIDARG;
							break;
						}

						was_selected = stream->is_active();
						stream->activate(selected);
						if (selected)
						{
							if (was_selected)
							{
								hr = _event_queue->QueueEventParamUnk(MEUpdatedStream, GUID_NULL, hr, stream);
								if (FAILED(hr))
									break;
							}
							else
							{
								hr = _event_queue->QueueEventParamUnk(MENewStream, GUID_NULL, hr, stream);
								if (FAILED(hr))
									break;
							}
							hr = stream->start(saop->data());
							if (FAILED(hr))
								break;
						}
						safe_release(sd);
					}
					if (FAILED(hr))
					{
						safe_release(sd);
						break;
					}

					PROPVARIANT var;
					PropVariantInit(&var);
					var.vt = VT_I8;
					var.hVal.QuadPart = 0;
					//hr = MFCreateMediaEvent(MESourceStarted, GUID_NULL, hr, &var, &evt);
					hr = _event_queue->QueueEventParamVar(MESourceStarted, GUID_NULL, hr, NULL);
					if (FAILED(hr))
					{
						// Failure. Send the MESourceStarted or MESourceSeeked event with the error code. 
						// Note: It's possible that QueueEvent itself failed, in which case it is likely
						// to fail again. But there is no good way to recover in that case.
						(void)_event_queue->QueueEventParamVar(MESourceStarted, GUID_NULL, hr, NULL);
					}

					_state = sld::lib::mf::source::rtsp::source::state_t::started;
				} while (FALSE);

				safe_release(pd);
				safe_release(evt);
				break;
			}
			case sld::lib::mf::source::rtsp::async_operation::type_t::stop:
			{
				hr = S_OK;
				ATL::CComPtr<sld::lib::mf::source::rtsp::stream> stream = NULL;
				do
				{
					std::map<int32_t, ATL::CAdapt<ATL::CComPtr<sld::lib::mf::source::rtsp::stream>>>::iterator iter;
					for (iter = _streams.begin(); iter != _streams.end(); iter++)
					{
						stream = iter->second;
						if (stream->is_active())
							hr = stream->stop();

						if (FAILED(hr))
							break;
					}

					if (FAILED(hr))
						break;

					release_samples();
					_state = sld::lib::mf::source::rtsp::source::state_t::stopped;

				} while (FALSE);

				(void)_event_queue->QueueEventParamVar(MESourceStopped, GUID_NULL, S_OK, NULL);
				break;
			}
			case sld::lib::mf::source::rtsp::async_operation::type_t::pause:
			{
				hr = S_OK;
				ATL::CComPtr<sld::lib::mf::source::rtsp::stream> stream = NULL;

				do
				{
					if (_state != sld::lib::mf::source::rtsp::source::state_t::started)
					{
						hr = MF_E_INVALID_STATE_TRANSITION;
						break;
					}

					std::map<int32_t, ATL::CAdapt<ATL::CComPtr<sld::lib::mf::source::rtsp::stream>>>::iterator iter;
					for (iter = _streams.begin(); iter != _streams.end(); iter++)
					{
						stream = iter->second;
						if (stream->is_active())
							hr = stream->pause();

						if (FAILED(hr))
							break;
					}

					_state = sld::lib::mf::source::rtsp::source::state_t::paused;

				} while (FALSE);

				(void)_event_queue->QueueEventParamVar(MESourcePaused, GUID_NULL, hr, NULL);
				break;
			}
		}
	}

	safe_release(state);
	if (FAILED(hr))
		error_handle(hr);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::check_shutdown(void) const
{
	return _state == sld::lib::mf::source::rtsp::source::state_t::shutdown ? MF_E_SHUTDOWN : S_OK;
}

HRESULT sld::lib::mf::source::rtsp::source::is_initialized(void) const
{
	if (_state == sld::lib::mf::source::rtsp::source::state_t::opening || _state == sld::lib::mf::source::rtsp::source::state_t::invalid)
		return MF_E_NOT_INITIALIZED;
	else
		return S_OK;
}

HRESULT sld::lib::mf::source::rtsp::source::validate_presentation_descriptor(IMFPresentationDescriptor * pd)
{
	HRESULT hr = S_OK;
	BOOL selected = FALSE;
	DWORD nstreams = 0;
	IMFStreamDescriptor * sd = NULL;

	do
	{
		hr = pd->GetStreamDescriptorCount(&nstreams);
		if (FAILED(hr))
			break;

		for (DWORD i = 0; i < nstreams; i++)
		{
			hr = pd->GetStreamDescriptorByIndex(i, &selected, &sd);
			safe_release(sd);
			if (!selected)
				break;
		}

		if (!selected)
			hr = E_INVALIDARG;

	} while (FALSE);

	safe_release(sd);
	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::initialize_presentation_descriptor(void)
{
	HRESULT hr = S_OK;

	if (_state != sld::lib::mf::source::rtsp::source::state_t::opening)
		return hr;

	do
	{
		std::map<int32_t, ATL::CAdapt<ATL::CComPtr<sld::lib::mf::source::rtsp::stream>>>::iterator iter;
		iter = _streams.find(sld::lib::mf::source::rtsp::source::media_type_t::video);
		if (iter == _streams.end())
			return hr;
		DWORD nstream = DWORD(_streams.size());
		IMFStreamDescriptor ** ppsd = new (std::nothrow)IMFStreamDescriptor*[nstream];
		ZeroMemory(ppsd, nstream * sizeof(IMFStreamDescriptor*));

		DWORD index = 0;
		for (iter = _streams.begin(); iter != _streams.end(); iter++, index++)
		{
			ATL::CComPtr<sld::lib::mf::source::rtsp::stream> stream = iter->second;
			stream->GetStreamDescriptor(&ppsd[index]);
		}

		hr = MFCreatePresentationDescriptor(nstream, ppsd, &_pd);
		if (FAILED(hr))
			break;

		for (DWORD i = 0; i < nstream; i++)
		{
			hr = _pd->SelectStream(i);
			if (FAILED(hr))
				break;
		}

		if (ppsd)
		{
			for (index = 0; index < nstream; index++)
				safe_release(ppsd[index]);
			delete[] ppsd;
		}

		_state = sld::lib::mf::source::rtsp::source::state_t::stopped;

		hr = complete_open(S_OK);

	} while (0);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::create_audio_aac_mediatype(IMFMediaType ** mt, uint8_t* extradata, int32_t extradata_size, int32_t samplerate, int32_t sampleformat, int32_t channels)
{
	HRESULT hr = S_OK;

	ATL::CComPtr<IMFMediaType> pmt = NULL;
	do
	{
		hr = MFCreateMediaType(&pmt);
		if (FAILED(hr))
			break;

		hr = pmt->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
		if (FAILED(hr))
			break;

		hr = pmt->SetGUID(MF_MT_SUBTYPE, MEDIASUBTYPE_RAW_AAC1);
		if (FAILED(hr))
			break;

		hr = pmt->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, samplerate);
		if (FAILED(hr))
			break;

		hr = pmt->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, channels);
		if (FAILED(hr))
			break;

		switch (sampleformat)
		{
		case sld::lib::mf::source::rtsp::source::audio_sample_t::fmt_s16 :
			hr = pmt->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16);
			break;
		case sld::lib::mf::source::rtsp::source::audio_sample_t::fmt_s32 :
			hr = pmt->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 32);
			break;
		case sld::lib::mf::source::rtsp::source::audio_sample_t::fmt_s64:
			hr = pmt->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 64);
			break;
		case sld::lib::mf::source::rtsp::source::audio_sample_t::fmt_flt:
			hr = pmt->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 32);
			break;
		case sld::lib::mf::source::rtsp::source::audio_sample_t::fmt_dbl:
			hr = pmt->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 64);
			break;
		}
		if (FAILED(hr))
			break;

		if (extradata_size > 0)
		{
			hr = pmt->SetBlob(MF_MT_USER_DATA, extradata, extradata_size);
			if (FAILED(hr))
				break;
		}

		*mt = pmt;
		(*mt)->AddRef();

	} while (0);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::create_audio_mp3_mediatype(IMFMediaType ** mt, int32_t samplerate, int32_t sampleformat, int32_t channels)
{
	HRESULT hr = S_OK;
	ATL::CComPtr<IMFMediaType> pmt = NULL;
	do
	{
		hr = MFCreateMediaType(&pmt);
		if (FAILED(hr))
			break;

		hr = pmt->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
		if (FAILED(hr))
			break;

		hr = pmt->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_MP3);
		if (FAILED(hr))
			break;

		hr = pmt->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, samplerate);
		if (FAILED(hr))
			break;

		hr = pmt->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, channels);
		if (FAILED(hr))
			break;

		switch (sampleformat)
		{
		case sld::lib::mf::source::rtsp::source::audio_sample_t::fmt_s16:
			hr = pmt->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16);
			break;
		case sld::lib::mf::source::rtsp::source::audio_sample_t::fmt_s32:
			hr = pmt->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 32);
			break;
		case sld::lib::mf::source::rtsp::source::audio_sample_t::fmt_s64:
			hr = pmt->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 64);
			break;
		case sld::lib::mf::source::rtsp::source::audio_sample_t::fmt_flt:
			hr = pmt->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 32);
			break;
		case sld::lib::mf::source::rtsp::source::audio_sample_t::fmt_dbl:
			hr = pmt->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 64);
			break;
		}
		if (FAILED(hr))
			break;

		*mt = pmt;
		(*mt)->AddRef();

	} while (0);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::create_audio_ac3_mediatype(IMFMediaType ** mt, int32_t samplerate, int32_t sampleformat, int32_t channels)
{
	HRESULT hr = S_OK;
	ATL::CComPtr<IMFMediaType> pmt = NULL;
	do
	{
		hr = MFCreateMediaType(&pmt);
		if (FAILED(hr))
			break;

		hr = pmt->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
		if (FAILED(hr))
			break;

		hr = pmt->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_Dolby_AC3);
		if (FAILED(hr))
			break;

		hr = pmt->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, samplerate);
		if (FAILED(hr))
			break;

		hr = pmt->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, channels);
		if (FAILED(hr))
			break;

		switch (sampleformat)
		{
		case sld::lib::mf::source::rtsp::source::audio_sample_t::fmt_s16:
			hr = pmt->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16);
			break;
		case sld::lib::mf::source::rtsp::source::audio_sample_t::fmt_s32:
			hr = pmt->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 32);
			break;
		case sld::lib::mf::source::rtsp::source::audio_sample_t::fmt_s64:
			hr = pmt->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 64);
			break;
		case sld::lib::mf::source::rtsp::source::audio_sample_t::fmt_flt:
			hr = pmt->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 32);
			break;
		case sld::lib::mf::source::rtsp::source::audio_sample_t::fmt_dbl:
			hr = pmt->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 64);
			break;
		}
		if (FAILED(hr))
			break;

		*mt = pmt;
		(*mt)->AddRef();

	} while (0);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::create_video_h264_mediatype(IMFMediaType ** mt, uint8_t * extradata, int32_t extradata_size)
{
	HRESULT hr = S_OK;
	ATL::CComPtr<IMFMediaType> pmt = NULL;

	do
	{
		hr = MFCreateMediaType(&pmt);
		if (FAILED(hr))
			break;

		hr = pmt->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
		if (FAILED(hr))
			break;

		hr = pmt->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264_ES);
		if (FAILED(hr))
			break;

		hr = pmt->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
		if (FAILED(hr))
			break;

		hr = pmt->SetUINT32(MF_MT_SAMPLE_SIZE, 1);
		if (FAILED(hr))
			break;

		hr = pmt->SetUINT32(MF_MT_MPEG4_CURRENT_SAMPLE_ENTRY, 0);
		if (FAILED(hr))
			break;

		if (extradata_size > 0)
		{
			hr = pmt->SetBlob(MF_MT_MPEG_SEQUENCE_HEADER, extradata, extradata_size);
			if (FAILED(hr))
				break;
		}

		hr = MFSetAttributeSize(pmt, MF_MT_FRAME_SIZE, 3840, 2160);
		if (FAILED(hr))
			break;

		//hr = MFSetAttributeRatio(pmt, MF_MT_FRAME_RATE, (UINT32*)&pRatio->Numerator, (UINT32*)&pRatio->Denominator);
		/*
		if ((width > 0) && (height > 0))
		{
			hr = MFSetAttributeSize(pmt, MF_MT_FRAME_SIZE, (UINT32)width, (UINT32)height);
			if (FAILED(hr))
				break;
		}
		else
		{
			hr = E_FAIL;
			break;
		}
		*/

		*mt = pmt;
		(*mt)->AddRef();

	} while (0);

	return hr;
}


HRESULT sld::lib::mf::source::rtsp::source::create_video_hevc_mediatype(IMFMediaType ** mt, uint8_t * extradata, int32_t extradata_size)
{
	HRESULT hr = S_OK;
	ATL::CComPtr<IMFMediaType> pmt = NULL;

	do
	{
		hr = MFCreateMediaType(&pmt);
		if (FAILED(hr))
			break;

		hr = pmt->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
		if (FAILED(hr))
			break;

		hr = pmt->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_HEVC_ES);
		if (FAILED(hr))
			break;

		hr = pmt->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
		if (FAILED(hr))
			break;

		hr = pmt->SetUINT32(MF_MT_SAMPLE_SIZE, 1);
		if (FAILED(hr))
			break;

		hr = pmt->SetUINT32(MF_MT_MPEG4_CURRENT_SAMPLE_ENTRY, 0);
		if (FAILED(hr))
			break;

		if (extradata_size > 0)
		{
			hr = pmt->SetBlob(MF_MT_MPEG_SEQUENCE_HEADER, extradata, extradata_size);
			if (FAILED(hr))
				break;
		}


		hr = MFSetAttributeSize(pmt, MF_MT_FRAME_SIZE, 3840, 2160);
		if (FAILED(hr))
			break;
		/*
		if ((width > 0) && (height > 0))
		{
			hr = MFSetAttributeSize(pmt, MF_MT_FRAME_SIZE, (UINT32)width, (UINT32)height);
			if (FAILED(hr))
				break;
		}
		else
		{
			hr = E_FAIL;
			break;
		}
		*/

		*mt = pmt;
		(*mt)->AddRef();

	} while (0);
	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::create_video_mediatype(int32_t codec, uint8_t * extradata, int32_t extradata_size)
{
	HRESULT hr = S_OK;
	IMFMediaType * mt = NULL;
	ATL::CComPtr<IMFStreamDescriptor> sd = NULL;
	ATL::CComPtr<sld::lib::mf::source::rtsp::stream> stream = NULL;
	ATL::CComPtr<IMFMediaTypeHandler> handler = NULL;
	do
	{
		switch (codec)
		{
		case sld::lib::mf::source::rtsp::source::video_codec_t::avc:
			hr = create_video_h264_mediatype(&mt, extradata, extradata_size);
			break;
		case sld::lib::mf::source::rtsp::source::video_codec_t::hevc:
			hr = create_video_hevc_mediatype(&mt, extradata, extradata_size);
			break;
		}
		if (FAILED(hr))
			break;

		hr = MFCreateStreamDescriptor(sld::lib::mf::source::rtsp::source::media_type_t::video, 1, &mt, &sd);
		if (FAILED(hr))
			break;

		hr = sd->GetMediaTypeHandler(&handler);
		if (FAILED(hr))
			break;

		hr = handler->SetCurrentMediaType(mt);
		if (FAILED(hr))
			break;

		stream.Attach(new sld::lib::mf::source::rtsp::stream());
		hr = stream->initialize(this, sd, sld::lib::mf::source::rtsp::source::media_type_t::video);
		if (FAILED(hr))
			break;

		_streams.insert(std::make_pair(sld::lib::mf::source::rtsp::source::media_type_t::video, stream));

		//hr = initialize_presentation_descriptor();

	} while (0);


	safe_release(mt);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::create_audio_mediatype(int32_t codec, int32_t samplerate, int32_t sampleformat, int32_t channels, uint8_t * extradata, int32_t extradata_size)
{
	HRESULT hr = S_OK;

	IMFMediaType * mt = NULL;
	ATL::CComPtr<IMFStreamDescriptor> sd = NULL;
	ATL::CComPtr<sld::lib::mf::source::rtsp::stream> stream = NULL;
	ATL::CComPtr<IMFMediaTypeHandler> handler = NULL;
	do
	{
		switch (codec)
		{
		case sld::lib::mf::source::rtsp::source::audio_codec_t::aac:
			hr = create_audio_aac_mediatype(&mt, extradata, extradata_size, samplerate, sampleformat, channels);
			break;
		case sld::lib::mf::source::rtsp::source::audio_codec_t::mp3:
			hr = create_audio_mp3_mediatype(&mt, samplerate, sampleformat, channels);
			break;
		case sld::lib::mf::source::rtsp::source::audio_codec_t::ac3:
			hr = create_audio_mp3_mediatype(&mt, samplerate, sampleformat, channels);
			break;
		}
		if (FAILED(hr))
			break;

		hr = MFCreateStreamDescriptor(sld::lib::mf::source::rtsp::source::media_type_t::audio, 1, &mt, &sd);
		if (FAILED(hr))
			break;

		hr = sd->GetMediaTypeHandler(&handler);
		if (FAILED(hr))
			break;

		hr = handler->SetCurrentMediaType(mt);
		if (FAILED(hr))
			break;

		stream.Attach(new sld::lib::mf::source::rtsp::stream());
		hr = stream->initialize(this, sd, sld::lib::mf::source::rtsp::source::media_type_t::audio);
		if (FAILED(hr))
			break;

		_streams.insert(std::make_pair(sld::lib::mf::source::rtsp::source::media_type_t::audio, stream));

	} while (0);

	safe_release(mt);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::create_video_sample(IMFSample** sample, const uint8_t* extradata, int32_t extradata_size, const uint8_t* bytes, int32_t nbytes, long long pts, long long duration)
{
	HRESULT hr = S_OK;
	ATL::CComPtr<IMFSample> psample = NULL;
	ATL::CComPtr<IMFMediaBuffer> pbuffer = NULL;
	BYTE* pdata = NULL;

	do
	{
		hr = MFCreateMemoryBuffer(extradata_size + nbytes, &pbuffer);
		if (FAILED(hr))
			break;

		hr = pbuffer->Lock(&pdata, NULL, NULL);
		if (FAILED(hr))
			break;

		if(extradata && extradata_size>0)
			memmove(pdata, extradata, extradata_size);
		memmove(pdata + extradata_size, bytes, nbytes);

		hr = pbuffer->Unlock();
		if (FAILED(hr))
			break;

		hr = pbuffer->SetCurrentLength(nbytes);
		if (FAILED(hr))
			break;

		hr = MFCreateSample(&psample);
		if (FAILED(hr))
			break;

		hr = psample->AddBuffer(pbuffer);
		if (FAILED(hr))
			break;

		if (extradata && extradata_size > 0)
		{
			hr = psample->SetUINT32(MFSampleExtension_CleanPoint, TRUE);
			if (FAILED(hr))
				break;

			//hr = psample->SetUINT32(MFSampleExtension_Discontinuity, TRUE);
			//if (FAILED(hr))
			//	break;
		}

		/*
		if (_discontinuity)
		{
			hr = psample->SetUINT32(MFSampleExtension_Discontinuity, TRUE);
			if (FAILED(hr))
				break;

			_discontinuity = FALSE;
		}
		*/

		hr = psample->SetSampleTime(pts);
		if (FAILED(hr))
			break;

		hr = psample->SetSampleDuration(duration);
		if (FAILED(hr))
			break;

		* sample = psample;
		(*sample)->AddRef();

	} while (0);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::create_video_sample(IMFSample ** sample, const uint8_t * bytes, int32_t nbytes, long long pts, long long duration)
{
	HRESULT hr = S_OK;
	ATL::CComPtr<IMFSample> psample = NULL;
	ATL::CComPtr<IMFMediaBuffer> pbuffer = NULL;
	BYTE * pdata = NULL;

	do
	{
		hr = MFCreateMemoryBuffer(nbytes, &pbuffer);
		if (FAILED(hr))
			break;

		hr = pbuffer->Lock(&pdata, NULL, NULL);
		if (FAILED(hr))
			break;

		memmove(pdata, bytes, nbytes);

		hr = pbuffer->Unlock();
		if (FAILED(hr))
			break;

		hr = pbuffer->SetCurrentLength(nbytes);
		if (FAILED(hr))
			break;

		hr = MFCreateSample(&psample);
		if (FAILED(hr))
			break;

		hr = psample->AddBuffer(pbuffer);
		if (FAILED(hr))
			break;

		/*
		if (sps && sps_size > 0 && pps && pps_size > 0)
		{
			hr = psample->SetUINT32(MFSampleExtension_CleanPoint, TRUE);
			if (FAILED(hr))
				break;
		}
		*/

		/*
		if (_discontinuity)
		{
			hr = psample->SetUINT32(MFSampleExtension_Discontinuity, TRUE);
			if (FAILED(hr))
				break;

			_discontinuity = FALSE;
		}
		*/

		/*
		hr = psample->SetSampleTime(pts);
		if (FAILED(hr))
			break;
		*/
		
		hr = psample->SetSampleTime(pts);
		if (FAILED(hr))
			break;

		hr = psample->SetSampleDuration(duration);
		if (FAILED(hr))
			break;

		*sample = psample;
		(*sample)->AddRef();

	} while (0);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::create_audio_sample(IMFSample ** sample, const uint8_t * bytes, int32_t nbytes, long long pts, long long duration)
{
	HRESULT hr = S_OK;
	ATL::CComPtr<IMFSample> psample = NULL;
	ATL::CComPtr<IMFMediaBuffer> pbuffer = NULL;
	BYTE * pdata = NULL;

	do
	{
		hr = MFCreateMemoryBuffer(nbytes, &pbuffer);
		if (FAILED(hr))
			break;

		hr = pbuffer->Lock(&pdata, NULL, NULL);
		if (FAILED(hr))
			break;

		memmove(pdata, bytes, nbytes);

		hr = pbuffer->Unlock();
		if (FAILED(hr))
			break;

		pbuffer->SetCurrentLength(nbytes);
		if (FAILED(hr))
			break;

		hr = MFCreateSample(&psample);
		if (FAILED(hr))
			break;

		hr = psample->AddBuffer(pbuffer);
		if (FAILED(hr))
			break;

		//if (_start_time == 0)
		//{
		//	hr = psample->SetUINT32(MFSampleExtension_Discontinuity, TRUE);
		//	if (FAILED(hr))
		//		break;
		//}


		hr = psample->SetSampleTime(pts);
		if (FAILED(hr))
			break;

		hr = psample->SetSampleDuration(duration);
		if (FAILED(hr))
			break;
		/*
		LONGLONG last = _start_time;
		_start_time = 10000i64 * pts;
		hr = psample->SetSampleTime(_start_time);
		if (FAILED(hr))
			break;

		hr = psample->SetSampleDuration(_start_time - last);
		if (FAILED(hr))
			break;
		*/
		*sample = psample;
		(*sample)->AddRef();

	} while (0);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::complete_open(const HRESULT hr_status)
{
	HRESULT hr = S_OK;

	if (_begin_open_result)
	{
		do
		{
			hr = _begin_open_result->SetStatus(hr_status);
			if (FAILED(hr))
				break;

			hr = MFInvokeCallback(_begin_open_result);
			if (FAILED(hr))
				break;

		} while (FALSE);
	}

	_begin_open_result.Release();

	return hr;
}

void sld::lib::mf::source::rtsp::source::error_handle(const HRESULT hr)
{
	if (_state == sld::lib::mf::source::rtsp::source::state_t::opening)
	{
		complete_open(hr);
	}
	else if(_state!= sld::lib::mf::source::rtsp::source::state_t::shutdown)
	{
		QueueEvent(MEError, GUID_NULL, hr, NULL);
	}
}

void sld::lib::mf::source::rtsp::source::release_samples(void)
{
	sld::lib::exclusive_scopedlock mutex(&_samples_lock);

	std::map<int32_t, std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>>*>::iterator iter;
	for (iter = _samples.begin(); iter != _samples.end(); iter++)
	{
		std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>>* q = iter->second;
		if (q)
		{
			q->clear();
			delete q;
			q = nullptr;
		}
	}
	_samples.clear();
}

HRESULT sld::lib::mf::source::rtsp::source::get_video_sample(IMFSample ** sample)
{
	HRESULT hr = S_OK;

	ATL::CComPtr<sld::lib::mf::source::rtsp::stream> stream = NULL;
	{
		auto_lock lock(&_stream_lock);
		std::map<int32_t, ATL::CAdapt<ATL::CComPtr<sld::lib::mf::source::rtsp::stream>>>::iterator stream_iter;
		stream_iter = _streams.find(sld::lib::mf::source::rtsp::source::media_type_t::video);
		if (stream_iter != _streams.end())
			stream = stream_iter->second;
	}

	sld::lib::exclusive_scopedlock mutex(&_samples_lock);
	std::map<int32_t, std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>>*>::iterator iter;// _samples
	iter = _samples.find(sld::lib::mf::source::rtsp::source::media_type_t::video);
	if (iter != _samples.end())
	{
		std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>> * vq = iter->second;
		if (vq && vq->size()>0)
		{
			ATL::CComPtr<IMFSample> psample = vq->front();
			(*sample) = psample;
			(*sample)->AddRef();
			vq->erase(vq->begin());
		}
	}
	return hr;
}

HRESULT sld::lib::mf::source::rtsp::source::get_audio_sample(IMFSample ** sample)
{
	HRESULT hr = S_OK;
	ATL::CComPtr<sld::lib::mf::source::rtsp::stream> stream = NULL;
	{
		auto_lock lock(&_stream_lock);
		std::map<int32_t, ATL::CAdapt<ATL::CComPtr<sld::lib::mf::source::rtsp::stream>>>::iterator stream_iter;
		stream_iter = _streams.find(sld::lib::mf::source::rtsp::source::media_type_t::audio);
		if (stream_iter != _streams.end())
			stream = stream_iter->second;
	}

	sld::lib::exclusive_scopedlock mutex(&_samples_lock);
	std::map<int32_t, std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>>*>::iterator iter;
	iter = _samples.find(sld::lib::mf::source::rtsp::source::media_type_t::audio);
	if (iter != _samples.end())
	{
		std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>> * aq = iter->second;
		if (aq && aq->size()>0)
		{
			ATL::CComPtr<IMFSample> psample = aq->front();
			(*sample) = psample;
			(*sample)->AddRef();
			aq->erase(aq->begin());
		}

		/*
		if (stream && _state == sld::lib::mf::source::rtsp::source::state_t::started && !stream->is_buffering() && aq->size() <= MIN_AUDIO_BUFFER_COUNT)
		{
			hr = QueueEvent(MEBufferingStarted, GUID_NULL, hr, NULL);
			if (SUCCEEDED(hr))
				stream->set_buffering(TRUE);
		}
		*/
	}
	return hr;
}

void sld::lib::mf::source::rtsp::source::on_begin_video(int32_t codec, uint8_t* extradata, int32_t extradata_size)
{
	HRESULT hr = S_OK;
	IMFMediaType * pmt = NULL;
	IMFStreamDescriptor * psd = NULL;

	std::map<int32_t, ATL::CAdapt<ATL::CComPtr<sld::lib::mf::source::rtsp::stream>>>::iterator iter;
	iter = _streams.find(sld::lib::mf::source::rtsp::source::media_type_t::video);
	if (iter == _streams.end())
	{
		create_video_mediatype(codec, extradata, extradata_size);

		sld::lib::exclusive_scopedlock mutex(&_samples_lock);
		std::map<int32_t, std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>>*>::iterator sample_iter;
		sample_iter = _samples.find(sld::lib::mf::source::rtsp::source::media_type_t::video);
		if (sample_iter == _samples.end())
		{
			std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>>* vp = new std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>>();
			_samples.insert(std::make_pair(sld::lib::mf::source::rtsp::source::media_type_t::video, vp));
		}
	}

	if (extradata && extradata_size > 0)
	{
		_extradata_size = extradata_size;
		::memmove(_extradata, extradata, _extradata_size);
	}

	_video_codec = codec;
	_video_wait_idr = TRUE;
	_video_start_time = -1;
}

void sld::lib::mf::source::rtsp::source::on_recv_video(uint8_t* bytes, int32_t nbytes, long long pts, long long duration)
{
	HRESULT hr = S_OK;

	BOOL recvIDR = FALSE;
	//char debug[MAX_PATH] = { 0 };
	//_snprintf_s(debug, MAX_PATH, "on_recv_video : %lld\n", pts);
	//::OutputDebugStringA(debug);
	do
	{
		if (_video_codec == sld::lib::mf::source::rtsp::source::video_codec_t::avc)
		{
			//if ((bytes[4] & 0x1F) == 0x07)
			//	recvIDR = TRUE;
			//if ((bytes[4] & 0x1F) == 0x08)
			//	recvIDR = TRUE;
			if ((bytes[4] & 0x1F) == 0x05)
				recvIDR = TRUE;
		}
		else if (_video_codec == sld::lib::mf::source::rtsp::source::video_codec_t::hevc)
		{
			//if (((bytes[4] >> 1) & 0x3F) == 0x20)	// vps
			//	recvIDR = TRUE;
			//if (((bytes[4] >> 1) & 0x3F) == 0x21)	// sps
			//	recvIDR = TRUE;
			//if (((bytes[4] >> 1) & 0x3F) == 0x22)	// pps
			//	recvIDR = TRUE;
			if ((((bytes[4] >> 1) & 0x3F) == 0x13) || (((bytes[4] >> 1) & 0x3F) == 0x14))	// idr
				recvIDR = TRUE;
		}

		if (_video_wait_idr)
		{
			_video_start_time = pts;
			_video_wait_idr = FALSE;
		}
		
		ATL::CComPtr<IMFSample> sample = NULL;
		if(recvIDR)
			hr = create_video_sample(&sample, _extradata, _extradata_size, bytes, nbytes, pts - _video_start_time, duration);
		else
			hr = create_video_sample(&sample, bytes, nbytes, pts - _video_start_time, duration);
		if (FAILED(hr))
			return;

		ATL::CComPtr<sld::lib::mf::source::rtsp::stream> stream = NULL;
		{
			auto_lock lock(&_stream_lock);
			std::map<int32_t, ATL::CAdapt<ATL::CComPtr<sld::lib::mf::source::rtsp::stream>>>::iterator stream_iter;
			stream_iter = _streams.find(sld::lib::mf::source::rtsp::source::media_type_t::video);
			if (stream_iter != _streams.end())
				stream = stream_iter->second;
		}

		{
			sld::lib::exclusive_scopedlock mutex(&_samples_lock);
			std::map<int32_t, std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>>*>::iterator iter;// _samples
			iter = _samples.find(sld::lib::mf::source::rtsp::source::media_type_t::video);
			if (iter != _samples.end())
			{
				std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>>* vq = iter->second;
				if (vq)
				{
					vq->push_back(sample);
					if (_state == sld::lib::mf::source::rtsp::source::state_t::started)
					{
						if (stream->is_buffering())
						{
							if (vq->size() >= MIN_VIDEO_BUFFER_COUNT)
							{
								hr = QueueEvent(MEBufferingStopped, GUID_NULL, hr, NULL);
								if (SUCCEEDED(hr))
									stream->set_buffering(FALSE);
							}
						}
						else
						{
							if (vq->size() < 2)
							{
								hr = QueueEvent(MEBufferingStarted, GUID_NULL, hr, NULL);
								if (SUCCEEDED(hr))
									stream->set_buffering(TRUE);
							}
						}
					}
				}
			}
		}

		initialize_presentation_descriptor();

	} while (0);
}

void sld::lib::mf::source::rtsp::source::on_end_video(void)
{

}

void sld::lib::mf::source::rtsp::source::on_begin_audio(int32_t codec, uint8_t* extradata, int32_t extradata_size, int32_t samplerate, int32_t channels)
{
	HRESULT hr = S_OK;
	{
		std::map<int32_t, ATL::CAdapt<ATL::CComPtr<sld::lib::mf::source::rtsp::stream>>>::iterator iter;
		iter = _streams.find(sld::lib::mf::source::rtsp::source::media_type_t::audio);
		if (iter == _streams.end())
		{
			create_audio_mediatype(codec, samplerate, sld::lib::mf::source::rtsp::source::audio_sample_t::fmt_flt, channels, extradata, extradata_size);
			sld::lib::exclusive_scopedlock mutex(&_samples_lock);
			std::map<int32_t, std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>>*>::iterator sample_iter;
			sample_iter = _samples.find(sld::lib::mf::source::rtsp::source::media_type_t::audio);
			if (sample_iter == _samples.end())
			{
				std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>>* ap = new std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>>();
				_samples.insert(std::make_pair(sld::lib::mf::source::rtsp::source::media_type_t::audio, ap));
			}
		}
	}
}

void sld::lib::mf::source::rtsp::source::on_recv_audio(uint8_t* bytes, int32_t nbytes, long long pts, long long duration)
{
	HRESULT hr = S_OK;
	//char debug[MAX_PATH] = { 0 };
	//_snprintf_s(debug, MAX_PATH, "on_recv_audio : %lld[%lld]\n", pts, duration);
	//::OutputDebugStringA(debug);
	do
	{
		ATL::CComPtr<IMFSample> sample = NULL;
		hr = create_audio_sample(&sample, bytes, nbytes, pts, duration);
		if (FAILED(hr))
			return;

		ATL::CComPtr<sld::lib::mf::source::rtsp::stream> stream = NULL;
		{
			auto_lock lock(&_stream_lock);
			std::map<int32_t, ATL::CAdapt<ATL::CComPtr<sld::lib::mf::source::rtsp::stream>>>::iterator stream_iter;
			stream_iter = _streams.find(sld::lib::mf::source::rtsp::source::media_type_t::audio);
			if (stream_iter != _streams.end())
				stream = stream_iter->second;

		}

		{
			sld::lib::exclusive_scopedlock mutex(&_samples_lock);
			std::map<int32_t, std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>>*>::iterator iter;// _samples
			iter = _samples.find(sld::lib::mf::source::rtsp::source::media_type_t::audio);
			if (iter != _samples.end())
			{
				std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>>* aq = iter->second;
				if (aq)
				{
					aq->push_back(sample);
					if (aq->size() > MIN_AUDIO_BUFFER_COUNT)
					{
						if (stream->is_buffering() && _state == sld::lib::mf::source::rtsp::source::state_t::started)
						{
							hr = QueueEvent(MEBufferingStopped, GUID_NULL, hr, NULL);
							if (SUCCEEDED(hr))
								stream->set_buffering(FALSE);
						}
					}
					else
					{
						if (!stream->is_buffering() && _state == sld::lib::mf::source::rtsp::source::state_t::started)
						{
							hr = QueueEvent(MEBufferingStarted, GUID_NULL, hr, NULL);
							if (SUCCEEDED(hr))
								stream->set_buffering(TRUE);
						}
					}
				}
			}
		}

		initialize_presentation_descriptor();

	} while (0);
}

void sld::lib::mf::source::rtsp::source::on_end_audio(void)
{

}


long long sld::lib::mf::source::rtsp::source::elapsed_100nanoseconds(void)
{
	LARGE_INTEGER elapsed_microseconds;
	LARGE_INTEGER now;
	::QueryPerformanceCounter(&now);
	elapsed_microseconds.QuadPart = now.QuadPart - _begin_elapsed_microseconds.QuadPart;
	elapsed_microseconds.QuadPart *= 10000000;
	if (_frequency.QuadPart > 0)
		elapsed_microseconds.QuadPart /= _frequency.QuadPart;
	else
		return 0;
	return elapsed_microseconds.QuadPart;
}