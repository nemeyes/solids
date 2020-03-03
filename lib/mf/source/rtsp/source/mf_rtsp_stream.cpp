#include "mf_rtsp_stream.h"

sld::lib::mf::source::rtsp::stream::stream(void)
	: _event_queue(NULL)
	, _state(sld::lib::mf::source::rtsp::stream::state_t::stopped)
	, _active(FALSE)
	, _eos(FALSE)
	, _buffering(FALSE)
{

}

sld::lib::mf::source::rtsp::stream::~stream(void)
{
	release();
}

HRESULT sld::lib::mf::source::rtsp::stream::QueryInterface(REFIID iid, void** ppv)
{
	if (!ppv)
		return E_POINTER;
	if (iid == IID_IUnknown)
		*ppv = static_cast<IUnknown*>(this);
	else if (iid == __uuidof(IMFMediaStream))
		*ppv = static_cast<IMFMediaStream*>(this);
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

ULONG sld::lib::mf::source::rtsp::stream::AddRef(void)
{ 
	return sld::lib::mf::refcount_object::AddRef(); 
}

ULONG sld::lib::mf::source::rtsp::stream::Release(void)
{ 
	return sld::lib::mf::refcount_object::Release();
}

HRESULT sld::lib::mf::source::rtsp::stream::BeginGetEvent(IMFAsyncCallback * callback, IUnknown * unk)
{
	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);

	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		hr = _event_queue->BeginGetEvent(callback, unk);
		if (FAILED(hr))
			break;

	} while (0);
	return hr;
}

HRESULT sld::lib::mf::source::rtsp::stream::EndGetEvent(IMFAsyncResult * result, IMFMediaEvent ** evt)
{
	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);

	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		hr = _event_queue->EndGetEvent(result, evt);

	} while (0);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::stream::GetEvent(DWORD flags, IMFMediaEvent ** evt)
{
	HRESULT hr = S_OK;
	IMFMediaEventQueue * queue = NULL;

	do
	{
		sld::lib::mf::auto_lock mutex(&_lock);
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		// Cache a local pointer to the queue.
		queue =_event_queue;
		queue->AddRef();

		hr = queue->GetEvent(flags, evt);

	} while (0);

	safe_release(queue);
	return hr;
}

HRESULT sld::lib::mf::source::rtsp::stream::QueueEvent(MediaEventType met, REFGUID guidExtendedType, HRESULT hrStatus, const PROPVARIANT* pvValue)
{
	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);
	
	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		hr = _event_queue->QueueEventParamVar(met, guidExtendedType, hrStatus, pvValue);

	} while (0);

	return hr;
}

//-------------------------------------------------------------------
// IMFMediaStream methods
//-------------------------------------------------------------------
HRESULT sld::lib::mf::source::rtsp::stream::GetMediaSource(IMFMediaSource ** source)
{
	if (!source)
		return E_POINTER;
	if (!_source)
		return E_UNEXPECTED;

	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);

	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		hr = _source->QueryInterface(IID_PPV_ARGS(source));

	} while (0);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::stream::GetStreamDescriptor(IMFStreamDescriptor ** sd)
{
	HRESULT hr = S_OK;
	if (!sd)
		return E_POINTER;
	if (!_sd)
		return E_UNEXPECTED;

	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		*sd = _sd;
		(*sd)->AddRef();

	} while (0);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::stream::RequestSample(IUnknown * token)
{
	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);

	IMFMediaSource* source = NULL;
	IMFSample * sample = NULL;
	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		if (_state == sld::lib::mf::source::rtsp::stream::state_t::stopped)
		{
			hr = MF_E_MEDIA_SOURCE_WRONGSTATE;
			break;
		}

		if (_state == sld::lib::mf::source::rtsp::stream::state_t::paused)
		{
			hr = S_OK;
			break;
		}

		if (!_active)
		{
			hr = MF_E_INVALIDREQUEST;
			break;
		}

		// Fail if we reached the end of the stream AND the sample queue is empty,
		int32_t total_waiting_duration = 1000;
		int32_t waiting_duration = 10;
		int32_t index = 0;
		do
		{
			if (_type == sld::lib::mf::source::rtsp::source::media_type_t::video)
				_source->get_video_sample(&sample);
			else if (_type == sld::lib::mf::source::rtsp::source::media_type_t::audio)
				_source->get_audio_sample(&sample);

			if (sample || index>=(total_waiting_duration/waiting_duration) || _eos)
				break;

			::Sleep(waiting_duration);

			index++;

		} while (1);

		if (sample)
		{
			if (token)
			{
				IUnknown * ptoken = NULL;
				ptoken = token;
				ptoken->AddRef();

				hr = sample->SetUnknown(MFSampleExtension_Token, ptoken);

				safe_release(ptoken);
				if (FAILED(hr))
					break;
			}

			/*
			DWORD pBufferCnt;
			sample->GetBufferCount(&pBufferCnt);
			IMFMediaBuffer * pBuffer = NULL;
			sample->GetBufferByIndex(0, &pBuffer);
			
			DWORD pDataLength = 0;
			pBuffer->GetCurrentLength(&pDataLength);
			BYTE * pSrc = NULL;
			BYTE* pDst = new BYTE[pDataLength];
			hr = pBuffer->Lock(&pSrc, NULL, NULL);
			if (FAILED(hr))
				break;

			::memmove(pDst, pSrc, pDataLength);

			hr = pBuffer->Unlock();
			if (FAILED(hr))
				break;

			delete [] pDst;
			*/

			hr = _event_queue->QueueEventParamUnk(MEMediaSample, GUID_NULL, hr, sample);
			if (FAILED(hr))
				break;

			safe_release(sample);

		}
		else
		{
			if (_eos)
			{
				hr = MF_E_END_OF_STREAM;
				break;
			}
			else
			{

				if (_type == sld::lib::mf::source::rtsp::source::media_type_t::video)
					::OutputDebugStringW(L"===========================Video RequestSample Failed================================\n");
				else if (_type == sld::lib::mf::source::rtsp::source::media_type_t::audio)
					::OutputDebugStringW(L"===========================Audio RequestSample Failed================================\n");

				hr = MF_E_INVALIDREQUEST;
			}
		}

	} while (0);

	// If there was an error, queue MEError from the source (except after shutdown).
	if (FAILED(hr) && (_state != sld::lib::mf::source::rtsp::stream::state_t::finalized))
		hr = _source->QueueEvent(MEError, GUID_NULL, hr, NULL);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::stream::initialize(sld::lib::mf::source::rtsp::source * source, IMFStreamDescriptor * sd, int32_t type)
{
	_source = source;
	_source->AddRef();

	_sd = sd;
	_sd->AddRef();

	_type = type;

	_buffering = FALSE;

	HRESULT hr = MFCreateEventQueue(&_event_queue);
	return hr;
}

HRESULT sld::lib::mf::source::rtsp::stream::release(void)
{
	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);

	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		
		if (_event_queue)
			hr = _event_queue->Shutdown();

		if (FAILED(hr))
			break;


		_state = sld::lib::mf::source::rtsp::stream::state_t::finalized;

	} while (FALSE);

	sld::lib::mf::safe_release(_sd);
	sld::lib::mf::safe_release(_source);
	sld::lib::mf::safe_release(_event_queue);

	return hr;
}

// Other methods (called by source)
HRESULT sld::lib::mf::source::rtsp::stream::activate(BOOL active)
{
	sld::lib::mf::auto_lock mutex(&_lock);

	if (_active == active)
		return S_OK;

	_active = active;
	return S_OK;
}

HRESULT sld::lib::mf::source::rtsp::stream::start(const PROPVARIANT & start)
{
	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);

	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;
		hr = QueueEvent(MEStreamStarted, GUID_NULL, S_OK, &start);
		if (FAILED(hr))
			break;

		_buffering = FALSE;
		_state = sld::lib::mf::source::rtsp::stream::state_t::started;

	} while (0);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::stream::pause(void)
{
	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);

	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		hr = QueueEvent(MEStreamPaused, GUID_NULL, S_OK, NULL);
		if (FAILED(hr))
			break;

		_state = sld::lib::mf::source::rtsp::stream::state_t::paused;

	} while (0);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::stream::stop(void)
{
	HRESULT hr = S_OK;
	sld::lib::mf::auto_lock mutex(&_lock);

	do
	{
		hr = check_shutdown();
		if (FAILED(hr))
			break;

		_eos = TRUE;

		hr = QueueEvent(MEStreamStopped, GUID_NULL, S_OK, NULL);
		if (FAILED(hr))
			break;

		_state = sld::lib::mf::source::rtsp::stream::state_t::stopped;

	} while (0);

	return hr;
}

BOOL sld::lib::mf::source::rtsp::stream::is_active(void) const 
{ 
	return _active; 
}

void sld::lib::mf::source::rtsp::stream::set_buffering(BOOL buffering)
{
	_buffering = buffering;
}

BOOL sld::lib::mf::source::rtsp::stream::is_buffering(void) const
{
	return _buffering;
}

//private function
HRESULT sld::lib::mf::source::rtsp::stream::check_shutdown(void) const
{
	return _state == sld::lib::mf::source::rtsp::stream::state_t::finalized ? MF_E_SHUTDOWN : S_OK;
}