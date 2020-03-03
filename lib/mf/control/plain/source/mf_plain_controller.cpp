#include "mf_plain_controller.h"
#include "mf_topology_builder.h"

solids::lib::mf::control::plain::controller::core::core(void)
	: _context(nullptr)
	, _refcount(1)
	, _state(solids::lib::mf::control::plain::controller::state_t::closed)
	, _session(NULL)
	, _clock(NULL)
	, _topology(NULL)
	, _media_source(NULL)
	, _device_manager(NULL)
	, _presentation_clock(NULL)
	, _video_display(NULL)
	, _rate_control(NULL)
	, _rate_support(NULL)
	, _repeat_count(0)
	, _thinning(FALSE)
{
	MFStartup(MF_VERSION);
	_close_event = ::CreateEvent(NULL, FALSE, FALSE, NULL);
}

solids::lib::mf::control::plain::controller::core::~core(void)
{
	solids::lib::mf::safe_release(_video_display);
	solids::lib::mf::safe_release(_presentation_clock);
	
	//stop();
	close_session();
	shutdown_source();
	solids::lib::mf::safe_release(_topology);
	solids::lib::mf::safe_release(_device_manager);
	MFShutdown();
	::CloseHandle(_close_event);
}

// Playback control
int32_t solids::lib::mf::control::plain::controller::core::open(solids::lib::mf::control::plain::controller::context_t * context)
{
	int32_t status = solids::lib::mf::control::plain::controller::err_code_t::success;
	HRESULT hr = S_OK;
	
	_context = context;
	do
	{

		// Step 1: create a media session if one doesn't exist already
		// close the session if one is already created
		
		_repeat_count = 0;
		stop();
		close_session();
		shutdown_source();
		solids::lib::mf::safe_release(_topology);
		solids::lib::mf::safe_release(_device_manager);

		hr = create_session();
		BREAK_ON_FAIL(hr);
		 
		// Step 2 : create a media source for specified URL string, The URL can be a path to a stream or it can be a path to a local file
		
		hr = solids::lib::mf::control::plain::topology::builder::create_source(_context->url, &_media_source);
		if (FAILED(hr))
		{
			if (HRESULT_FACILITY(hr) == 7) //WIN32 ERROR
			{
				if(HRESULT_CODE(hr)== ERROR_FILE_NOT_FOUND)
				{
					status = solids::lib::mf::control::plain::controller::err_code_t::invalid_file_path;
				}
			}
			else
			{
				status = solids::lib::mf::control::plain::controller::err_code_t::unsupported_media_file;
			}
			break;
		}

		// Step 3: build the topology
		CComQIPtr<IMFPresentationDescriptor> present_descriptor;
		DWORD number_of_streams = 0;
		do
		{
			hr = MFCreateTopology(&_topology);
			BREAK_ON_FAIL(hr);

			hr = _media_source->CreatePresentationDescriptor(&present_descriptor);
			do
			{
				if (hr == MF_E_NOT_INITIALIZED)
					Sleep(100);
				else
					break;
			} while (1);
			BREAK_ON_FAIL(hr);

			hr = present_descriptor->GetStreamDescriptorCount(&number_of_streams);
			BREAK_ON_FAIL(hr);

			for (DWORD index = 0; index < number_of_streams; index++)
			{
				hr = solids::lib::mf::control::plain::topology::builder::add_branch_to_partial_topology(_topology, _media_source, index, present_descriptor, _context, &_device_manager);
				BREAK_ON_FAIL(hr);
			}
		} while (0);
		BREAK_ON_FAIL(hr);

		// Step 4: add the topology to the internal queue of topologies associated with this
		hr = _topology->SetUINT32(MF_TOPOLOGY_HARDWARE_MODE, MFTOPOLOGY_HWMODE_USE_ONLY_HARDWARE);
		BREAK_ON_FAIL(hr);

		hr = _topology->SetUINT32(MF_TOPOLOGY_DXVA_MODE, MFTOPOLOGY_DXVA_FULL);
		BREAK_ON_FAIL(hr);

		hr = _topology->SetUINT32(MF_TOPOLOGY_STATIC_PLAYBACK_OPTIMIZATIONS, TRUE);
		BREAK_ON_FAIL(hr);

		// Step 5: add the topology to the internal queue of topologies associated with this
		// media session
		hr = _session->SetTopology(0, _topology);
		BREAK_ON_FAIL(hr);

		// If we've just initialized a brand new topology in step 1, set the player state 
		// to "open pending" - not playing yet, but ready to begin.
		if (_state == solids::lib::mf::control::plain::controller::state_t::ready)
		{
			_state = solids::lib::mf::control::plain::controller::state_t::open_pending;
		}
	} while (0);

	if (FAILED(hr))
	{
		_state = solids::lib::mf::control::plain::controller::state_t::closed;
		if (status == solids::lib::mf::control::plain::controller::err_code_t::success)
			return solids::lib::mf::control::plain::controller::err_code_t::generic_fail;
	}
	return solids::lib::mf::control::plain::controller::err_code_t::success;
}

int32_t solids::lib::mf::control::plain::controller::core::play(void)
{
	{
		solids::lib::mf::auto_lock lock(&_lock);
		if (_state != solids::lib::mf::control::plain::controller::state_t::open_pending &&
			_state != solids::lib::mf::control::plain::controller::state_t::paused && 
			_state != solids::lib::mf::control::plain::controller::state_t::stopped)
			return solids::lib::mf::control::plain::controller::err_code_t::generic_fail;
	}

	if (_session == NULL)
		return solids::lib::mf::control::plain::controller::err_code_t::generic_fail;

	if ((_state == solids::lib::mf::control::plain::controller::state_t::paused) ||
		(_state == solids::lib::mf::control::plain::controller::state_t::stopped))
	{
		HRESULT hr = start_session();
		if (SUCCEEDED(hr))
			return solids::lib::mf::control::plain::controller::err_code_t::success;
		else
			return solids::lib::mf::control::plain::controller::err_code_t::generic_fail;
	}
	return solids::lib::mf::control::plain::controller::err_code_t::success;
}

int32_t solids::lib::mf::control::plain::controller::core::pause(void)
{
	{
		solids::lib::mf::auto_lock lock(&_lock);
		if (_state != solids::lib::mf::control::plain::controller::state_t::started)
			return solids::lib::mf::control::plain::controller::err_code_t::generic_fail;
	}

	if (_session == NULL)
		return solids::lib::mf::control::plain::controller::err_code_t::generic_fail;

	HRESULT hr = _session->Pause();
	if (SUCCEEDED(hr))
	{
		_state = solids::lib::mf::control::plain::controller::state_t::paused;
		return solids::lib::mf::control::plain::controller::err_code_t::success;
	}
	else
	{
		return solids::lib::mf::control::plain::controller::err_code_t::generic_fail;
	}
}

int32_t solids::lib::mf::control::plain::controller::core::stop(void)
{
	HRESULT hr = S_OK;
	do
	{
		{
			solids::lib::mf::auto_lock lock(&_lock);
			if (_state != solids::lib::mf::control::plain::controller::state_t::started)
			{
				hr = MF_E_INVALIDREQUEST;
				break;
			}
		}
		BREAK_ON_NULL(_session, E_UNEXPECTED);

		HRESULT hr = _session->Stop();
		BREAK_ON_FAIL(hr);

		_state = solids::lib::mf::control::plain::controller::state_t::stopped;

	} while (0);

	if (SUCCEEDED(hr))
	{
		return solids::lib::mf::control::plain::controller::err_code_t::success;
	}
	else
	{
		return solids::lib::mf::control::plain::controller::err_code_t::generic_fail;
	}
}

int32_t solids::lib::mf::control::plain::controller::core::close(void)
{
	stop();
	close_session();
	shutdown_source();
	solids::lib::mf::safe_release(_topology);
	solids::lib::mf::safe_release(_device_manager);

	return solids::lib::mf::control::plain::controller::err_code_t::success;
}

int32_t solids::lib::mf::control::plain::controller::core::state(void) const
{ 
	return _state; 
}

HRESULT solids::lib::mf::control::plain::controller::core::QueryInterface(REFIID riid, void ** ppv)
{
	HRESULT hr = S_OK;

	if (ppv == NULL)
	{
		return E_POINTER;
	}

	if (riid == IID_IMFAsyncCallback)
	{
		*ppv = static_cast<IMFAsyncCallback*>(this);
	}
	else if (riid == IID_IUnknown)
	{
		*ppv = static_cast<IUnknown*>(this);
	}
	else
	{
		*ppv = NULL;
		hr = E_NOINTERFACE;
	}

	if (SUCCEEDED(hr))
		AddRef();

	return hr;
}

ULONG solids::lib::mf::control::plain::controller::core::AddRef(void)
{
	return InterlockedIncrement(&_refcount);
}

ULONG solids::lib::mf::control::plain::controller::core::Release(void)
{
	ULONG uCount = InterlockedDecrement(&_refcount);
	if (uCount == 0)
	{
		delete this;
	}
	return uCount;
}

//
// IMFAsyncCallback::Invoke implementation.  This is the function called by media session
// whenever anything of note happens or an asynchronous operation is complete.
//
// async_result - a pointer to the asynchronous result object which references the event 
// itself in the IMFMediaEventGenerator's event queue.  (The media session is the object
// that implements the IMFMediaEventGenerator interface.)
HRESULT solids::lib::mf::control::plain::controller::core::Invoke(IMFAsyncResult * async_result)
{
	ATL::CComPtr<IMFMediaEvent> media_event;
	HRESULT hr = S_OK;
	MediaEventType media_event_type;

	do
	{
		if (!_session) break;

		// Get the event from the event queue.
		hr = _session->EndGetEvent(async_result, &media_event);
		BREAK_ON_FAIL(hr);

		// Get the event type.
		hr = media_event->GetType(&media_event_type);
		BREAK_ON_FAIL(hr);

		// MESessionClosed event is guaranteed to be the last event fired by the session. 
		// Fire the m_closeCompleteEvent to let the player know that it can safely shut 
		// down the session and release resources associated with the session.
		if (media_event_type == MESessionClosed)
		{
			SetEvent(_close_event);
		}
		else
		{
			// If this is not the final event, tell the Media Session that this player is 
			// the object that will handle the next event in the queue.
			hr = _session->BeginGetEvent(this, NULL);
			BREAK_ON_FAIL(hr);
		}
		// If we are in a normal state, handle the event by passing it to the HandleEvent()
		// function.  Otherwise, if we are in the closing state, do nothing with the event.
		//log_debug("amadeus.player", "Invoke is %d", _state);
		if (_state != solids::lib::mf::control::plain::controller::state_t::closing)
		{
			process_event(media_event);
		}
	} while (0);

	return S_OK;
}

HRESULT solids::lib::mf::control::plain::controller::core::process_event(ATL::CComPtr<IMFMediaEvent> & mediaEvent)
{
	HRESULT hrStatus = S_OK;            // Event status
	HRESULT hr = S_OK;
	MF_TOPOSTATUS topoStatus = MF_TOPOSTATUS::MF_TOPOSTATUS_INVALID;
	MediaEventType mediaEventType;

	do
	{
		BREAK_ON_NULL(mediaEvent, E_POINTER);

		// Get the event type.
		hr = mediaEvent->GetType(&mediaEventType);
		BREAK_ON_FAIL(hr);

		// Get the event status. If the operation that triggered the event did
		// not succeed, the status is a failure code.
		hr = mediaEvent->GetStatus(&hrStatus);

		//Check if the async operation succeeded.
		BREAK_ON_FAIL(hr);

		// Switch on the event type. Update the internal state of the CPlayer as needed.
		if (mediaEventType == MESessionTopologyStatus)
		{
			// Get the status code.
			hr = mediaEvent->GetUINT32(MF_EVENT_TOPOLOGY_STATUS, (UINT32*)&topoStatus);
			BREAK_ON_FAIL(hr);

			if (topoStatus == MF_TOPOSTATUS_READY)
			{
				hr = topology_ready_cb();
			}

			::OutputDebugStringW(L"MF_TOPOSTATUS_READY\n");
		}
		else if (mediaEventType == MEEndOfPresentation)
		{
			hr = presentation_ended_cb();
			::OutputDebugStringW(L"MEEndOfPresentation\n");
		}
		else if (mediaEventType == MESessionStarted)
		{

			_state = solids::lib::mf::control::plain::controller::state_t::started;
			::OutputDebugStringW(L"MESessionStarted\n");
		}
		else if (mediaEventType == MESessionPaused)
		{
			_state = solids::lib::mf::control::plain::controller::state_t::paused;
			::OutputDebugStringW(L"MESessionPaused\n");
		}
		else if (mediaEventType == MESessionRateChanged)
		{
			/*
			solids::lib::mf::auto_lock lock(&_lock);

			PROPVARIANT var_play;
			PropVariantInit(&var_play);
			var_play.vt = VT_I8;
			var_play.hVal.QuadPart = _current_time * UNIT_SECS_2_100NSECS;
			HRESULT hr = _session->Start(&GUID_NULL, &var_play);
			PropVariantClear(&var_play);
			*/
			::OutputDebugStringW(L"MESessionRateChanged\n");
		}
		else if (mediaEventType == MEExtendedType)
		{
			::OutputDebugStringW(L"MEExtendedType\n");
		}
		else if (mediaEventType == MENewPresentation)
		{
			::OutputDebugStringW(L"MENewPresentation\n");
		}
		else if (mediaEventType == MEEndOfPresentationSegment)
		{
			::OutputDebugStringW(L"MEEndOfPresentationSegment\n");
		}
		else if (mediaEventType == MESessionCapabilitiesChanged)
		{
			::OutputDebugStringW(L"MESessionCapabilitiesChanged\n");
		}
		else if (mediaEventType == MESessionStopped)
		{
			::OutputDebugStringW(L"MESessionStopped\n");
		}
		else if (mediaEventType == MESessionClosed)
		{
			::OutputDebugStringW(L"MESessionClosed\n");
		}
		else if (mediaEventType == MESessionEnded)
		{
			::OutputDebugStringW(L"MESessionEnded\n");
		}
		else if (mediaEventType == MESessionUnknown)
		{
			::OutputDebugStringW(L"MESessionUnknown\n");
		}
		else if (mediaEventType == MESessionTopologySet)
		{
			::OutputDebugStringW(L"MESessionTopologySet\n");
		}
		else if (mediaEventType == MESessionTopologiesCleared)
		{
			::OutputDebugStringW(L"MESessionTopologiesCleared\n");
		}
		else if (mediaEventType == MESessionStopped)
		{
			::OutputDebugStringW(L"MESessionStopped\n");
		}
		else if (mediaEventType == MESessionClosed)
		{
			::OutputDebugStringW(L"MESessionClosed\n");
		}
		else if (mediaEventType == MESessionEnded)
		{
			::OutputDebugStringW(L"MESessionEnded\n");
		}
		else if (mediaEventType == MESessionScrubSampleComplete)
		{
			::OutputDebugStringW(L"MESessionScrubSampleComplete\n");
		}
		else if (mediaEventType == MESessionTopologyStatus)
		{
			::OutputDebugStringW(L"MESessionTopologyStatus\n");
		}
		else if (mediaEventType == MESessionNotifyPresentationTime)
		{
			::OutputDebugStringW(L"MESessionNotifyPresentationTime\n");
		}
		else if (mediaEventType == MENewPresentation)
		{
			::OutputDebugStringW(L"MENewPresentation\n");
		}
		else if (mediaEventType == MELicenseAcquisitionStart)
		{
			::OutputDebugStringW(L"MELicenseAcquisitionStart\n");
		}
		else if (mediaEventType == MELicenseAcquisitionCompleted)
		{
			::OutputDebugStringW(L"MELicenseAcquisitionCompleted\n");
		}
		else if (mediaEventType == MEIndividualizationStart)
		{
			::OutputDebugStringW(L"MEIndividualizationStart\n");
		}
		else if (mediaEventType == MEIndividualizationCompleted)
		{
			::OutputDebugStringW(L"MEIndividualizationCompleted\n");
		}
		else if (mediaEventType == MEEnablerProgress)
		{
			::OutputDebugStringW(L"MEEnablerProgress\n");
		}
		else if (mediaEventType == MEEnablerCompleted)
		{
			::OutputDebugStringW(L"MEEnablerCompleted\n");
		}
		else if (mediaEventType == MEPolicyError)
		{
			::OutputDebugStringW(L"MEPolicyError\n");
		}
		else if (mediaEventType == MEPolicyReport)
		{
			::OutputDebugStringW(L"MEPolicyReport\n");
		}
		else if (mediaEventType == MEBufferingStarted)
		{
			::OutputDebugStringW(L"MEBufferingStarted\n");
		}
		else if (mediaEventType == MEBufferingStopped)
		{

			::OutputDebugStringW(L"MEBufferingStopped\n");
		}
		else if (mediaEventType == MEConnectStart)
		{
			::OutputDebugStringW(L"MEConnectStart\n");
		}
		else if (mediaEventType == MEConnectEnd)
		{
			::OutputDebugStringW(L"MEConnectEnd\n");
		}
		else if (mediaEventType == MEReconnectStart)
		{
			::OutputDebugStringW(L"MEReconnectStart\n");
		}
		else if (mediaEventType == MEReconnectEnd)
		{
			::OutputDebugStringW(L"MEReconnectEnd\n");
		}
		else if (mediaEventType == MERendererEvent)
		{
			::OutputDebugStringW(L"MERendererEvent\n");
		}
		else if (mediaEventType == MESessionStreamSinkFormatChanged)
		{
			::OutputDebugStringW(L"MESessionStreamSinkFormatChanged\n");
		}
		else if (mediaEventType == MESourceUnknown)
		{
			::OutputDebugStringW(L"MESourceUnknown\n");
		}
		else if (mediaEventType == MESourceStarted)
		{
			::OutputDebugStringW(L"MESourceStarted\n");
		}
		else if (mediaEventType == MEStreamStarted)
		{
			::OutputDebugStringW(L"MEStreamStarted\n");
		}
		else if (mediaEventType == MESourceSeeked)
		{
			::OutputDebugStringW(L"MESourceSeeked\n");
		}
		else if (mediaEventType == MEStreamSeeked)
		{
			::OutputDebugStringW(L"MEStreamSeeked\n");
		}
		else if (mediaEventType == MENewStream)
		{
			::OutputDebugStringW(L"MENewStream\n");
		}
		else if (mediaEventType == MEUpdatedStream)
		{
			::OutputDebugStringW(L"MEUpdatedStream\n");
		}
		else if (mediaEventType == MESourceStopped)
		{
			::OutputDebugStringW(L"MESourceStopped\n");
		}
		else if (mediaEventType == MEStreamStopped)
		{
			::OutputDebugStringW(L"MEStreamStopped\n");
		}
		else if (mediaEventType == MESourcePaused)
		{
			::OutputDebugStringW(L"MESourcePaused\n");
		}
		else if (mediaEventType == MEStreamPaused)
		{
			::OutputDebugStringW(L"MEStreamPaused\n");
		}
		else if (mediaEventType == MEEndOfStream)
		{
			::OutputDebugStringW(L"MEEndOfStream\n");
		}
		else if (mediaEventType == MEMediaSample)
		{
			::OutputDebugStringW(L"MEMediaSample\n");
		}
		else if (mediaEventType == MEStreamTick)
		{
			::OutputDebugStringW(L"MEStreamTick\n");
		}
		else if (mediaEventType == MEStreamThinMode)
		{
			::OutputDebugStringW(L"MEStreamThinMode\n");
		}
		else if (mediaEventType == MEStreamFormatChanged)
		{
			::OutputDebugStringW(L"MEStreamFormatChanged\n");
		}
		else if (mediaEventType == MESourceRateChanged)
		{
			::OutputDebugStringW(L"MESourceRateChanged\n");
		}
		else if (mediaEventType == MESourceCharacteristicsChanged)
		{
			::OutputDebugStringW(L"MESourceCharacteristicsChanged\n");
		}
		else if (mediaEventType == MESourceRateChangeRequested)
		{
			::OutputDebugStringW(L"MESourceRateChangeRequested\n");
		}
		else if (mediaEventType == MESourceMetadataChanged)
		{
			::OutputDebugStringW(L"MESourceMetadataChanged\n");
		}
		else if (mediaEventType == MESequencerSourceTopologyUpdated)
		{
			::OutputDebugStringW(L"MESequencerSourceTopologyUpdated\n");
		}
		else if (mediaEventType == MESinkUnknown)
		{
			::OutputDebugStringW(L"MESinkUnknown\n");
		}
		else if (mediaEventType == MEStreamSinkStarted)
		{
			::OutputDebugStringW(L"MEStreamSinkStarted\n");
		}
		else if (mediaEventType == MEStreamSinkStopped)
		{
			::OutputDebugStringW(L"MEStreamSinkStopped\n");
		}
		else if (mediaEventType == MEStreamSinkPaused)
		{
			::OutputDebugStringW(L"MEStreamSinkPaused\n");
		}
		else if (mediaEventType == MEStreamSinkRateChanged)
		{
			::OutputDebugStringW(L"MEStreamSinkRateChanged\n");
		}
		else if (mediaEventType == MEStreamSinkRequestSample)
		{
			::OutputDebugStringW(L"MEStreamSinkRequestSample\n");
		}
		else if (mediaEventType == MEStreamSinkMarker)
		{
			::OutputDebugStringW(L"MEStreamSinkMarker\n");
		}
		else if (mediaEventType == MEStreamSinkPrerolled)
		{
			::OutputDebugStringW(L"MEStreamSinkPrerolled\n");
		}
		else if (mediaEventType == MEStreamSinkScrubSampleComplete)
		{
			::OutputDebugStringW(L"MEStreamSinkScrubSampleComplete\n");
		}
		else if (mediaEventType == MEStreamSinkFormatChanged)
		{
			::OutputDebugStringW(L"MEStreamSinkFormatChanged\n");
		}
		else if (mediaEventType == MEStreamSinkDeviceChanged)
		{
			::OutputDebugStringW(L"MEStreamSinkDeviceChanged\n");
		}
		else if (mediaEventType == MEQualityNotify)
		{
			::OutputDebugStringW(L"MEQualityNotify\n");
		}
		else if (mediaEventType == MESinkInvalidated)
		{
			::OutputDebugStringW(L"MESinkInvalidated\n");
		}
		else if (mediaEventType == MEAudioSessionNameChanged)
		{
			::OutputDebugStringW(L"MEAudioSessionNameChanged\n");
		}
		else if (mediaEventType == MEAudioSessionVolumeChanged)
		{
			::OutputDebugStringW(L"MEAudioSessionVolumeChanged\n");
		}
		else if (mediaEventType == MEAudioSessionDeviceRemoved)
		{
			::OutputDebugStringW(L"MEAudioSessionDeviceRemoved\n");
		}
		else if (mediaEventType == MEAudioSessionServerShutdown)
		{
			::OutputDebugStringW(L"MEAudioSessionServerShutdown\n");
		}
		else if (mediaEventType == MEAudioSessionGroupingParamChanged)
		{
			::OutputDebugStringW(L"MEAudioSessionGroupingParamChanged\n");
		}
		else if (mediaEventType == MEAudioSessionIconChanged)
		{
			::OutputDebugStringW(L"MEAudioSessionIconChanged\n");
		}
		else if (mediaEventType == MEAudioSessionFormatChanged)
		{
			::OutputDebugStringW(L"MEAudioSessionFormatChanged\n");
		}
		else if (mediaEventType == MEAudioSessionDisconnected)
		{
			::OutputDebugStringW(L"MEAudioSessionDisconnected\n");
		}
		else if (mediaEventType == MEAudioSessionExclusiveModeOverride)
		{
			::OutputDebugStringW(L"MEAudioSessionExclusiveModeOverride\n");
		}
		else if (mediaEventType == MECaptureAudioSessionVolumeChanged)
		{
			::OutputDebugStringW(L"MECaptureAudioSessionVolumeChanged\n");
		}
		else if (mediaEventType == MECaptureAudioSessionDeviceRemoved)
		{
			::OutputDebugStringW(L"MECaptureAudioSessionDeviceRemoved\n");
		}
		else if (mediaEventType == MECaptureAudioSessionFormatChanged)
		{
			::OutputDebugStringW(L"MECaptureAudioSessionFormatChanged\n");
		}
		else if (mediaEventType == MECaptureAudioSessionDisconnected)
		{
			::OutputDebugStringW(L"MECaptureAudioSessionDisconnected\n");
		}
		else if (mediaEventType == MECaptureAudioSessionExclusiveModeOverride)
		{
			::OutputDebugStringW(L"MECaptureAudioSessionExclusiveModeOverride\n");
		}
		else if (mediaEventType == MECaptureAudioSessionServerShutdown)
		{
			::OutputDebugStringW(L"MECaptureAudioSessionServerShutdown\n");
		}
		else if (mediaEventType == METrustUnknown)
		{
			::OutputDebugStringW(L"METrustUnknown\n");
		}
		else if (mediaEventType == MEPolicyChanged)
		{
			::OutputDebugStringW(L"MEPolicyChanged\n");
		}
		else if (mediaEventType == MEContentProtectionMessage)
		{
			::OutputDebugStringW(L"MEContentProtectionMessage\n");
		}
		else if (mediaEventType == MEPolicySet)
		{
			::OutputDebugStringW(L"MEPolicySet\n");
		}
		else if (mediaEventType == MEWMDRMLicenseBackupCompleted)
		{
			::OutputDebugStringW(L"MEWMDRMLicenseBackupCompleted\n");
		}
		else if (mediaEventType == MEWMDRMLicenseBackupProgress)
		{
			::OutputDebugStringW(L"MEWMDRMLicenseBackupProgress\n");
		}
		else if (mediaEventType == MEWMDRMLicenseRestoreCompleted)
		{
			::OutputDebugStringW(L"MEWMDRMLicenseRestoreCompleted\n");
		}
		else if (mediaEventType == MEWMDRMLicenseRestoreProgress)
		{
			::OutputDebugStringW(L"MEWMDRMLicenseRestoreProgress\n");
		}
		else if (mediaEventType == MEWMDRMLicenseAcquisitionCompleted)
		{
			::OutputDebugStringW(L"MEWMDRMLicenseAcquisitionCompleted\n");
		}
		else if (mediaEventType == MEWMDRMIndividualizationCompleted)
		{
			::OutputDebugStringW(L"MEWMDRMIndividualizationCompleted\n");
		}
		else if (mediaEventType == MEWMDRMIndividualizationProgress)
		{
			::OutputDebugStringW(L"MEWMDRMIndividualizationProgress\n");
		}
		else if (mediaEventType == MEWMDRMProximityCompleted)
		{
			::OutputDebugStringW(L"MEWMDRMProximityCompleted\n");
		}
		else if (mediaEventType == MEWMDRMLicenseStoreCleaned)
		{
			::OutputDebugStringW(L"MEWMDRMLicenseStoreCleaned\n");
		}
		else if (mediaEventType == MEWMDRMRevocationDownloadCompleted)
		{
			::OutputDebugStringW(L"MEWMDRMRevocationDownloadCompleted\n");
		}
		else if (mediaEventType == METransformUnknown)
		{
			::OutputDebugStringW(L"METransformUnknown\n");	
		}
		else if (mediaEventType == METransformNeedInput)
		{
			::OutputDebugStringW(L"METransformNeedInput\n");
		}
		else if (mediaEventType == METransformHaveOutput)
		{
			::OutputDebugStringW(L"METransformHaveOutput\n");
		}
		else if (mediaEventType == METransformDrainComplete)
		{
			::OutputDebugStringW(L"METransformDrainComplete\n");
		}
		else if (mediaEventType == METransformMarker)
		{
			::OutputDebugStringW(L"METransformMarker\n");
		}
		else if (mediaEventType == MEByteStreamCharacteristicsChanged)
		{
			::OutputDebugStringW(L"MEByteStreamCharacteristicsChanged\n");
		}
		else if (mediaEventType == MEVideoCaptureDeviceRemoved)
		{
			::OutputDebugStringW(L"MEVideoCaptureDeviceRemoved\n");
		}
		else if (mediaEventType == MEVideoCaptureDevicePreempted)
		{
			::OutputDebugStringW(L"MEVideoCaptureDevicePreempted\n");
		}
		else if (mediaEventType == MEStreamSinkFormatInvalidated)
		{
			::OutputDebugStringW(L"MEStreamSinkFormatInvalidated\n");
		}
		else if (mediaEventType == MEEncodingParameters)
		{
			::OutputDebugStringW(L"MEEncodingParameters\n");
		}
		else if (mediaEventType == MEContentProtectionMetadata)
		{
			::OutputDebugStringW(L"MEContentProtectionMetadata\n");
		}
	} while (0);
	return hr;
}


HRESULT solids::lib::mf::control::plain::controller::core::topology_ready_cb(void)
{
	HRESULT hr = S_OK;
	// release any previous instance of the m_pVideoDisplay interface

	solids::lib::mf::safe_release(_video_display);

	// Ask the session for the IMFVideoDisplayControl interface. 
	MFGetService(_session, MR_VIDEO_RENDER_SERVICE, IID_PPV_ARGS(&_video_display));

	solids::lib::mf::safe_release(_clock);
	
	_session->GetClock(&_clock);
	_clock->QueryInterface(IID_PPV_ARGS(&_presentation_clock));

	// get the rate control service that can be used to change the playback rate of the service
	{
		_rate_control = NULL;
		_rate_support = NULL;
		MFGetService(_session, MF_RATE_CONTROL_SERVICE, IID_PPV_ARGS(&_rate_control));
		MFGetService(_session, MF_RATE_CONTROL_SERVICE, IID_PPV_ARGS(&_rate_support));

		//_session->GetSessionCapabilities(&_session_cap);
	}
	
	// since the topology is ready, start playback
	hr = start_session();

	return hr;
}

HRESULT solids::lib::mf::control::plain::controller::core::presentation_ended_cb(void)
{
	// The session puts itself into the stopped state automatically.
	if (_context && _context->repeat)
	{
		PROPVARIANT var_play;
		PropVariantInit(&var_play);
		var_play.vt = VT_I8;
		var_play.hVal.QuadPart = 0;

		_repeat_count++;

		HRESULT hr = pause();
		if (SUCCEEDED(hr))
			_state = solids::lib::mf::control::plain::controller::state_t::paused;
		
		if (SUCCEEDED(hr))
		{
			HRESULT hr = _session->Start(&GUID_NULL, &var_play);
			if (SUCCEEDED(hr))
				_state = solids::lib::mf::control::plain::controller::state_t::started;
		}
		PropVariantClear(&var_play);
	}
	else
	{
		_state = solids::lib::mf::control::plain::controller::state_t::stopped;
		//_session->Stop();
		//_session->Shutdown();
	}

	return S_OK;
}

HRESULT solids::lib::mf::control::plain::controller::core::create_session(void)
{
	// Close the old session, if any.
	HRESULT hr = S_OK;
	do
	{
		assert(_state == solids::lib::mf::control::plain::controller::state_t::closed);

		// Create the media session.
		hr = MFCreateMediaSession(NULL, &_session);
		BREAK_ON_FAIL(hr);

		_state = solids::lib::mf::control::plain::controller::state_t::ready;

		// designate this class as the one that will be handling events from the media 
		hr = _session->BeginGetEvent((IMFAsyncCallback*)this, NULL);
		BREAK_ON_FAIL(hr);
	} while (0);

	return hr;
}

HRESULT solids::lib::mf::control::plain::controller::core::close_session(void)
{
	HRESULT hr = S_OK;
	DWORD wait_result = 0;

	//_state = amadeus::mf::player::framework::vr360::state_t::closing;

	// release the video display object
	solids::lib::mf::safe_release(_video_display);

	// Call the asynchronous Close() method and then wait for the close
	// operation to complete on another thread
	if (_session != NULL)
	{
		_state = solids::lib::mf::control::plain::controller::state_t::closing;
		hr = _session->Close();
		if (SUCCEEDED(hr))
		{
			// Begin waiting for the Win32 close event, fired in CPlayer::Invoke(). The 
			// close event will indicate that the close operation is finished, and the 
			// session can be shut down.
			wait_result = WaitForSingleObject(_close_event, 3000);
			if (wait_result == WAIT_TIMEOUT)
			{
			}
		}
	}

	// Shut down the media session. (Synchronous operation, no events.)  Releases all of the
	// internal session resources.
	if (_session != NULL)
	{
		_session->Shutdown();
	}

	_session = NULL;
	_state = solids::lib::mf::control::plain::controller::state_t::closed;
	return hr;
}

HRESULT solids::lib::mf::control::plain::controller::core::start_session(void)
{
	// If Start fails later, we will get an MESessionStarted event with an error code, 
	// and will update our state. Passing in GUID_NULL and VT_EMPTY indicates that
	// playback should start from the current position.
	if (_state != solids::lib::mf::control::plain::controller::state_t::started)
	{
		assert(_session != NULL);
		PROPVARIANT var_play;
		PropVariantInit(&var_play);
		var_play.vt = VT_EMPTY;

		HRESULT hr = _session->Start(&GUID_NULL, &var_play);
		if (SUCCEEDED(hr))
			_state = solids::lib::mf::control::plain::controller::state_t::started;
		PropVariantClear(&var_play);
		return hr;
	}
	return S_OK;
}

HRESULT solids::lib::mf::control::plain::controller::core::shutdown_source(void)
{
	HRESULT hr = S_OK;
	if (_media_source)
	{
		// shut down the source
		hr = _media_source->Shutdown();
		// release the source, since all subsequent calls to it will fail
		solids::lib::mf::safe_release(_media_source);
	}
	else
	{
		hr = E_UNEXPECTED;
	}
	return hr;
}

#ifdef _DEBUG
const char * solids::lib::mf::control::plain::controller::core::event_type(DWORD evt)
{
	switch (evt)
	{
	case MEError: return "MEError";
	case MEExtendedType: return "MEExtendedType";
	case MESessionTopologySet: return "MESessionTopologySet";
	case MESessionTopologiesCleared: return "MESessionTopologiesCleared";
	case MENewPresentation: return "MENewPresentation";
	case MESessionStarted: return "MESessionStarted";
	case MESessionPaused: return "MESessionPaused";
	case MESessionStopped: return "MESessionStopped";
	case MESessionClosed: return "MESessionClosed";
	case MESessionEnded: return "MESessionEnded";
	case MESessionRateChanged: return "MESessionRateChanged";
	case MESessionCapabilitiesChanged: return "MESessionCapabilitiesChanged";
	case MESessionTopologyStatus: return "MESessionTopologyStatus";
	case MESessionNotifyPresentationTime: return "MESessionNotifyPresentationTime";
	case MESessionStreamSinkFormatChanged: return "MESessionStreamSinkFormatChanged";
	case MEEndOfPresentation: return "MEEndOfPresentation";
	case MEStreamThinMode: return "MEStreamThinMode";
	case MEEndOfPresentationSegment: return "MEEndOfPresentationSegment";
	default: return NULL;
	}
}
#endif