#include "mf_merge_transform.h"

//#include "mf_module_interface.h"
HRESULT solids::lib::mf::transform::merge::transform::CreateInstance(REFIID riid, void ** ppv)
{
	if (!ppv)
		return E_POINTER;
	*ppv = NULL;

	HRESULT hr = S_OK;
	solids::lib::mf::transform::merge::transform * transform = new solids::lib::mf::transform::merge::transform(); // Created with ref count = 1.

	if (transform == NULL)
		hr = E_OUTOFMEMORY;

	if (SUCCEEDED(hr))
		hr = transform->QueryInterface(riid, ppv);

	solids::lib::mf::safe_release(transform);

	return hr;
}

solids::lib::mf::transform::merge::transform::transform(void)
	: _input_stream_count(MIN_INPUT_STREAM_COUNT)
	, _is_output_type_set(FALSE)
	, _selected_id(0)
	, _selected_only(FALSE)
	, _overlapped(FALSE)
	, _overlapped_count(0)
	, _output_type(NULL)
	, _is_first_sample(TRUE)
	, _ratios(NULL)
	, _stream_duration(0)
	, _curr_pts(0)
	, _prev_pts(0)
{
	_input_stream_ids = new DWORD[MAX_INPUT_STREAM_COUNT];
	for (DWORD i = 0; i < MAX_INPUT_STREAM_COUNT; i++)
	{
		_input_buffer[i] = NULL;
		_input_type[i] = NULL;
		_is_input_type_set[i] = FALSE;
		_input_stream_ids[i] = i;
		_input_updated[i] = FALSE;
	}
}

solids::lib::mf::transform::merge::transform::~transform(void)
{
	if (_input_stream_ids)
		delete[] _input_stream_ids;
	if(_ratios)
		delete[] _ratios;
}

// IUnknown
HRESULT solids::lib::mf::transform::merge::transform::QueryInterface(REFIID iid, void** ppv)
{
	if (NULL == ppv)
	{
		return E_POINTER;
	}
	else if (iid == __uuidof(IUnknown))
	{
		*ppv = static_cast<IUnknown*>(static_cast<IMFTransform*>(this));
	}
	else if (iid == __uuidof(IMFTransform))
	{
		*ppv = static_cast<IMFTransform*>(this);
	}
	else if (iid == __uuidof(IMergeTransformContext))
	{
		*ppv = static_cast<IMergeTransformContext*>(this);
	}
	else
	{
		*ppv = NULL;
		return E_NOINTERFACE;
	}
	AddRef();
	return S_OK;
}

ULONG solids::lib::mf::transform::merge::transform::AddRef(void)
{
	return refcount_object::AddRef();
}

ULONG solids::lib::mf::transform::merge::transform::Release(void)
{
	return refcount_object::Release();
}

HRESULT solids::lib::mf::transform::merge::transform::GetStreamLimits(DWORD * inputMinimum, DWORD * inputMaximum, DWORD * outputMinimum, DWORD * outputMaximum)
{
	if (inputMinimum)	
		*inputMinimum = MIN_INPUT_STREAM_COUNT;
	if (inputMaximum)	
		*inputMaximum = MAX_INPUT_STREAM_COUNT;
	if (outputMinimum)	
		*outputMinimum = 1;
	if (outputMaximum)	
		*outputMaximum = 1;
	return S_OK;
}

HRESULT solids::lib::mf::transform::merge::transform::GetStreamCount(DWORD * inputStreams, DWORD * outputStreams)
{
	if (inputStreams)		
		*inputStreams = _input_stream_count;
	if (outputStreams)	
		*outputStreams = 1;
	return S_OK;
}

HRESULT solids::lib::mf::transform::merge::transform::GetStreamIDs(DWORD inputIDSize, DWORD * inputIDs, DWORD outputIDSize, DWORD * outputIDs)
{
	if (inputIDs)
	{
		for (DWORD i = 0; i < _input_stream_count; i++)
			inputIDs[i] = _input_stream_ids[i];
	}
	if(outputIDs)
		outputIDs[0] = 0;
	return E_NOTIMPL;
}

HRESULT solids::lib::mf::transform::merge::transform::GetInputStreamInfo(DWORD inputStreamID, MFT_INPUT_STREAM_INFO * streamInfo)
{
	HRESULT hr = S_OK;
	if (streamInfo == NULL)
		return E_POINTER;
	if (get_input_stream_index(inputStreamID)==NO_INDEX)
		return MF_E_INVALIDSTREAMNUMBER;

	{
		streamInfo->dwFlags = MFT_INPUT_STREAM_WHOLE_SAMPLES | MFT_INPUT_STREAM_REMOVABLE | MFT_INPUT_STREAM_PROCESSES_IN_PLACE | MFT_INPUT_STREAM_DOES_NOT_ADDREF;
		streamInfo->hnsMaxLatency = 0;
		streamInfo->cbSize = 0;
		streamInfo->cbMaxLookahead = 0;
		streamInfo->cbAlignment = 0;
	}
	return hr;
}

HRESULT solids::lib::mf::transform::merge::transform::GetOutputStreamInfo(DWORD outputStreamID, MFT_OUTPUT_STREAM_INFO * streamInfo)
{
	HRESULT hr = S_OK;
	if (streamInfo == NULL)
		return E_POINTER;
	if (outputStreamID > BASE_OUTPUT_STREAM_ID)
		return MF_E_INVALIDSTREAMNUMBER;
	
	{																							
		solids::lib::mf::auto_lock lock(&_lock);

		streamInfo->dwFlags = MFT_OUTPUT_STREAM_WHOLE_SAMPLES | MFT_OUTPUT_STREAM_REMOVABLE | MFT_OUTPUT_STREAM_PROVIDES_SAMPLES;
		streamInfo->cbSize = 0;
		streamInfo->cbAlignment = 0;
	}
	return S_OK;
}

HRESULT solids::lib::mf::transform::merge::transform::GetAttributes(IMFAttributes ** attributes)
{
	return E_NOTIMPL;
}

HRESULT solids::lib::mf::transform::merge::transform::GetInputStreamAttributes(DWORD inputStreamID, IMFAttributes ** attributes)
{
	return E_NOTIMPL;
}

HRESULT solids::lib::mf::transform::merge::transform::GetOutputStreamAttributes(DWORD outputStreamID, IMFAttributes ** attributes)
{
	return E_NOTIMPL;
}

HRESULT solids::lib::mf::transform::merge::transform::DeleteInputStream(DWORD streamID)
{
	BOOL found = FALSE; 
	if (_input_stream_count == MIN_INPUT_STREAM_COUNT)
		return MF_E_INVALIDREQUEST;

	if (!(GetInputStreamInfo(streamID, NULL) & MFT_INPUT_STREAM_REMOVABLE))
		return MF_E_INVALIDREQUEST;

	for (DWORD i = 0; i < _input_stream_count; i++)
	{
		if (streamID == _input_stream_ids[i])
			found = TRUE;
		if (found)
		{
			if (i == _input_stream_count - 1)
				break;
			_input_stream_ids[i] = _input_stream_ids[i + 1];
		}	
	}
	if (found == FALSE)
		return MF_E_INVALIDSTREAMNUMBER;

	_input_stream_count--;
	return S_OK;
}

HRESULT solids::lib::mf::transform::merge::transform::AddInputStreams(DWORD streamSize, DWORD * streamIDs)
{
	if (streamSize + _input_stream_count > MAX_INPUT_STREAM_COUNT)
		return E_INVALIDARG;
	for (DWORD i = 0; i < _input_stream_count; i++)
	{
		for (DWORD j = 0; j < streamSize; j++)
		{
			if (_input_stream_ids[i] == streamIDs[j])
			{
				return E_INVALIDARG;
			}
		}
	}
	for (DWORD i = 0; i < streamSize; i++)
		_input_stream_ids[_input_stream_count + i] = streamIDs[i];
	_input_stream_count += streamSize;
	return S_OK;
}

HRESULT solids::lib::mf::transform::merge::transform::GetInputAvailableType(DWORD inputStreamID, DWORD typeIndex, IMFMediaType ** ppmt)
{
	return E_NOTIMPL;	//MFT does not have a list of preferred input types
/*
	if (ppType == NULL)
		return E_INVALIDARG;
	if (FAILED(IsValidInputStream(dwInputStreamID)))
		return MF_E_INVALIDSTREAMNUMBER;
	if (dwTypeIndex > 0) 
		return MF_E_NO_MORE_TYPES;
	
	auto_lock lock(&_cs);
	{
		DWORD index = GetInputStreamIndexById(dwInputStreamID);
		if (_input_type[index])
		{
			*ppType = _input_type[index];
			(*ppType)->AddRef();
		}
	}
	return S_OK;
*/
}

HRESULT solids::lib::mf::transform::merge::transform::GetOutputAvailableType(DWORD outputStreamID, DWORD typeIndex, IMFMediaType ** ppmt)
{
	//return E_NOTIMPL;	// MFT does not have a list of preferred output types
	if (ppmt == NULL)
		return E_INVALIDARG;
	if (outputStreamID > BASE_OUTPUT_STREAM_ID)
		return MF_E_INVALIDSTREAMNUMBER;
	if (typeIndex > 0)
		return MF_E_NO_MORE_TYPES;

	{
		solids::lib::mf::auto_lock lock(&_lock);
#if 0 
		if (_output_type)
		{
			*ppType = _output_type;
			(*ppType)->AddRef();
			return S_OK;
		}
		return MF_E_TRANSFORM_TYPE_NOT_SET;
#endif
		for (uint32_t i = 0; i < _input_stream_count; i++)
		{
			if (_input_type[i])
			{
				*ppmt = _input_type[i];
				(*ppmt)->AddRef();
				return S_OK;
			}
		}
		return MF_E_TRANSFORM_TYPE_NOT_SET;
	}
}

HRESULT solids::lib::mf::transform::merge::transform::SetInputType(DWORD id, IMFMediaType * mt, DWORD flags)
{
	if ((flags & MFT_SET_TYPE_TEST_ONLY) && (mt == NULL))
		return E_INVALIDARG;
	if (FAILED(is_valid_input_stream(id)))
		return MF_E_INVALIDSTREAMNUMBER;

	if ((flags & MFT_SET_TYPE_TEST_ONLY) && (mt != NULL))
	{
		if (SUCCEEDED(is_type_acceptable(solids::lib::mf::transform::merge::transform::direction_t::input, id, mt)))
			return S_OK;
		else
			return MF_E_INVALIDMEDIATYPE;
	}	
		
	if (flags & ~MFT_SET_TYPE_TEST_ONLY) 
		return E_INVALIDARG;

	{
		solids::lib::mf::auto_lock lock(&_lock);
		for (DWORD i = 0; i < _input_stream_count; i++)
		{
			if ((id == _input_stream_ids[i]) && _input_buffer[i] != NULL)
				return MF_E_TRANSFORM_CANNOT_CHANGE_MEDIATYPE_WHILE_PROCESSING;
		}
		if (SUCCEEDED(is_type_acceptable(solids::lib::mf::transform::merge::transform::direction_t::input, id, mt)))
			return set_media_type(solids::lib::mf::transform::merge::transform::direction_t::input, id, mt);
		else
			return MF_E_INVALIDMEDIATYPE;
	}
}

HRESULT solids::lib::mf::transform::merge::transform::SetOutputType(DWORD id, IMFMediaType * mt, DWORD flags)
{
	if (_output_type == mt)
		return S_OK;
	if ((flags & MFT_SET_TYPE_TEST_ONLY) && (mt == NULL))
		return E_INVALIDARG;

	if (id > BASE_OUTPUT_STREAM_ID)
		return MF_E_INVALIDSTREAMNUMBER;

	if ((flags & MFT_SET_TYPE_TEST_ONLY) && (mt != NULL))
	{
		if (SUCCEEDED(is_type_acceptable(solids::lib::mf::transform::merge::transform::direction_t::output, id, mt)))
			return S_OK;
		else
			return MF_E_INVALIDMEDIATYPE;
	}

	if (flags & ~MFT_SET_TYPE_TEST_ONLY)
		return E_INVALIDARG;
	
	{
		solids::lib::mf::auto_lock lock(&_lock);
		//If we have output, the client cannot change the type now.
		//if (_output_buffer != NULL)
		//	return MF_E_TRANSFORM_CANNOT_CHANGE_MEDIATYPE_WHILE_PROCESSING;
		HRESULT hr = S_OK;
		if (SUCCEEDED(is_type_acceptable(solids::lib::mf::transform::merge::transform::direction_t::output, id, mt)))
			return set_media_type(solids::lib::mf::transform::merge::transform::direction_t::output, id, mt);
		else
			return MF_E_INVALIDMEDIATYPE;
	}
	return S_OK;
}

HRESULT solids::lib::mf::transform::merge::transform::GetInputCurrentType(DWORD id, IMFMediaType ** ppmt)
{
	if (ppmt == NULL)
		return E_POINTER;

	DWORD index = get_input_stream_index(id);
	if (index == NO_INDEX)
		return MF_E_INVALIDSTREAMNUMBER;

	{
		solids::lib::mf::auto_lock lock(&_lock);

		if (_input_type[index] == NULL)
			return 	MF_E_TRANSFORM_TYPE_NOT_SET;
		*ppmt = _input_type[index];
		(*ppmt)->AddRef();
	}
	return S_OK;
}

HRESULT solids::lib::mf::transform::merge::transform::GetOutputCurrentType(DWORD id, IMFMediaType ** ppmt)
{
	if (ppmt == NULL)
		return E_POINTER;
	if (id != BASE_OUTPUT_STREAM_ID)
		return MF_E_INVALIDSTREAMNUMBER;

	{
		solids::lib::mf::auto_lock lock(&_lock);

		if (_output_type == NULL)
			return 	MF_E_TRANSFORM_TYPE_NOT_SET;
		*ppmt = _output_type;
		(*ppmt)->AddRef();
	}
	return S_OK;
}

HRESULT solids::lib::mf::transform::merge::transform::GetInputStatus(DWORD id, DWORD * flags)
{
	if (flags == NULL)
		return E_POINTER;
	DWORD index = get_input_stream_index(id);
	if (index == NO_INDEX)
		return MF_E_INVALIDSTREAMNUMBER;
	
	{
		solids::lib::mf::auto_lock lock(&_lock);

		if (_input_updated[index] == FALSE)
			*flags = MFT_INPUT_STATUS_ACCEPT_DATA;
		else
			*flags = 0;
	}
	return S_OK;
}

HRESULT solids::lib::mf::transform::merge::transform::GetOutputStatus(DWORD * flags)
{
	if (flags == NULL)
		return E_POINTER;

	*flags = MFT_OUTPUT_STATUS_SAMPLE_READY;
	do
	{
		solids::lib::mf::auto_lock lock(&_lock);

		if (_overlapped == TRUE)
			break;
		else
		{
			for (DWORD i = 0; i < _input_stream_count; i++)
			{
				if (_input_updated[i] == FALSE)
				{
					*flags = 0;
					break;
				}
			}
		}
	} while (0);

	return S_OK;
}

HRESULT solids::lib::mf::transform::merge::transform::SetOutputBounds(LONGLONG lowerBound, LONGLONG upperBound)
{
	return E_NOTIMPL;
}

HRESULT solids::lib::mf::transform::merge::transform::ProcessEvent(DWORD isid, IMFMediaEvent* evt)
{
	return E_NOTIMPL;
}

HRESULT solids::lib::mf::transform::merge::transform::ProcessMessage(MFT_MESSAGE_TYPE msg, ULONG_PTR param)
{
	HRESULT hr = S_OK;
	switch (msg)
	{
	case MFT_MESSAGE_NOTIFY_BEGIN_STREAMING:
		::OutputDebugString(L"[ MFT ] MFT_MESSAGE_NOTIFY_BEGIN_STREAMING\n");
		break;
	case MFT_MESSAGE_NOTIFY_START_OF_STREAM:
		::OutputDebugString(L"[ MFT ] MFT_MESSAGE_NOTIFY_START_OF_STREAM");
		break;
	case MFT_MESSAGE_SET_D3D_MANAGER:
		::OutputDebugString(L"[ MFT ] MFT_MESSAGE_SET_D3D_MANAGER");
		break;
	case MFT_MESSAGE_COMMAND_FLUSH:
		::OutputDebugString(L"[ MFT ] MFT_MESSAGE_COMMAND_FLUSH");
		break;
	case MFT_MESSAGE_COMMAND_DRAIN:
		::OutputDebugString(L"[ MFT ] MFT_MESSAGE_COMMAND_DRAIN");
		break;
	case MFT_MESSAGE_NOTIFY_END_STREAMING:
		::OutputDebugString(L"[ MFT ] MFT_MESSAGE_NOTIFY_END_STREAMING");
		break;
	case MFT_MESSAGE_DROP_SAMPLES:
		::OutputDebugString(L"[ MFT ] MFT_MESSAGE_DROP_SAMPLES");
		break;
	case MFT_MESSAGE_NOTIFY_END_OF_STREAM:
		::OutputDebugString(L"[ MFT ] MFT_MESSAGE_NOTIFY_END_OF_STREAM");
		break;
	case MFT_MESSAGE_COMMAND_MARKER:
		::OutputDebugString(L"[ MFT ] MFT_MESSAGE_COMMAND_MARKER");
		break;
	}
	return S_OK;
}

HRESULT solids::lib::mf::transform::merge::transform::ProcessInput(DWORD isid, IMFSample * sample, DWORD flags)
{
	HRESULT hr = S_OK;

	if (sample == NULL)
		return E_POINTER;
	DWORD index = get_input_stream_index(isid);
	if (index == NO_INDEX)
		return MF_E_INVALIDSTREAMNUMBER;
	if (flags != 0)
		return E_INVALIDARG;
	if (_selected_only && (index != _selected_id))
		return S_OK;
#if 0
	LONGLONG timestamp;
	pSample->GetSampleTime(&timestamp);
	switch (dwInputStreamID)
	{
	case 0: log_debug("[ MFT ] [ ProcessInput ] [%s] ■ □ □ □ %d", _selected_only == true ? L"A" : L"V", (int)timestamp ); break;
	case 1: log_debug("[ MFT ] [ ProcessInput ] [%s] □ ■ □ □ %d", _selected_only == true ? L"A" : L"V", (int)timestamp ); break;
	case 2:	log_debug("[ MFT ] [ ProcessInput ] [%s] □ □ ■ □ %d", _selected_only == true ? L"A" : L"V", (int)timestamp ); break;
	case 3:	log_debug("[ MFT ] [ ProcessInput ] [%s] □ □ □ ■ %d", _selected_only == true ? L"A" : L"V", (int)timestamp ); break;
	}
#endif

	{
		solids::lib::mf::auto_lock lock(&_lock);

		DWORD flag = 0;
		GetOutputStatus(&flag);
		if (flag == MFT_OUTPUT_STATUS_SAMPLE_READY)
			return MF_E_NOTACCEPTING;

		if (_input_buffer[index] != NULL)
			solids::lib::mf::safe_release(_input_buffer[index]);

		sample->GetBufferByIndex(0, &_input_buffer[index]); 
		if (_input_updated[index] == TRUE)
		{
			_overlapped = TRUE;
		}	
		_input_updated[index] = true;
		if (index == _selected_id)
		{
			sample->GetSampleTime(&_curr_pts); //_curr_presentation_time += _stream_duration;
			sample->GetSampleDuration(&_stream_duration);
		}	
	}
	return S_OK;
}

HRESULT solids::lib::mf::transform::merge::transform::ProcessOutput(DWORD flags, DWORD outputBufferCount, MFT_OUTPUT_DATA_BUFFER * outputSamples, DWORD * status)
{
	HRESULT hr = S_OK;
	static int cnt = 0;
	if (flags != 0)
		return E_INVALIDARG;
	if (outputSamples == NULL || status == NULL)
		return E_POINTER;
	if (outputBufferCount != 1)
		return E_INVALIDARG;

	{
		solids::lib::mf::auto_lock lock(&_lock);

		LONGLONG hnsDuration = 0;
		LONGLONG hnsTime = 0;

		IMFSample* sample = NULL;

		if (_selected_only == TRUE)
		{
			if (_input_updated[_selected_id] == FALSE)
				return MF_E_TRANSFORM_NEED_MORE_INPUT;
			_input_updated[_selected_id] = FALSE;

			MFCreateSample(&sample);
			hr = sample->AddBuffer(_input_buffer[_selected_id]);
			if (FAILED(hr)) 
				return hr;
		}
		else
		{
			DWORD flag = 0;
			GetOutputStatus(&flag);
			if (flag == 0)
				return MF_E_TRANSFORM_NEED_MORE_INPUT;
			MFCreateSample(&sample);

			for (DWORD i = 0; i < _input_stream_count; i++)
			{
				_input_updated[i] = FALSE;
				hr = sample->AddBuffer(_input_buffer[i]);
				if (FAILED(hr)) 
					return hr;
			}
		}

		if (sample)
		{
			if (_overlapped == TRUE)
			{
				_overlapped = FALSE;
				sample->SetUINT32(MFSampleExtension_Discontinuity, TRUE);
			}
			outputSamples[0].pSample = sample;
			outputSamples[0].pSample->AddRef();
			outputSamples[0].pSample->SetSampleDuration(_stream_duration);
			if (_prev_pts == _curr_pts)
				_curr_pts += _stream_duration;
			//else if (_prev_presentation_time > _curr_presentation_time)
			//	_curr_presentation_time = _prev_presentation_time;
			_prev_pts = _curr_pts;
			outputSamples[0].pSample->SetSampleTime(_curr_pts);
			outputSamples[0].dwStatus = 0;
			//log_debug("[ MFT ] [ ProcessOutput ] [%s] Created : %d ", _selected_only == true ? L"A" : L"V", _curr_presentation_time / 10 / 1000);
			if (_is_first_sample)
			{
				_is_first_sample = FALSE;
				set_active_video_info(&sample);
			}
			solids::lib::mf::safe_release(sample);
		}
	}
	return S_OK;
}

HRESULT	solids::lib::mf::transform::merge::transform::is_valid_input_stream(DWORD isid) const
{
	for (DWORD i = 0 ; i < _input_stream_count ; i++)
		if(_input_stream_ids[i] == isid)
			return S_OK;
	return S_FALSE;
}

HRESULT solids::lib::mf::transform::merge::transform::is_type_acceptable(int32_t dir, DWORD id, IMFMediaType * mt) const
{	
	if (dir == solids::lib::mf::transform::merge::transform::direction_t::input)
	{
		DWORD index = get_input_stream_index(id);
		{	// Audio Decoder와의 Type Negotiation에서 SUB_TYPE 가 MFAudioFormat_PCM 가 되어야 함
			// 왜냐하면 fdk-aac encoder의 input은 PCM 형태가 되어야 함
			// Decoder에서 최초 제시하는 Type은 bitspersample : 32 sub_type : WAVE_FORMAT_IEEE_FLOAT
			GUID major_type = GUID_NULL;
			mt->GetGUID(MF_MT_MAJOR_TYPE, &major_type);
			if (major_type == MFMediaType_Audio)
			{
				GUID sub_type = GUID_NULL;
				mt->GetGUID(MF_MT_SUBTYPE, &sub_type);
				if (sub_type != MFAudioFormat_PCM)
					return MF_E_INVALIDTYPE;
			}
		}
		if (_input_type[index])
		{
			DWORD flags = 0;
			if (SUCCEEDED(mt->IsEqual(_input_type[index], &flags)))
				return S_OK;
			else
				return MF_E_INVALIDTYPE;

		}
	}
	else
	{
		if (_output_type)
		{
			DWORD flags = 0;
			if (SUCCEEDED(mt->IsEqual(_output_type, &flags)))
				return S_OK;
			else
				return MF_E_INVALIDTYPE;
		}
	}
	//TODO : specific validation is needed
	return S_OK;
}

HRESULT solids::lib::mf::transform::merge::transform::set_media_type(const int32_t dir, const DWORD id, IMFMediaType * mt)
{
	HRESULT hr = S_OK;

	DWORD index = 0 ;
	if (dir == solids::lib::mf::transform::merge::transform::direction_t::input)
		index = get_input_stream_index(id);

	if (mt)
	{	
		if (dir == solids::lib::mf::transform::merge::transform::direction_t::input)
		{
			solids::lib::mf::safe_release(_input_type[index]);
			_input_type[index] = mt;
			_input_type[index]->AddRef();
			_is_input_type_set[index] = true;
		}
		else
		{
			solids::lib::mf::safe_release(_output_type);
			_output_type = mt;
			_output_type->AddRef();
			_is_output_type_set = true;
		}
	}
	else
	{
		if (dir == solids::lib::mf::transform::merge::transform::direction_t::input)
		{
			_input_type[index] = NULL;
			_is_input_type_set[index] = FALSE;
		}
		else
		{
			_output_type = NULL;
			_is_output_type_set = FALSE;
		}
	}
	return hr;
}

DWORD solids::lib::mf::transform::merge::transform::get_input_stream_index(DWORD id) const
{
	for (DWORD i = 0; i < _input_stream_count; i++)
		if (_input_stream_ids[i] == id)
			return i;
	return NO_INDEX;
}

void solids::lib::mf::transform::merge::transform::set_active_video_info(IMFSample** sample)
{
	HRESULT hr = S_OK;
	if (_ratios == NULL)
	{
		_ratios = new double[_input_stream_count * 2];
		for (DWORD i = 0; i < _input_stream_count; i++)
		{
			UINT32 adjusted_width, adjusted_height = 0;
			hr = MFGetAttributeSize(_input_type[i], MF_MT_FRAME_SIZE, &adjusted_width, &adjusted_height);
			if (FAILED(hr)) break;

			MFVideoArea active_area = { 0, };
			hr = _input_type[i]->GetBlob(MF_MT_MINIMUM_DISPLAY_APERTURE, (UINT8*)&active_area, sizeof(MFVideoArea), NULL);

			_ratios[2 * i] = (double)active_area.Area.cx / adjusted_width;
			_ratios[2 * i + 1] = (double)active_area.Area.cy / adjusted_height;
		}
		if (SUCCEEDED(hr))
			(*sample)->SetBlob(MF_MT_ACTIVE_IMAGE_RATIO, (const UINT8*)_ratios, sizeof(double) * 2 * _input_stream_count);
	}
}

HRESULT solids::lib::mf::transform::merge::transform::SetEnableID(UINT id)
{
	solids::lib::mf::auto_lock lock(&_lock);
	{
		DWORD index = get_input_stream_index(id);
		if (index == NO_INDEX)
			return MF_E_INVALIDSTREAMNUMBER;
		_selected_id = index;
	}
	return S_OK;
}
HRESULT solids::lib::mf::transform::merge::transform::SetSeletedOnly(BOOL enable)
{
	_selected_only = enable;
	return S_OK;
}