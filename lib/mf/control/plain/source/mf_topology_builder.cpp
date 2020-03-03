#include "mf_topology_builder.h"

HRESULT solids::lib::mf::control::plain::topology::builder::create_source(const wchar_t * url, IMFMediaSource ** mediaSource)
{
	HRESULT hr = S_OK;
	MF_OBJECT_TYPE objType = MF_OBJECT_INVALID;
	ATL::CComPtr<IMFSourceResolver> srcResolver = NULL;
	ATL::CComPtr<IUnknown> src;

	wchar_t url2[MAX_PATH] = { 0 };
	if (!wcsncmp(url, L"rtsp", wcslen(L"rtsp")))
	{
		url2[0] = 'p';
		url2[1] = 'l';
		url2[2] = 's';
		wcsncpy_s(url2 + 3, MAX_PATH - 3, url, MAX_PATH - 3);
	}
	else
	{
		wcsncpy_s(url2, url, wcslen(url));
	}
	do
	{
		hr = MFCreateSourceResolver(&srcResolver);
		BREAK_ON_FAIL(hr);

		hr = srcResolver->CreateObjectFromURL(url2, MF_RESOLUTION_MEDIASOURCE | MF_RESOLUTION_CONTENT_DOES_NOT_HAVE_TO_MATCH_EXTENSION_OR_MIME_TYPE, NULL, &objType, &src);
		
		BREAK_ON_FAIL(hr);

		hr = src->QueryInterface(IID_PPV_ARGS(mediaSource));
		BREAK_ON_NULL(mediaSource, E_NOINTERFACE);
	} while (0);

	return hr;
}

HRESULT solids::lib::mf::control::plain::topology::builder::add_branch_to_partial_topology(IMFTopology* topology, IMFMediaSource* mediaSource, DWORD streamIndex, IMFPresentationDescriptor* presentDescriptor, solids::lib::mf::control::plain::controller::context_t* context, IUnknown** pDeviceManager)
{
	HRESULT hr = S_OK;
	ATL::CComPtr<IMFStreamDescriptor> streamDescriptor = NULL;
	ATL::CComPtr<IMFTopologyNode> srcNode = NULL;
	ATL::CComPtr<IMFTopologyNode> trnsNode = NULL;
	ATL::CComPtr<IMFTopologyNode> snkNode = NULL;
	BOOL streamSelected = FALSE;

	do
	{
		BREAK_ON_NULL(topology, E_UNEXPECTED);

		hr = presentDescriptor->GetStreamDescriptorByIndex(streamIndex, &streamSelected, &streamDescriptor);
		BREAK_ON_FAIL(hr);

		if (streamSelected)
		{
			/////////////////create a source node for this stream///////////////////
			hr = solids::lib::mf::control::plain::topology::builder::create_stream_source_node(mediaSource, presentDescriptor, streamDescriptor, &srcNode);
			
						/////////////////create a output node for renderer///////////////////
			ATL::CComPtr<IMFMediaTypeHandler> mediaTypeHandler = NULL;
			ATL::CComPtr<IMFMediaType> mediaType;
			ATL::CComPtr<IMFActivate> rendererActivate = NULL;
			GUID majorType = GUID_NULL;

			do
			{
				hr = streamDescriptor->GetMediaTypeHandler(&mediaTypeHandler);
				BREAK_ON_FAIL(hr);

				hr = mediaTypeHandler->GetCurrentMediaType(&mediaType);
				BREAK_ON_FAIL(hr);
				
				hr = mediaType->GetMajorType(&majorType);
				BREAK_ON_FAIL(hr);
				// Create an IMFActivate controller object for the renderer, based on the media type.
				if (majorType == MFMediaType_Audio && context->hwnd != NULL)
				{
#if 1
					hr = MFCreateAudioRendererActivate(&rendererActivate);
					// create the node which will represent the renderer
					hr = MFCreateTopologyNode(MF_TOPOLOGY_OUTPUT_NODE, &snkNode);
					BREAK_ON_FAIL(hr);

					// store the IActivate object in the sink node - it will be extracted later by the media session during the topology render phase.
					hr = snkNode->SetObject(rendererActivate);
					BREAK_ON_FAIL(hr);

					hr = create_audio_decoder_node(mediaType, &trnsNode);
					BREAK_ON_FAIL(hr);
#else
					hr = solids::lib::mf::control::plain::topology::builder::create_audio_renderer_activate(&rendererActivate);
					BREAK_ON_FAIL(hr);

					ATL::CComPtr<IMFMediaSink> media_sink;
					hr = rendererActivate->ActivateObject(IID_PPV_ARGS(&media_sink));
					BREAK_ON_FAIL(hr);

					ATL::CComPtr<IMFStreamSink> stream_sink;
					DWORD stream_sink_count = 0;
					hr = media_sink->GetStreamSinkCount(&stream_sink_count);
					BREAK_ON_FAIL(hr);
					hr = media_sink->GetStreamSinkByIndex((stream_sink_count - 1), &stream_sink);
					BREAK_ON_FAIL(hr);

					hr = solids::lib::mf::control::plain::topology::builder::create_audio_decoder_node(mediaType, &trnsNode);
					BREAK_ON_FAIL(hr);

#if 0
					hr = mf_topology_builder::create_stream_sink_node(stream_sink, stream_index, &sink_node);
					BREAK_ON_FAIL(hr);
#else
					DWORD id = 0;
					hr = stream_sink->GetIdentifier(&id);
					BREAK_ON_FAIL(hr);

					hr = solids::lib::mf::control::plain::topology::builder::create_stream_sink_node(stream_sink, id, &snkNode);
					BREAK_ON_FAIL(hr);
					
					//hr = sink_node->SetUINT32(MF_TOPONODE_CONNECT_METHOD, MF_CONNECT_ALLOW_CONVERTER);
					//BREAK_ON_FAIL(hr);
#endif
#endif
				}
				else if (majorType == MFMediaType_Video && context->hwnd != NULL)
				{
#if 0
					hr = MFCreateVideoRendererActivate(context->hwnd, &rendererActivate);
					BREAK_ON_FAIL(hr);

					// create the node which will represent the renderer
					hr = MFCreateTopologyNode(MF_TOPOLOGY_OUTPUT_NODE, &snkNode);
					BREAK_ON_FAIL(hr);

					// store the IActivate object in the sink node - it will be extracted later by the media session during the topology render phase.
					hr = snkNode->SetObject(rendererActivate);
					BREAK_ON_FAIL(hr);

					hr = create_video_decoder_node(mediaType, NULL, &trnsNode);
					BREAK_ON_FAIL(hr);
#else
					hr = solids::lib::mf::control::plain::topology::builder::create_video_renderer_activate(context->hwnd, &rendererActivate);
					BREAK_ON_FAIL(hr);

					ATL::CComPtr<IMFMediaSink> mediaSink;
					hr = rendererActivate->ActivateObject(IID_PPV_ARGS(&mediaSink));
					BREAK_ON_FAIL(hr);
			

					DWORD charcteristic;
					mediaSource->GetCharacteristics(&charcteristic);
					if (charcteristic & MFMEDIASOURCE_CAN_SEEK)
					{//get duration from source node & set duration
#if 0						
						CComPtr<IMFMediaSource> source;
						CComPtr<IMFPresentationDescriptor> presentation_descriptor;
						hr = source_node->GetUnknown(MF_TOPONODE_SOURCE, IID_IMFMediaSource, (void**)&source);
						BREAK_ON_FAIL(hr);
						hr = source->CreatePresentationDescriptor(&presentation_descriptor);
						BREAK_ON_FAIL(hr);
						MFTIME duration;
						hr = presentation_descriptor->GetUINT64(MF_PD_DURATION, (UINT64*)&duration);
						//hr = presentation_descriptor->GetUINT64(MF_PD_DURATION, (UINT64*)&config->duration);
						BREAK_ON_FAIL(hr);
						config->duration = (int32_t)(duration / (10 * 1000 * 1000));
						hr = vcodec_configurator->SetDuration(config->duration);
						BREAK_ON_FAIL(hr);
#else
						MFTIME duration;
						hr = presentDescriptor->GetUINT64(MF_PD_DURATION, (UINT64*)&duration);
						BREAK_ON_FAIL(hr);
#endif
					}

					ATL::CComPtr<IMFGetService> getService;
					hr = mediaSink->QueryInterface(IID_PPV_ARGS(&getService));
					BREAK_ON_FAIL(hr);

					ATL::CComPtr<IMFStreamSink> streamSink;
					DWORD streamSinkCount = 0;
					hr = mediaSink->GetStreamSinkCount(&streamSinkCount);
					BREAK_ON_FAIL(hr);
					hr = mediaSink->GetStreamSinkByIndex((streamSinkCount - 1), &streamSink);
					BREAK_ON_FAIL(hr);

					hr = getService->GetService(MR_VIDEO_ACCELERATION_SERVICE, IID_IMFDXGIDeviceManager, (void**)pDeviceManager);
					BREAK_ON_FAIL(hr);

					LONG_PTR ptrDeviceManager = reinterpret_cast<ULONG_PTR>(*pDeviceManager);
					hr = solids::lib::mf::control::plain::topology::builder::create_video_decoder_node(mediaType, ptrDeviceManager, &trnsNode);
					BREAK_ON_FAIL(hr);

#if 1
					hr = create_stream_sink_node(streamSink, streamIndex, &snkNode);
					BREAK_ON_FAIL(hr);
#else
					DWORD id = 0;
					hr = streamSink->GetIdentifier(&id);
					BREAK_ON_FAIL(hr);

					hr = create_stream_sink_node(streamSink, id, &snkNode);
					BREAK_ON_FAIL(hr);
#endif
#endif
				}
				else
				{
					hr = E_FAIL;
				}
				BREAK_ON_FAIL(hr);
			} while (0);
			BREAK_ON_FAIL(hr);

			if (srcNode)
			{
				hr = topology->AddNode(srcNode);
				BREAK_ON_FAIL(hr);
			}

			if (trnsNode)
			{
				hr = topology->AddNode(trnsNode);
				BREAK_ON_FAIL(hr);
			}

			if (snkNode)
			{
				hr = topology->AddNode(snkNode);
				BREAK_ON_FAIL(hr);
			}

			// Connect the source node to the output node.  The topology will find the
			// intermediate nodes needed to convert media types.
			if (srcNode && trnsNode && snkNode)
			{
				hr = srcNode->ConnectOutput(0, trnsNode, 0);
				hr = trnsNode->ConnectOutput(0, snkNode, 0);
			}
			else if (srcNode && snkNode)
			{
				hr = srcNode->ConnectOutput(0, snkNode, 0);
			}
		}
	} while (0);

	if (FAILED(hr))
	{
		hr = presentDescriptor->DeselectStream(streamIndex);
	}

	return hr;
}

HRESULT solids::lib::mf::control::plain::topology::builder::create_stream_source_node(IMFMediaSource * mediaSource, IMFPresentationDescriptor * presentDescriptor, IMFStreamDescriptor * streamDescriptor, IMFTopologyNode ** node)
{
	HRESULT hr = S_OK;
	do
	{
		// create a source node for this stream
		hr = MFCreateTopologyNode(MF_TOPOLOGY_SOURCESTREAM_NODE, node);
		BREAK_ON_FAIL(hr);

		// associate the node with the souce by passing in a pointer to the media source and indicating that it is the source
		hr = (*node)->SetUnknown(MF_TOPONODE_SOURCE, mediaSource);
		BREAK_ON_FAIL(hr);

		// set the node presentation descriptor attribute of the node by passing in a pointer to the presentation descriptor
		hr = (*node)->SetUnknown(MF_TOPONODE_PRESENTATION_DESCRIPTOR, presentDescriptor);
		BREAK_ON_FAIL(hr);

		// set the node stream descriptor attribute by passing in a pointer to the stream descriptor
		hr = (*node)->SetUnknown(MF_TOPONODE_STREAM_DESCRIPTOR, streamDescriptor);
		BREAK_ON_FAIL(hr);

		hr = (*node)->SetUINT32(MF_TOPONODE_CONNECT_METHOD, MF_CONNECT_ALLOW_DECODER);
		BREAK_ON_FAIL(hr);

	} while (0);

	return hr;
}

HRESULT solids::lib::mf::control::plain::topology::builder::create_video_renderer_activate(HWND hwnd, IMFActivate ** activate)
{
	if (activate == NULL)
		return E_POINTER;
	
	HMODULE rendererDll = ::LoadLibrary(L"MFD3D11Renderer.dll");
	if (rendererDll == NULL)
	{
		DWORD err = ::GetLastError();
		return E_FAIL;
	}
	
	LPCSTR fnName = "CreateMFD3D11RendererActivate";
	FARPROC createVideoRendererActivateProc = ::GetProcAddress(rendererDll, fnName);
	if (createVideoRendererActivateProc == NULL)
		return E_FAIL;

	typedef HRESULT(STDAPICALLTYPE* LPCreateVideoRendererActivate)(HWND, IMFActivate**);
	LPCreateVideoRendererActivate createVideoRendererActivate = reinterpret_cast<LPCreateVideoRendererActivate> (createVideoRendererActivateProc);
	HRESULT hr = createVideoRendererActivate(hwnd, activate);
	return hr;
}

HRESULT solids::lib::mf::control::plain::topology::builder::create_audio_renderer_activate(IMFActivate ** activate)
{
	if (activate == NULL)
		return E_POINTER;
	
	HMODULE rendererDll = ::LoadLibrary(L"sld_mf_audio_renderer.dll");
	if (rendererDll == NULL)
	{
		DWORD err = ::GetLastError();
		return E_FAIL;
	}

	LPCSTR fnName = "create_mf_audio_renderer_activate";
	FARPROC createAudioRendererActivateProc = ::GetProcAddress(rendererDll, fnName);
	if (createAudioRendererActivateProc == NULL)
		return E_FAIL;

	typedef HRESULT(STDAPICALLTYPE* LPCreateAudioRendererActivate)(IMFActivate**);
	LPCreateAudioRendererActivate creatAudioRendererActivate = reinterpret_cast<LPCreateAudioRendererActivate>(createAudioRendererActivateProc);
	HRESULT hr = creatAudioRendererActivate(activate);

	return hr;
}

HRESULT solids::lib::mf::control::plain::topology::builder::create_merger_activate(IMFActivate ** activate)
{
	if (activate == NULL)
		return E_POINTER;
	HMODULE rendererDll = ::LoadLibrary(L"sld_mf_merger.dll");
	if (rendererDll == NULL)
	{
		DWORD err = ::GetLastError();
		return E_FAIL;
	}

	LPCSTR fnName = "create_mf_merger_activate";
	FARPROC createMergerActivateProc = ::GetProcAddress(rendererDll, fnName);
	if (createMergerActivateProc == NULL)
		return E_FAIL;

	typedef HRESULT(STDAPICALLTYPE* LPCreateMergerActivate)(IMFActivate**);
	LPCreateMergerActivate createMergerActivate = reinterpret_cast<LPCreateMergerActivate>(createMergerActivateProc);
	HRESULT hr = createMergerActivate(activate);

	return hr;
}
HRESULT solids::lib::mf::control::plain::topology::builder::create_video_decoder_node(IMFMediaType * mediaType, ULONG_PTR pDeviceManager, IMFTopologyNode ** node)
{
	HRESULT hr;

	GUID subtype;
	hr = mediaType->GetGUID(MF_MT_SUBTYPE, &subtype);
	RETURN_ON_FAIL(hr);

	ATL::CComPtr<IMFTransform> decoderTransform;

	if (IsEqualGUID(subtype, MFVideoFormat_HEVC))
	{
		::OutputDebugString(L"MFVideoFormat_HEVC\n");
	}
	else if(IsEqualGUID(subtype, MFVideoFormat_HEVC_ES))
	{
		::OutputDebugString(L"MFVideoFormat_HEVC_ES\n");
	}
	/*
	UINT32 width = 0;
	UINT32 height = 0;
	hr = MFGetAttributeSize(mediaType, MF_MT_FRAME_SIZE, &width, &height);
	RETURN_ON_FAIL(hr);
	*/

	hr = find_video_decoder(subtype, &decoderTransform);
	RETURN_ON_FAIL(hr);

	ATL::CComPtr<IMFAttributes> decoderAttribute;
	hr = decoderTransform->GetAttributes(&decoderAttribute);
	RETURN_ON_FAIL(hr);

	UINT32 transform_async = 0;
	hr = decoderAttribute->GetUINT32(MF_TRANSFORM_ASYNC, &transform_async);
	if (SUCCEEDED(hr) && transform_async == TRUE)
	{
		hr = decoderAttribute->SetUINT32(MF_TRANSFORM_ASYNC_UNLOCK, TRUE);
		RETURN_ON_FAIL(hr);
	}

	if (pDeviceManager != NULL)
	{
		ATL::CComPtr<IUnknown> device_manager_unknown = reinterpret_cast<IUnknown*>(pDeviceManager);
		ATL::CComPtr<IUnknown> dxgi_device_manager;
		CLSID d3d_aware_attribute;
		hr = device_manager_unknown->QueryInterface(IID_IMFDXGIDeviceManager, (void**)(&dxgi_device_manager));
		if (SUCCEEDED(hr))
			d3d_aware_attribute = MF_SA_D3D11_AWARE;
		else
			d3d_aware_attribute = MF_SA_D3D_AWARE;

		UINT32 d3d_aware;
		hr = decoderAttribute->GetUINT32(d3d_aware_attribute, &d3d_aware);
		if (SUCCEEDED(hr) && d3d_aware != 0)
		{
			hr = decoderTransform->ProcessMessage(MFT_MESSAGE_SET_D3D_MANAGER, pDeviceManager);
			RETURN_ON_FAIL(hr);
		}
	}

	hr = decoderTransform->SetInputType(0, mediaType, 0);
	RETURN_ON_FAIL(hr);

	hr = MFCreateTopologyNode(MF_TOPOLOGY_TRANSFORM_NODE, node);
	RETURN_ON_FAIL(hr);

	hr = (*node)->SetObject(decoderTransform);
	RETURN_ON_FAIL(hr);

	hr = (*node)->SetUINT32(MF_TOPONODE_CONNECT_METHOD, MF_CONNECT_ALLOW_CONVERTER);
	RETURN_ON_FAIL(hr);

	return hr;
}

HRESULT solids::lib::mf::control::plain::topology::builder::create_audio_decoder_node(IMFMediaType * mediaType, IMFTopologyNode ** node)
{
	HRESULT hr;
	
	GUID subtype;
	hr = mediaType->GetGUID(MF_MT_SUBTYPE, &subtype);
	RETURN_ON_FAIL(hr);

	ATL::CComPtr<IMFTransform> decoderTransform;

	hr = find_audio_decoder(subtype, &decoderTransform);
	RETURN_ON_FAIL(hr);

	ATL::CComPtr<IMFAttributes> decoderAttribute;
	hr = decoderTransform->GetAttributes(&decoderAttribute);
	if (SUCCEEDED(hr))
	{
		UINT32 transform_async = 0;
		hr = decoderAttribute->GetUINT32(MF_TRANSFORM_ASYNC, &transform_async);
		if (SUCCEEDED(hr) && transform_async == TRUE)
		{
			hr = decoderAttribute->SetUINT32(MF_TRANSFORM_ASYNC_UNLOCK, TRUE);
			RETURN_ON_FAIL(hr);
		}
	}

	hr = decoderTransform->SetInputType(0, mediaType, 0);
	RETURN_ON_FAIL(hr);

	hr = MFCreateTopologyNode(MF_TOPOLOGY_TRANSFORM_NODE, node);
	RETURN_ON_FAIL(hr);

	hr = (*node)->SetObject(decoderTransform);
	RETURN_ON_FAIL(hr);

	hr = (*node)->SetUINT32(MF_TOPONODE_CONNECT_METHOD, MF_CONNECT_ALLOW_CONVERTER);
	RETURN_ON_FAIL(hr);

	return hr;
}

HRESULT solids::lib::mf::control::plain::topology::builder::create_stream_sink_node(IUnknown * streamSink, DWORD streamNumber, IMFTopologyNode ** node)
{
	HRESULT hr;
	hr = MFCreateTopologyNode(MF_TOPOLOGY_OUTPUT_NODE, node);
	RETURN_ON_FAIL(hr);

	hr = (*node)->SetObject(streamSink);
	RETURN_ON_FAIL(hr);

	hr = (*node)->SetUINT32(MF_TOPONODE_STREAMID, streamNumber);
	RETURN_ON_FAIL(hr);

	hr = (*node)->SetUINT32(MF_TOPONODE_NOSHUTDOWN_ON_REMOVE, FALSE);
	RETURN_ON_FAIL(hr);

	return hr;
}

HRESULT solids::lib::mf::control::plain::topology::builder::find_video_decoder(REFCLSID subtype, IMFTransform ** decoder)
{
	HRESULT hr;
	UINT32 flags = MFT_ENUM_FLAG_SORTANDFILTER;

	MFT_REGISTER_TYPE_INFO inputRegisterTypeInfo = { MFMediaType_Video, subtype };
	flags |= MFT_ENUM_FLAG_SYNCMFT;
	flags |= MFT_ENUM_FLAG_ASYNCMFT;
	flags |= MFT_ENUM_FLAG_HARDWARE;

	IMFActivate ** activate = NULL;
	UINT32 numberOfRegisteredDecoders = 0;

	const CLSID supported_output_subtypes[] = { MFVideoFormat_NV12, MFVideoFormat_P010 };
	BOOL foundDecoder = FALSE;
	for (UINT32 x = 0; !foundDecoder && x < ARRAYSIZE(supported_output_subtypes); x++)
	{
		MFT_REGISTER_TYPE_INFO outputRegisterTypeInfo = { MFMediaType_Video, supported_output_subtypes[x] };
		hr = MFTEnumEx(MFT_CATEGORY_VIDEO_DECODER, flags, &inputRegisterTypeInfo, &outputRegisterTypeInfo, &activate, &numberOfRegisteredDecoders);
		RETURN_ON_FAIL(hr);

		if (SUCCEEDED(hr) && (numberOfRegisteredDecoders == 0))
			hr = MF_E_TOPO_CODEC_NOT_FOUND;

		if (SUCCEEDED(hr))
		{
			hr = activate[0]->ActivateObject(IID_PPV_ARGS(decoder));
			foundDecoder = TRUE;
		}
		for (UINT32 y = 0; y < numberOfRegisteredDecoders; y++)
		{
			activate[y]->Release();
		}
		CoTaskMemFree(activate);
	}

	if (!foundDecoder)
		return E_FAIL;

	return S_OK;
}

HRESULT solids::lib::mf::control::plain::topology::builder::find_audio_decoder(REFCLSID subtype, IMFTransform ** decoder)
{
	HRESULT hr;
	UINT32 flags = MFT_ENUM_FLAG_SORTANDFILTER;

	MFT_REGISTER_TYPE_INFO inputRegisterTypeInfo = { MFMediaType_Audio, subtype };
	flags |= MFT_ENUM_FLAG_SYNCMFT;
	flags |= MFT_ENUM_FLAG_ASYNCMFT;
	flags |= MFT_ENUM_FLAG_HARDWARE;

	IMFActivate ** activate = NULL;
	UINT32 numberOfRegisteredDecoders = 0;

	const CLSID supportedOutputSubtypes[] = { MFAudioFormat_PCM/*MFAudioFormat_AAC*/ };
	BOOL foundDecoder = FALSE;
	for (UINT32 x = 0; !foundDecoder && x < ARRAYSIZE(supportedOutputSubtypes); x++)
	{
		MFT_REGISTER_TYPE_INFO outputRegisterTypeInfo = { MFMediaType_Audio, supportedOutputSubtypes[x] };
		hr = MFTEnumEx(MFT_CATEGORY_AUDIO_DECODER, flags, &inputRegisterTypeInfo, &outputRegisterTypeInfo, &activate, &numberOfRegisteredDecoders);
		RETURN_ON_FAIL(hr);

		if (SUCCEEDED(hr) && (numberOfRegisteredDecoders == 0))
			hr = MF_E_TOPO_CODEC_NOT_FOUND;

		if (SUCCEEDED(hr))
		{
			hr = activate[0]->ActivateObject(IID_PPV_ARGS(decoder));
			foundDecoder = TRUE;
		}
		for (UINT32 y = 0; y < numberOfRegisteredDecoders; y++)
		{
			activate[y]->Release();
		}
		CoTaskMemFree(activate);
	}

	if (!foundDecoder)
		return E_FAIL;

	return S_OK;
}
