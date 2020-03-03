#ifndef _MF_TOPOLOGY_BUILDER_H_
#define _MF_TOPOLOGY_BUILDER_H_

#include "mf_base.h"
#include "sld_mf_plain_controller.h"

namespace solids
{
	namespace lib
	{
		namespace mf
		{
			namespace control
			{
				namespace plain
				{
					namespace topology
					{
						class builder
						{
						public:
							static HRESULT create_source(const wchar_t * url, IMFMediaSource ** mediaSource);

							static HRESULT add_branch_to_partial_topology(IMFTopology* topology, IMFMediaSource* mediaSource, DWORD streamIndex, IMFPresentationDescriptor* presentDescriptor, solids::lib::mf::control::plain::controller::context_t* context, IUnknown** pDeviceManager);

							static HRESULT create_stream_source_node(IMFMediaSource* mediaSource, IMFPresentationDescriptor* presentDescriptor, IMFStreamDescriptor* streamDescriptor, IMFTopologyNode** node);
							static HRESULT create_video_renderer_activate(HWND hwnd, IMFActivate** activate);
							static HRESULT create_audio_renderer_activate(IMFActivate** activate);
							static HRESULT create_merger_activate(IMFActivate** activate);

							static HRESULT create_audio_decoder_node(IMFMediaType* mediaType, IMFTopologyNode** node);
							static HRESULT create_video_decoder_node(IMFMediaType* mediaType, ULONG_PTR pDeviceManager, IMFTopologyNode** node);
							static HRESULT create_stream_sink_node(IUnknown* streamSink, DWORD streamNumber, IMFTopologyNode** node);

						private:
							static HRESULT find_video_decoder(REFCLSID subtype, IMFTransform** decoder);
							static HRESULT find_audio_decoder(REFCLSID subtype, IMFTransform** decoder);
						};
					};
				};
			};
		};
	};
};

#endif