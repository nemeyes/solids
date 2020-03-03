#ifndef _SLD_MF_RTSP_SOURCE_H_
#define _SLD_MF_RTSP_SOURCE_H_

#include <mf_base.h>

#if defined(WITH_FFRTSPCLIENT)
#include <sld_ff_rtsp_client.h>
#pragma comment(lib, "FFRTSPClient.lib")
#else
#include <sld_rtsp_client.h>
#pragma comment(lib, "RTSPClient.lib")
#endif
#include "mf_rtsp_source_async_operation.h"

namespace sld
{
	namespace lib
	{
		namespace mf
		{
			namespace source
			{
				namespace rtsp
				{
					class stream;
					class source
						: sld::lib::mf::base
						, sld::lib::mf::refcount_object
						, public IMFMediaSource
#if defined(WITH_FFRTSPCLIENT)
						, public sld::lib::net::rtsp::ff::client
#else
						, public sld::lib::net::rtsp::client
#endif
					{
					public:
						static const int32_t sample_queue_size = 2;
						typedef struct _state_t
						{
							static const int32_t invalid = 0; // Initial state. Have not started opening the stream.
							static const int32_t opening = 1; // BeginOpen is in progress.
							static const int32_t stopped = 2;
							static const int32_t paused = 3;
							static const int32_t started = 4;
							static const int32_t shutdown = 5;
						} state_t;

						typedef struct _bitstream_t
						{
							uint8_t*	bitstream;
							int32_t		bitstream_size;
							_bitstream_t(int32_t size)
								: bitstream_size(size)
							{
								bitstream = new uint8_t[bitstream_size];
							}

							~_bitstream_t(void)
							{
								delete[] bitstream;
							}
						} bitstream_t;

						static HRESULT create_instance(sld::lib::mf::source::rtsp::source ** source);

						// IUnknown
						STDMETHODIMP	QueryInterface(REFIID iid, void** ppv);
						STDMETHODIMP_(ULONG) AddRef(void);
						STDMETHODIMP_(ULONG) Release(void);

						// IMFMediaEventGenerator
						STDMETHODIMP BeginGetEvent(IMFAsyncCallback * callback, IUnknown * state);
						STDMETHODIMP EndGetEvent(IMFAsyncResult * result, IMFMediaEvent ** evt);
						STDMETHODIMP GetEvent(DWORD flags, IMFMediaEvent ** evt);
						STDMETHODIMP QueueEvent(MediaEventType met, REFGUID guid_ext_type, HRESULT status, const PROPVARIANT * value);

						// IMFMediaSource
						STDMETHODIMP CreatePresentationDescriptor(IMFPresentationDescriptor ** pd);
						STDMETHODIMP GetCharacteristics(DWORD * characteristics);
						STDMETHODIMP Pause(void);
						STDMETHODIMP Shutdown(void);
						STDMETHODIMP Start(IMFPresentationDescriptor * pd, const GUID * guid_time_fmt, const PROPVARIANT * start_position);
						STDMETHODIMP Stop(void);

						HRESULT begin_open(LPCWSTR url, IMFAsyncCallback * callback, IUnknown * state);
						HRESULT end_open(IMFAsyncResult * result);

						HRESULT initialize(void);
						//HRESULT release(void);

						HRESULT queue_async_operation(sld::lib::mf::source::rtsp::async_operation * aop);
						HRESULT queue_async_operation(int32_t op);

						HRESULT get_video_sample(IMFSample ** sample);
						HRESULT get_audio_sample(IMFSample ** sample);

					private:
						source(void);
						~source(void);

						HRESULT dispatch_workitem_cb(IMFAsyncResult * ar);
						HRESULT check_shutdown(void) const;
						HRESULT is_initialized(void) const;

						HRESULT validate_presentation_descriptor(IMFPresentationDescriptor * pd);
						HRESULT initialize_presentation_descriptor(void);

						HRESULT create_audio_aac_mediatype(IMFMediaType ** mt, uint8_t * extradata, int32_t extradata_size, int32_t samplerate, int32_t sampleformat, int32_t channels);
						HRESULT create_audio_mp3_mediatype(IMFMediaType ** mt, int32_t samplerate, int32_t sampleformat, int32_t channels);
						HRESULT create_audio_ac3_mediatype(IMFMediaType ** mt, int32_t samplerate, int32_t sampleformat, int32_t channels);
						HRESULT create_video_h264_mediatype(IMFMediaType ** mt, uint8_t * extradata, int32_t extradata_size);
						HRESULT create_video_hevc_mediatype(IMFMediaType ** mt, uint8_t * extradata_, int32_t extradata_size);

						HRESULT create_video_mediatype(int32_t codec, uint8_t * extradata, int32_t extradata_size);
						HRESULT create_audio_mediatype(int32_t codec, int32_t samplerate, int32_t sampleformat, int32_t channels, uint8_t * extradata, int32_t extradata_size);

						HRESULT create_video_sample(IMFSample** sample, const uint8_t * extradata, int32_t extradata_size, const uint8_t* bytes, int32_t nbytes, long long pts, long long duration);
						HRESULT create_video_sample(IMFSample ** sample, const uint8_t * bytes, int32_t nbytes, long long pts, long long duration);
						HRESULT create_audio_sample(IMFSample ** sample, const uint8_t * bytes, int32_t nbytes, long long pts, long long duration);

						HRESULT complete_open(const HRESULT hr_status);
						void	error_handle(const HRESULT hr);
						void	release_samples(void);

						void	on_begin_video(int32_t codec, uint8_t* extradata, int32_t extradata_size);
						void	on_recv_video(uint8_t* bytes, int32_t nbytes, long long pts, long long duration);
						void	on_end_video(void);

						void	on_begin_audio(int32_t codec, uint8_t* extradata, int32_t extradata_size, int32_t samplerate, int32_t channels);
						void	on_recv_audio(uint8_t* bytes, int32_t nbytes, long long pts, long long duration);
						void	on_end_audio(void);

						long long elapsed_100nanoseconds(void);
						
					private:
						critical_section		_lock;
						int32_t					_state;
						IMFMediaEventQueue *	_event_queue;
						std::map<int32_t, ATL::CAdapt<ATL::CComPtr<sld::lib::mf::source::rtsp::stream>>>				_streams;
						std::map<int32_t, std::vector<ATL::CAdapt<ATL::CComPtr<IMFSample>>>*>							_samples;
						//std::map<int32_t, std::queue<bitstream_t*>*>													_bitstreams;


						critical_section		_stream_lock;
						SRWLOCK					_samples_lock;

						int32_t					_video_codec;

						BOOL					_video_wait_idr;
						LONGLONG				_video_start_time;

						LARGE_INTEGER			_frequency;
						LARGE_INTEGER			_begin_elapsed_microseconds;

						DWORD					_work_queue_id;
						async_callback<sld::lib::mf::source::rtsp::source>	_work_queue_cb;
						ATL::CComPtr<IMFPresentationDescriptor>					_pd;
						ATL::CComPtr<IMFAsyncResult>							_begin_open_result;

						BOOL					_intialized_pd;
						//BOOL					_disable_audio;

						uint8_t					_extradata[MAX_PATH];
						int32_t					_extradata_size;
					};
				};
			};
		};
	};
};

#endif