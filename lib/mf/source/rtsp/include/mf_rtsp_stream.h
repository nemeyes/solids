#ifndef _SLD_MF_RTSP_STREAM_H_
#define _SLD_MF_RTSP_STREAM_H_

#include "mf_rtsp_source.h"

namespace solids
{
	namespace lib
	{
		namespace mf
		{
			namespace source
			{
				namespace rtsp
				{
					class stream
						: solids::lib::mf::base
						, solids::lib::mf::refcount_object
						, public IMFMediaStream
					{
					public:
						typedef struct _state_t
						{
							static const int32_t not_set = 0; // Initial state. Have not started opening the stream.
							static const int32_t ready = 1; // BeginOpen is in progress.
							static const int32_t started = 2;
							static const int32_t stopped = 3;
							static const int32_t paused = 4;
							static const int32_t finalized = 5;
						} state_t;

						stream(void);
						~stream(void);

						// IUnknown
						STDMETHODIMP QueryInterface(REFIID iid, void** ppv);
						STDMETHODIMP_(ULONG) AddRef(void);
						STDMETHODIMP_(ULONG) Release(void);

						// IMFMediaEventGenerator
						STDMETHODIMP BeginGetEvent(IMFAsyncCallback * callback, IUnknown * unk);
						STDMETHODIMP EndGetEvent(IMFAsyncResult * result, IMFMediaEvent ** evt);
						STDMETHODIMP GetEvent(DWORD flags, IMFMediaEvent ** evt);
						STDMETHODIMP QueueEvent(MediaEventType met, REFGUID guid_ext_type, HRESULT status, const PROPVARIANT * value);

						// IMFMediaStream
						STDMETHODIMP GetMediaSource(IMFMediaSource ** source);
						STDMETHODIMP GetStreamDescriptor(IMFStreamDescriptor ** sd);
						STDMETHODIMP RequestSample(IUnknown * token);


						HRESULT initialize(solids::lib::mf::source::rtsp::source * source, IMFStreamDescriptor * sd, int32_t type);
						HRESULT release(void);

						// Other methods (called by source)
						HRESULT activate(BOOL active);
						BOOL	is_active(void) const;

						HRESULT start(const PROPVARIANT & start);
						HRESULT pause(void);
						HRESULT stop(void);


						void	set_buffering(BOOL buffering);
						BOOL	is_buffering(void) const;
						int32_t type(void) { return _type; }


					private:
						HRESULT check_shutdown(void) const;

					private:
						critical_section		_lock;
						int32_t					_state;
						IMFMediaEventQueue *	_event_queue;
						IMFStreamDescriptor *	_sd;
						solids::lib::mf::source::rtsp::source * _source;
						BOOL	_active;
						BOOL	_eos;

						int32_t _type;
						BOOL	_buffering;
					};
				};
			};
		};
	};
};

#endif