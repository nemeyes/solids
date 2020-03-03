#ifndef _MF_PLAIN_CONTROLLER_H_
#define _MF_PLAIN_CONTROLLER_H_

#include "sld_mf_plain_controller.h"
#include <mf_base.h>

namespace sld
{
	namespace lib
	{
		namespace mf
		{
			namespace control
			{
				namespace plain
				{
					class controller::core : public IMFAsyncCallback
					{
						typedef struct _playback_rate_t
						{
							static const int32_t normal_backward = -1;
							static const int32_t stopped = 0;
							static const int32_t normal_forward = 1;
						} playback_rate_t;

					public:
						core(void);
						virtual ~core(void);

						// Playback control
						int32_t open(sld::lib::mf::control::plain::controller::context_t * context);
						int32_t close(void);
						int32_t play(void);
						int32_t pause(void);
						int32_t stop(void);
						int32_t state(void) const;

						STDMETHODIMP QueryInterface(REFIID riid, void** ppv);
						STDMETHODIMP_(ULONG) AddRef(void);
						STDMETHODIMP_(ULONG) Release(void);

						STDMETHODIMP GetParameters(DWORD* flags, DWORD* queue) { return E_NOTIMPL; }
						STDMETHODIMP Invoke(IMFAsyncResult* async_result);


					private:
						HRESULT create_session(void);
						HRESULT close_session(void);
						HRESULT start_session(void);
						HRESULT process_event(CComPtr<IMFMediaEvent>& media_event);

						// Media event handlers
						HRESULT topology_ready_cb(void);
						HRESULT presentation_ended_cb(void);

						HRESULT shutdown_source(void);

#ifdef _DEBUG
						const char * event_type(DWORD event_type);
#endif

					private:
						volatile long _refcount;
						sld::lib::mf::control::plain::controller::context_t * _context;
						int32_t									_state;
						IMFMediaSession *						_session;

						IMFClock *								_clock;

						IMFTopology *							_topology;
						IMFMediaSource *						_media_source;
						IUnknown *								_device_manager;
						IMFPresentationClock *					_presentation_clock;

						int32_t									_repeat_count;
						IMFVideoDisplayControl *				_video_display;
						
						int32_t									_current_time;
						IMFRateControl *						_rate_control;
						IMFRateSupport *						_rate_support;

						BOOL									_thinning;
						sld::lib::mf::critical_section		_lock;
						HANDLE									_close_event;

					};
				};
			};
		};
	};
};

#endif