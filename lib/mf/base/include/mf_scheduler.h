#ifndef _SLD_MF_SCHEDULER_H_
#define _SLD_MF_SCHEDULER_H_

namespace sld
{
	namespace lib
	{
		namespace mf
		{
			typedef struct _scheduler_callback_t
			{
				virtual HRESULT process(IMFSample* sample) = 0;
			} scheduler_callback_t;

			class scheduler
				: base
				, refcount_object
				, public IUnknown
			{
			public:
				scheduler(critical_section * lock)
					: _lock(lock)
					, _cb(NULL)
					, _queue_id(0)
					, _scheduled_samples()
					, _clock(NULL)
					, _rate(1.0f)
					, _wait_timer(NULL)
					, _last_sample_time(0)
					, _duration(0)
					, _key_timer(0)
					, _prev_delta(0)
					, _delta_count(0)
					, _presentation_offset(0)
					, _run(false)
				{
					::ZeroMemory(_uuid, sizeof(_uuid));
					::MFAllocateSerialWorkQueue(MFASYNC_CALLBACK_QUEUE_STANDARD, &_queue_id);
					//::MFAllocateWorkQueue(&_queue_id);
				}

				virtual ~scheduler(void)
				{
					{
						//sld::lib::mf::auto_lock lock(&_lock_wait_timer);
						if (_wait_timer != NULL)
						{
							::CloseHandle(_wait_timer);
							_wait_timer = NULL;
						}
					}

					_scheduled_samples.clear();
					safe_release(_clock);

					::MFUnlockWorkQueue(_queue_id);
				}

				STDMETHODIMP_(ULONG) AddRef(void)
				{
					return refcount_object::AddRef();
				}

				STDMETHODIMP_(ULONG) Release(void)
				{
					return refcount_object::Release();
				}

				STDMETHODIMP QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void ** ppv)
				{
					if (!ppv)
						return E_POINTER;

					if (iid == IID_IUnknown)
					{
						*ppv = static_cast<IUnknown*>(this);
					}
					else
					{
						*ppv = NULL;
						return E_NOINTERFACE;
					}
					AddRef();
					return S_OK;
				}

				void set_callback(scheduler_callback_t * cb)
				{
					_cb = cb;
				}

				void set_clock_rate(float rate)
				{
					_rate = rate;
				}

				void set_duration(LONGLONG duration)
				{
					_duration = duration;
				}

				const LONGLONG& last_sample_time(void) const
				{
					return _last_sample_time;
				}

				const LONGLONG& duration(void) const
				{
					return _duration;
				}

				DWORD get_count(void)
				{
					return _scheduled_samples.get_count();
				}

				HRESULT set_uuid(LPCTSTR uuid)
				{
					wcsncpy_s(_uuid, uuid, wcslen(uuid));
					return S_OK;
				}

				HRESULT start(IMFClock * clock)
				{
					HRESULT hr = S_OK;

					//sld::lib::mf::auto_lock lock(&_lock);

					_prev_delta = 0;
					_delta_count = 0;

					_scheduled_samples.clear();

					safe_release(_clock);
					_clock = clock;
					if (_clock != NULL)
						_clock->AddRef();

					::timeBeginPeriod(1);

					{
						//sld::lib::mf::auto_lock lock(&_lock_wait_timer);
						_wait_timer = ::CreateWaitableTimer(NULL, FALSE, NULL);
						if (_wait_timer == NULL)
							hr = HRESULT_FROM_WIN32(GetLastError());
					}

					_run = true;
					return hr;
				}

				HRESULT stop(void)
				{
					//sld::lib::mf::auto_lock lock(&_lock);
					_run = false;
					_scheduled_samples.clear();

					{
						//sld::lib::mf::auto_lock lock(&_lock_wait_timer);
						if (_wait_timer != NULL)
						{
							::CloseHandle(_wait_timer);
							_wait_timer = NULL;
						}
					}


					::timeEndPeriod(1);

					safe_release(_clock);
					return S_OK;
				}

				HRESULT flush(void)
				{
					//sld::lib::mf::auto_lock lock(&_lock);
					//_scheduled_samples.clear();

					if (_key_timer != NULL)
					{
						(void)MFCancelWorkItem(_key_timer);
						_key_timer = NULL;
					}

					_scheduled_samples.clear();
					return S_OK;
				}

				HRESULT schedule_sample(IMFSample * sample, BOOL process_now)
				{
					if (!_run)
						return MF_E_NOT_INITIALIZED;
					if (_cb == NULL)
						return MF_E_NOT_INITIALIZED;

					HRESULT hr = S_OK;
					if (process_now || (_clock == NULL))
						hr = _cb->process(sample);	//Process Sample Immediately
					else
					{
						hr = _scheduled_samples.queue(sample);
						if (SUCCEEDED(hr))
						{
							hr = MFPutWorkItem(_queue_id, &mf_timer_callback, nullptr);
						}
					}
					return hr;
				}

				HRESULT process_sample_in_queue(LONG * next_sleep)
				{
					HRESULT hr = S_OK;
					LONG wait = 0;
					IMFSample* sample = NULL;

					while (_scheduled_samples.dequeue(&sample) == S_OK)
					{
						hr = process_sample(sample, &wait);
						safe_release(sample);
						if (FAILED(hr) || wait > 0)
							break;
					}

					if (wait == 0)
						wait = INFINITE;
					*next_sleep = wait;
					return hr;
				}

				HRESULT process_sample(IMFSample * sample, LONG * next_sleep)
				{
					HRESULT hr = S_OK;
					LONGLONG pts = 0;
					LONGLONG now = 0;
					MFTIME stime = 0;
					LONGLONG delta = 0;

					BOOL process_now = TRUE;
					BOOL discard = FALSE;
					LONG nsleep = 0;
					if (_clock)
					{
						hr = sample->GetSampleTime(&pts);
						if (SUCCEEDED(hr))
							hr = _clock->GetCorrelatedTime(0, &now, &stime);
						if (_presentation_offset == 0)
						{
							_presentation_offset = now - pts;
						}
						if (SUCCEEDED(hr))
						{
							//cap_log4cplus_logger::make_debug_log("amadeus.player", "*$*$Audio*$*$ sample %dms now %dms ", (int)pts / 10000, (int)now / 10000);

							delta = pts - now + _presentation_offset;

							if (delta == _prev_delta)
								_delta_count++;

							if (_delta_count == 3)
							{
								discard = TRUE;
								_delta_count = 0;
							}

							//if (delta > 0)
							//{
							//	discard = TRUE;
							//	_delta_count = 0;
							//}

							_prev_delta = delta;

							if (!discard)
							{
								if (_rate < 0)
								{
									delta = -delta;
								}

								//if (1)
								if (delta < -(_duration))
									process_now = TRUE;
								else if (delta > (3 * _duration))
								{

									nsleep = static_cast<LONG>(delta - (3 * _duration));
									//nsleep = static_cast<LONG>(ticks2msecs(delta - (3 * _duration)));
									nsleep = static_cast<LONG>(nsleep / fabsf(_rate));
									process_now = FALSE;
								}
							}
						}
					}
					if (process_now && !discard)
						hr = _cb->process(sample);
					else if (!discard)
						hr = _scheduled_samples.push_back(sample);
					*next_sleep = nsleep;
					return hr;
				}

			private:
				HRESULT start_process_sample(void)
				{
					HRESULT hr = S_OK;
					LONG wait = INFINITE;
					BOOL exit_thread = FALSE;
					IMFAsyncResult* async_result = NULL;

					hr = process_sample_in_queue(&wait);
					if (SUCCEEDED(hr) && _wait_timer)
					{
						//sld::lib::mf::auto_lock lock(&_lock_wait_timer);
						if (!_wait_timer)
							return hr;
						if (wait != INFINITE && wait > 0)
						{
							//not time to process yet, wait until the right time
							LARGE_INTEGER due_time;
							due_time.QuadPart = -1 * wait;
							//due_time.QuadPart = -1 * msecs2ticks(wait);
							if (::SetWaitableTimer(_wait_timer, &due_time, 0, NULL, NULL, FALSE) == 0)
								hr = HRESULT_FROM_WIN32(::GetLastError());
							if (SUCCEEDED(hr))
							{
								hr = MFCreateAsyncResult(nullptr, &mf_timer_callback, nullptr, &async_result);
								if (SUCCEEDED(hr))
								{
									hr = MFPutWaitingWorkItem(_wait_timer, 0, async_result, &_key_timer);
								}
							}
						}
					}

					safe_release(async_result);
					return hr;
				}

				HRESULT timer_callback(__RPC__in_opt IMFAsyncResult * result)
				{
					HRESULT hr = S_OK;
#if 0
					//hr = result->GetStatus();
					//if(SUCCEEDED(hr))
					{
						sld::lib::mf::auto_lock lock(&_lock);
						if (_key_timer)
						{
							hr = MFCancelWorkItem(_key_timer);
							_key_timer = NULL;
						}
					}
					hr = start_process_sample();
					return hr;
#else
					sld::lib::mf::auto_lock lock(_lock);
					if (_key_timer)
					{
						hr = MFCancelWorkItem(_key_timer);
						_key_timer = NULL;
					}
					hr = start_process_sample();
					return hr;
#endif
				}

				METHODASYNCCALLBACKEX(timer_callback, scheduler, 0, MFASYNC_CALLBACK_QUEUE_STANDARD);

			private:
				sld::lib::mf::critical_section *		_lock;
				sld::lib::mf::critical_section		_lock_wait_timer;
				DWORD									_queue_id;
				sld::lib::mf::scheduler_callback_t *	_cb;
				sld::lib::mf::thread_safe_queue<IMFSample>			_scheduled_samples;
				IMFClock* _clock;
				float							_rate;
				HANDLE							_wait_timer;
				MFTIME							_last_sample_time;
				MFTIME							_duration;
				MFWORKITEM_KEY					_key_timer;

				LONGLONG						_prev_delta;
				int32_t							_delta_count;

				wchar_t							_uuid[MAX_PATH];
				//최초 샘플에 대해서는 바로 그릴 수 있도록 한다.
				//MF 내부의 타이머와 수신 샘플의 타이머를 동기화시킨다.
				//CASP Source를 사용해 수신/렌더링하는 경우를 고려하여 새로 생성된 변수
				LONGLONG						_presentation_offset;
				BOOL							_run;
			};
		};
	};
};

#endif