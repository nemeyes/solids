#ifndef _SLD_MF_MV_MEDIA_SINK_H_
#define _SLD_MF_MV_MEDIA_SINK_H_

#include <mf_base.h>

namespace solids
{
	namespace lib
	{
		namespace mf
		{
			namespace sink
			{
				namespace video
				{
					namespace multiview
					{
						class stream;
						class renderer;
						class media
							: solids::lib::mf::base
							, solids::lib::mf::refcount_object
							, public IMFMediaSink
							, public IMFClockStateSink
							, public IMFGetService
							, public IMFMediaSinkPreroll
							, public IPresenter
						{
							static const int32_t sink_stream_id;
						public:
							// Static method to create the object.
							static HRESULT create_instance(_In_ REFIID iid, _COM_Outptr_ void** ppSink);

							// IUnknown
							STDMETHODIMP_(ULONG) AddRef(void);
							STDMETHODIMP QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void** ppv);
							STDMETHODIMP_(ULONG) Release(void);

							// IMFMediaSink methods
							STDMETHODIMP AddStreamSink(DWORD dwStreamSinkIdentifier, __RPC__in_opt IMFMediaType* pMediaType, __RPC__deref_out_opt IMFStreamSink** ppStreamSink);
							STDMETHODIMP GetCharacteristics(__RPC__out DWORD* pdwCharacteristics);
							STDMETHODIMP GetPresentationClock(__RPC__deref_out_opt IMFPresentationClock** ppPresentationClock);
							STDMETHODIMP GetStreamSinkById(DWORD dwIdentifier, __RPC__deref_out_opt IMFStreamSink** ppStreamSink);
							STDMETHODIMP GetStreamSinkByIndex(DWORD dwIndex, __RPC__deref_out_opt IMFStreamSink** ppStreamSink);
							STDMETHODIMP GetStreamSinkCount(__RPC__out DWORD* pcStreamSinkCount);
							STDMETHODIMP RemoveStreamSink(DWORD dwStreamSinkIdentifier);
							STDMETHODIMP SetPresentationClock(__RPC__in_opt IMFPresentationClock* pPresentationClock);
							STDMETHODIMP Shutdown(void);

							// IMFClockStateSink methods
							STDMETHODIMP OnClockPause(MFTIME hnsSystemTime);
							STDMETHODIMP OnClockRestart(MFTIME hnsSystemTime);
							STDMETHODIMP OnClockSetRate(MFTIME hnsSystemTime, float flRate);
							STDMETHODIMP OnClockStart(MFTIME hnsSystemTime, LONGLONG llClockStartOffset);
							STDMETHODIMP OnClockStop(MFTIME hnsSystemTime);

							// IMFGetService
							STDMETHODIMP GetService(__RPC__in REFGUID guidService, __RPC__in REFIID riid, __RPC__deref_out_opt LPVOID* ppvObject);

							// IMFMediaSinkPreroll
							STDMETHODIMP NotifyPreroll(MFTIME hnsUpcomingStartTime);

							//IPresenter		
							STDMETHODIMP SetViewCount(INT count);
							STDMETHODIMP EnableCoordinatedCSConverter(BOOL enable);
							STDMETHODIMP SetViewPosition(INT index, FLOAT* position);
							STDMETHODIMP SetSelected(INT index);
							STDMETHODIMP Maximize(void);
							STDMETHODIMP ChangeRenderType(void);

							HRESULT QueueEvent(MediaEventType met, __RPC__in REFGUID guidExtendedType, HRESULT hrStatus, __RPC__in_opt const PROPVARIANT* pvValue);
						private:

							media(void);
							virtual ~media(void);

							HRESULT check_shutdown(void) const;
							HRESULT initialize(void);

							const DWORD				_stream_id;
							static critical_section _lock_streamsink_and_scheduler;
							critical_section		_lock;
							BOOL					_is_shutdown;

							
							solids::lib::mf::sink::video::multiview::stream *	_stream;
							IMFPresentationClock *								_clock;
							solids::lib::mf::scheduler *						_scheduler;
							solids::lib::mf::sink::video::multiview::renderer * _renderer;
							TCHAR												_uuid[MAX_PATH];
						};
					};
				};
			};
		};
	};
};

namespace amadeus
{
	namespace mf
	{
		class scheduler;
		namespace sink
		{
			namespace video
			{

			}
		}
	}
}

#endif