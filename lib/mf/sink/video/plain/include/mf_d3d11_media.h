#ifndef _SLD_MF_D3D11_MEDIA_H_
#define _SLD_MF_D3D11_MEDIA_H_


#include <mf_base.h>
#include "mf_d3d11_stream.h"
#include "mf_d3d11_renderer.h"

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
                    namespace plain
                    {
                        class media
                            : solids::lib::mf::base
                            , solids::lib::mf::refcount_object
                            , public IMFMediaSink
                            , public IMFClockStateSink
                            , public IMFGetService
                            , public IMFRateSupport
                            , public IMFMediaSinkPreroll
                        {
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

                            // IMFRateSupport
                            STDMETHODIMP GetFastestRate(MFRATE_DIRECTION eDirection, BOOL fThin, _Out_ float* pflRate);
                            STDMETHODIMP GetSlowestRate(MFRATE_DIRECTION eDirection, BOOL fThin, _Out_ float* pflRate);
                            STDMETHODIMP IsRateSupported(BOOL fThin, float flRate, __RPC__inout_opt float* pflNearestSupportedRate);

                            // IMFMediaSinkPreroll
                            STDMETHODIMP NotifyPreroll(MFTIME hnsUpcomingStartTime);

                        private:
                            media(void);
                            virtual ~media(void);

                            HRESULT initialize(void);
                            HRESULT check_shutdown(void) const;

                        private:
                            static solids::lib::mf::critical_section        _lock_stream_and_scheduler;
                            const DWORD                                     _stream_id;
                            solids::lib::mf::critical_section               _lock;
                            BOOL                                            _is_shutdown;
                            solids::lib::mf::sink::video::plain::stream *   _stream;
                            IMFPresentationClock *                          _clock;
                            solids::lib::mf::scheduler *                    _scheduler;
                            solids::lib::mf::sink::video::plain::renderer * _renderer;
                        };
                    };
                };
            };
        };
    };
};










#endif