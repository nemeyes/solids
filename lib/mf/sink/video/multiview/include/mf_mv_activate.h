#ifndef _SLD_MF_MV_ACTIVATE_H_
#define _SLD_MF_MV_ACTIVATE_H_

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
						class activate
							: solids::lib::mf::base
							, solids::lib::mf::refcount_object
							, public attributes<IMFActivate>
							, public IPersistStream
						{
						public:
							static HRESULT create_instance(HWND hwnd, IMFActivate ** activate);

							// IUnknown
							STDMETHODIMP_(ULONG) AddRef(void);
							STDMETHODIMP QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void** ppv);
							STDMETHODIMP_(ULONG) Release(void);

							// IMFActivate
							STDMETHODIMP ActivateObject(__RPC__in REFIID iid, __RPC__deref_out_opt void** ppv);
							STDMETHODIMP DetachObject(void);
							STDMETHODIMP ShutdownObject(void);

							// IPersistStream
							STDMETHODIMP GetSizeMax(__RPC__out ULARGE_INTEGER* size);
							STDMETHODIMP IsDirty(void);
							STDMETHODIMP Load(__RPC__in_opt IStream* stream);
							STDMETHODIMP Save(__RPC__in_opt IStream* stream, BOOL cleardirty);

							// IPersist (from IPersistStream)
							STDMETHODIMP GetClassID(__RPC__out CLSID* clsid);
							
						private:
							activate(void);
							~activate(void);

						private:
							IMFMediaSink *	_media_sink;
							HWND			_hwnd;
						};
					};
				};
			};
		};
	};
};

#endif