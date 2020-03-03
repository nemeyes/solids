#ifndef _SLD_MF_RTSP_SOURCE_SCHEME_HANDLER_H_
#define _SLD_MF_RTSP_SOURCE_SCHEME_HANDLER_H_

#include <mf_base.h>
#include <mf_refcount_object.h>

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
					class source;
					class handler
						: sld::lib::mf::base
						, sld::lib::mf::refcount_object
						, public IMFSchemeHandler
						, public IMFAsyncCallback
					{
					public:
						static HRESULT create_instance(IUnknown * unk, REFIID iid, void ** ppv);

						// IUnknown
						STDMETHODIMP QueryInterface(REFIID iid, void ** ppv);
						STDMETHODIMP_(ULONG) AddRef(void);
						STDMETHODIMP_(ULONG) Release(void);

						// IMFSchemeHandler
						STDMETHODIMP BeginCreateObject(LPCWSTR url, DWORD flags, IPropertyStore *, IUnknown ** unk_cancel_cookie, IMFAsyncCallback * callback, IUnknown * state);
						STDMETHODIMP CancelObjectCreation(IUnknown * unk_cancel_cookie);
						STDMETHODIMP EndCreateObject(IMFAsyncResult * result, MF_OBJECT_TYPE * ot, IUnknown ** ppv);

						// IMFAsyncCallback
						STDMETHODIMP GetParameters(DWORD*, DWORD*) { return E_NOTIMPL; }
						STDMETHODIMP Invoke(IMFAsyncResult * result);
					private:
						handler(void);
						virtual ~handler(void);

					private:
						IMFAsyncResult * _result;
						sld::lib::mf::source::rtsp::source * _source;
					};
				};
			};
		};
	};
};

#endif