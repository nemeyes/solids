#ifndef _SLD_MF_RTSP_SOURCE_CLASS_FACTORY_H_
#define _SLD_MF_RTSP_SOURCE_CLASS_FACTORY_H_

#include <mf_base.h>
#include <mf_refcount_object.h>

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
					class factory
						: solids::lib::mf::base
						, public IClassFactory
					{
					public:
						static BOOL is_locked(void);

						factory(void);
						~factory(void);

						// IUnknown
						STDMETHODIMP_(ULONG) AddRef(void);
						STDMETHODIMP QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void ** ppv);
						STDMETHODIMP_(ULONG) Release(void);

						// IClassFactory
						STDMETHODIMP CreateInstance(_In_opt_ IUnknown * unk, _In_ REFIID riid, _COM_Outptr_ void ** ppv);
						STDMETHODIMP LockServer(BOOL lock);

					private:
						static volatile long _lock_count;
						volatile long _ref_count;
					};
				};
			};
		};
	};
};

#endif