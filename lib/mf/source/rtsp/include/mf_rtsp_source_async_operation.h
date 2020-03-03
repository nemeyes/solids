#ifndef _SLD_MF_RTSP_SOURCE_ASYNC_OPERATION_H_
#define _SLD_MF_RTSP_SOURCE_ASYNC_OPERATION_H_

#include <mf_base.h>

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
					class async_operation
						: solids::lib::mf::refcount_object
						, public IUnknown
					{
					public:
						typedef struct _type_t
						{
							static const int32_t start = 0;
							static const int32_t pause = 1;
							static const int32_t stop = 2;
							static const int32_t count = stop + 1;
						} type_t;

						async_operation(int32_t op);

						// IUnknown methods.
						STDMETHODIMP QueryInterface(REFIID iid, void** ppv);
						STDMETHODIMP_(ULONG) AddRef();
						STDMETHODIMP_(ULONG) Release();

						int32_t op(void);
						HRESULT data(const PROPVARIANT & var);
						const PROPVARIANT & data(void);

					protected:
						virtual ~async_operation(void);

					protected:
						int32_t	_op;
						PROPVARIANT _data;
					};

					class start_async_operation : public async_operation
					{
					public:
						start_async_operation(IMFPresentationDescriptor * pd);
						~start_async_operation(void);

						HRESULT presentation_descriptor(IMFPresentationDescriptor ** pd);

					protected:
						IMFPresentationDescriptor * _pd;
					};
				};
			};
		};
	};
};

#endif