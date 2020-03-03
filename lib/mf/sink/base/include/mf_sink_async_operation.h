#ifndef _SLD_MF_SINK_ASYNC_OPERATION_H_
#define _SLD_MF_SINK_ASYNC_OPERATION_H_

#include <mf_base.h>

namespace solids
{
	namespace lib
	{
		namespace mf
		{
			namespace sink
			{
				class async_operation
					: solids::lib::mf::refcount_object
					, public IUnknown
				{
				public:
					typedef struct _type_t
					{
						static const int32_t set_media_type = 0;
						static const int32_t start = 1;
						static const int32_t restart = 2;
						static const int32_t pause = 3;
						static const int32_t stop = 4;
						static const int32_t process_sample = 5;
						static const int32_t place_marker = 6;
						static const int32_t count = 7;

					} type_t;

					async_operation(int32_t op);

					// IUnknown methods.
					STDMETHODIMP QueryInterface(REFIID iid, void** ppv);
					STDMETHODIMP_(ULONG) AddRef();
					STDMETHODIMP_(ULONG) Release();

					int32_t op(void);

				private:
					virtual ~async_operation(void);

				private:
					int32_t	_op;
				};
			};
		};
	};
};

#endif