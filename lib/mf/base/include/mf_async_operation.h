#ifndef _SLD_MF_OPERATION_H_
#define _SLD_MF_OPERATION_H_

namespace solids
{
	namespace lib
	{
		namespace mf
		{
			namespace source
			{
				class async_operation
					: refcount_object,
					public IUnknown
				{
				public:

					typedef struct _command_t
					{
						static const int32_t start = 0;
						static const int32_t pause = 1;
						static const int32_t stop = 2;
						static const int32_t stream_end = 3;
						static const int32_t count = stream_end + 1;
					} command_t;

					async_operation(int32_t command)
					{
						_command = command;
						PropVariantInit(&_data);
					}

					virtual ~async_operation(void)
					{
						PropVariantClear(&_data);
					}

					HRESULT QueryInterface(REFIID iid, void** ppv)
					{
						if (ppv == NULL)
							return E_POINTER;
						if (iid == IID_IUnknown)
							*ppv = static_cast<IUnknown*>(this);
						else
						{
							*ppv = NULL;
							return E_NOINTERFACE;
						}
						AddRef();
						return S_OK;
					}

					ULONG AddRef(void)
					{
						return refcount_object::AddRef();
					}

					ULONG Release(void)
					{
						return refcount_object::Release();
					}

					int32_t get_command(void) const 
					{ 
						return _command; 
					}

					HRESULT set_data(const PROPVARIANT & var) 
					{ 
						return PropVariantCopy(&_data, &var); 
					};
					
					const PROPVARIANT & get_data(void) const 
					{ 
						return _data; 
					};

				private:
					int32_t _command;
					PROPVARIANT _data;
				};
			}
		}
	}
}

#endif