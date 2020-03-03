#ifndef _SLD_MF_ASYNC_CALLBACK_H_
#define _SLD_MF_ASYNC_CALLBACK_H_

namespace sld
{
	namespace lib
	{
		namespace mf
		{
			#define STATICASYNCCALLBACK(callback, parent) \
			class callback##_async_callback; \
			friend class callback##_async_callback; \
			class callback##_async_callback : public IMFAsyncCallback \
			{ \
			public: \
				STDMETHOD_(ULONG, AddRef)() \
				{ \
					parent * self = ((parent*)((BYTE*)this - offsetof(parent, mf_##callback))); \
					return self->AddRef(); \
				} \
				STDMETHOD_(ULONG, Release)() \
				{ \
					parent * self = ((parent*)((BYTE*)this - offsetof(parent, mf_##callback))); \
					return self->Release(); \
				} \
				STDMETHOD(QueryInterface)(REFIID riid, __RPC__deref_out _Result_nullonfailure_ void ** ppv) \
				{ \
					return E_NOINTERFACE; \
				} \
				STDMETHOD(GetParameters)(__RPC__out DWORD * flags, __RPC__out DWORD * queue) \
				{ \
					return S_OK; \
				} \
				STDMETHOD(Invoke)(__RPC__in_opt IMFAsyncResult * result) \
				{ \
					callback(result); \
					return S_OK; \
				} \
			} mf_##callback;

			#define METHODASYNCCALLBACKEX(callback, parent, flag, queue) \
			class callback##_async_callback; \
			friend class callback##_async_callback; \
			class callback##_async_callback : public IMFAsyncCallback \
			{ \
			public: \
				STDMETHOD_(ULONG, AddRef)() \
				{ \
					return get_parent()->AddRef(); \
				} \
				STDMETHOD_(ULONG, Release)() \
				{ \
					return get_parent()->Release(); \
				} \
				STDMETHOD(QueryInterface)(REFIID riid, __RPC__deref_out _Result_nullonfailure_ void ** ppv) \
				{ \
					if(riid == IID_IMFAsyncCallback || riid == IID_IUnknown) \
					{ \
						(*ppv) = this; \
						AddRef(); \
						return S_OK; \
					} \
					(*ppv) = NULL; \
					return E_NOINTERFACE; \
				} \
				STDMETHOD(GetParameters)(__RPC__out DWORD * pflags, __RPC__out DWORD * pqueue) \
				{ \
					*pflags = flag; \
					*pqueue = queue; \
					return S_OK; \
				} \
				STDMETHOD(Invoke)(__RPC__in_opt IMFAsyncResult * result) \
				{ \
					get_parent()->callback(result); \
					return S_OK; \
				} \
			protected: \
				parent * get_parent() \
				{ \
					return ((parent*)((BYTE*)this - offsetof(parent, mf_##callback))); \
				} \
			} mf_##callback;

			template<class T>
			class async_callback : public IMFAsyncCallback
			{
			public:

				typedef HRESULT(T::*invoke_fn)(IMFAsyncResult * ar);

				async_callback(T* parent, invoke_fn fn)
					: _parent(parent)
					, _invoke_fn(fn)
				{}
				// IUnknown
				STDMETHODIMP_(ULONG) AddRef(void)
				{
					// Delegate to parent class.
					return _parent->AddRef();
				}
				STDMETHODIMP_(ULONG) Release(void)
				{
					// Delegate to parent class.
					return _parent->Release();
				}
				STDMETHODIMP QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void** ppv)
				{
					if (!ppv)
						return E_POINTER;

					if (iid == __uuidof(IUnknown))
						*ppv = static_cast<IUnknown*>(static_cast<IMFAsyncCallback*>(this));
					else if (iid == __uuidof(IMFAsyncCallback))
						*ppv = static_cast<IMFAsyncCallback*>(this);
					else
					{
						*ppv = NULL;
						return E_NOINTERFACE;
					}
					AddRef();
					return S_OK;
				}

				// IMFAsyncCallback methods
				STDMETHODIMP GetParameters(__RPC__out DWORD* flags, __RPC__out DWORD* queue)
				{
					return E_NOTIMPL;
				}
				STDMETHODIMP Invoke(__RPC__in_opt IMFAsyncResult* ar)
				{
					return (_parent->*_invoke_fn)(ar);
				}

			private:
				T* _parent;
				invoke_fn	_invoke_fn;
			};
		};
	};
};


#endif