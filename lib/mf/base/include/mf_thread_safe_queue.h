#ifndef _SLD_MF_THREAD_SAFE_QUEUE_H_
#define _SLD_MF_THREAD_SAFE_QUEUE_H_

namespace sld
{
	namespace lib
	{
		namespace mf
		{
			template <class T>
			class thread_safe_queue
			{
			public:
				thread_safe_queue(void)
				{
					::InitializeCriticalSection(&_lock);
				}

				virtual ~thread_safe_queue(void)
				{
					::DeleteCriticalSection(&_lock);
				}

				HRESULT queue(T* p)
				{
					EnterCriticalSection(&_lock);
					HRESULT hr = _list.insert_back(p);
					LeaveCriticalSection(&_lock);
					return hr;
				}

				HRESULT dequeue(T** pp)
				{
					EnterCriticalSection(&_lock);
					HRESULT hr = S_OK;
					if (_list.is_empty())
					{
						*pp = NULL;
						hr = S_FALSE;
					}
					else
					{
						hr = _list.remove_front(pp);
					}
					LeaveCriticalSection(&_lock);
					return hr;
				}

				HRESULT push_back(T* p)
				{
					EnterCriticalSection(&_lock);
					HRESULT hr = _list.insert_front(p);
					LeaveCriticalSection(&_lock);
					return hr;
				}

				DWORD get_count(void)
				{
					EnterCriticalSection(&_lock);
					DWORD nCount = _list.get_count();
					LeaveCriticalSection(&_lock);
					return nCount;
				}

				void clear(void)
				{
					EnterCriticalSection(&_lock);
					_list.clear();
					LeaveCriticalSection(&_lock);
				}

			private:

				CRITICAL_SECTION    _lock;
				sld::lib::mf::com_ptr_list2<T>     _list;
			};
		};
	};
};

#endif