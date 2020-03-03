#ifndef _SLD_MF_GROWABLE_ARRAY_H_
#define _SLD_MF_GROWABLE_ARRAY_H_

namespace solids
{
	namespace lib
	{
		namespace mf
		{
			template <class T>
			class growable_array
			{
			public:
				growable_array(void)
					: _count(0)
					, _allocated(0)
					, _array(NULL)
				{}

				virtual ~growable_array(void)
				{
					safe_delete_array(_array);
				}

				// Allocate: Reserves memory for the array, but does not increase the count.
				HRESULT allocate(DWORD alloc)
				{
					HRESULT hr = S_OK;
					if (alloc > _allocated)
					{
						T * tmp = new T[alloc];
						if (tmp)
						{
							ZeroMemory(tmp, alloc * sizeof(T));
							assert(_count <= _allocated);

							// Copy the elements to the re-allocated array.
							for (DWORD i = 0; i < _count; i++)
							{
								tmp[i] = _array[i];
							}

							delete[] _array;

							_array = tmp;
							_allocated = alloc;
						}
						else
						{
							hr = E_OUTOFMEMORY;
						}
					}
					return hr;
				}

				// SetSize: Changes the count, and grows the array if needed.
				HRESULT size(DWORD count)
				{
					assert(_count <= _allocated);

					HRESULT hr = S_OK;
					if (count > _allocated)
					{
						hr = allocate(count);
					}
					if (SUCCEEDED(hr))
					{
						_count = count;
					}
					return hr;
				}

				DWORD count(void) const { return _count; }

				// Accessor.
				T& operator[](DWORD index)
				{
					assert(index < _count);
					return _array[index];
				}

				// Const accessor.
				const T& operator[](DWORD index) const
				{
					assert(index < _count);
					return _array[index];
				}

				// Return the underlying array.
				T* Ptr() { return _array; }

			protected:
				growable_array& operator=(const growable_array& r);
				growable_array(const growable_array &r);

				T       *_array;
				DWORD   _count;        // Nominal count.
				DWORD   _allocated;    // Actual allocation size.
			};
		};
	};
};

#endif