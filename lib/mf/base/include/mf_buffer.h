#ifndef _SLD_MF_BUFFER_H_
#define _SLD_MF_BUFFER_H_

namespace sld
{
	namespace lib
	{
		namespace mf
		{
			class buffer
			{
			public:
				static const int32_t GROW_BUFFER_SIZE = 4096;
				
				buffer(void)
					: _buffer(NULL)
					, _total_size(0)
					, _start_after_resize(0)
					, _start_position(0)
					, _end_position(0)
				{
				}

				virtual ~buffer(void)
				{
					if (_buffer)
						delete[] _buffer;

					_buffer = NULL;
				}

				HRESULT initialize(void)
				{
					// Initialize only once
					if (_buffer)
						return S_OK;

					_buffer = new (std::nothrow)BYTE[GROW_BUFFER_SIZE];

					if (_buffer == NULL)
						return E_OUTOFMEMORY;

					_total_size = GROW_BUFFER_SIZE;

					return S_OK;
				}

				HRESULT initialize(const DWORD size)
				{
					// Initialize only once
					if (_buffer)
						return S_OK;

					_buffer = new (std::nothrow)BYTE[size];

					if (_buffer == NULL)
						return E_OUTOFMEMORY;

					_total_size = size;

					return S_OK;
				}

				HRESULT reserve(const DWORD size)
				{
					// Initialize must be called first
					assert(_buffer);

					if (size == 0)
						return S_OK;

					if (size > MAXDWORD - _total_size)
						return E_UNEXPECTED;

					return set_size(size);
				}

				HRESULT set_start_position(const DWORD position)
				{
					if (position == 0)
						return S_OK;

					if (position > MAXDWORD - _start_position)
						return E_UNEXPECTED;

					DWORD new_position = _start_position + position;

					if (new_position > _end_position)
						return E_UNEXPECTED;

					_start_position = new_position;

					return S_OK;
				}

				HRESULT set_end_position(const DWORD position)
				{
					if (position == 0)
						return S_OK;

					if (position > MAXDWORD - _end_position)
						return E_UNEXPECTED;

					DWORD new_position = _end_position + position;

					if (new_position > _total_size)
						return E_UNEXPECTED;

					_end_position = new_position;
					_start_after_resize = 0;

					return S_OK;
				}

				void reset(void)
				{
					_start_position = 0;
					_end_position = 0;
					_start_after_resize = 0;
				}

				BYTE* get_read_start_buffer(void)
				{
					return _buffer + (_start_position + _start_after_resize);
				}

				BYTE* get_start_buffer(void)
				{
					return _buffer + _start_position;
				}

				DWORD get_buffer_size(void) const
				{
					assert(_end_position >= _start_position);
					return (_end_position - _start_position);
				}

				DWORD get_allocated_size(void) const
				{
					return _total_size;
				}

			private:
				HRESULT set_size(const DWORD size)
				{
					HRESULT hr = S_OK;

					DWORD current_size = get_buffer_size();

					// Todo check
					//if(dwCurrentSize == dwSize)
					//return hr;

					DWORD remaining_size = _total_size - current_size;

					if (size > remaining_size)
					{

						// Grow enough to not go here too many times (avoid lot of new).
						// We could use a multiple of 16 and use an aligned buffer.
						DWORD new_total_size = size + current_size + GROW_BUFFER_SIZE;

						BYTE* tmp = new (std::nothrow)BYTE[new_total_size];
						if (tmp != NULL)
						{
							if (_buffer != NULL)
							{
								memcpy(tmp, get_start_buffer(), current_size);
								delete[] _buffer;
							}

							_buffer = tmp;

							_start_after_resize = current_size;
							_start_position = 0;
							_end_position = current_size;
							_total_size = new_total_size;
						}
						else
						{
							hr = E_OUTOFMEMORY;
						}
					}
					else
					{
						if (current_size != 0)
							memcpy(_buffer, get_start_buffer(), current_size);

						_start_after_resize = current_size;
						_start_position = 0;
						_end_position = current_size;
					}

					return hr;
				}

			private:
				BYTE *	_buffer;
				DWORD	_total_size;
				DWORD	_start_after_resize;

				DWORD	_start_position;
				DWORD	_end_position;
			};
		};
	};
};

#endif