#ifndef _SLD_TEMPLATE_QUEUE_H_
#define _SLD_TEMPLATE_QUEUE_H_

#include <cstdint>

namespace sld
{
	template<class T>
	class queue
	{
	public:
		queue(void)
			: _buffer(NULL)
			, _size(0)
			, _pending_count(0)
			, _available_index(0)
			, _pending_index(0)
		{}
		virtual ~queue(void)
		{}

		void initialize(T *pItems, uint32_t size)
		{
			_size = size;
			_pending_count = 0;
			_available_index = 0;
			_pending_index = 0;
			_buffer = new T *[_size];
			for (uint32_t i = 0; i < _size; i++)
			{
				_buffer[i] = &pItems[i];
			}
		}

		void release(void)
		{
			if (_buffer)
			{
				delete[] _buffer;
				_buffer = NULL;
			}
		}

		T * get_available(void)
		{
			T *pItem = NULL;
			if (_pending_count == _size)
			{
				return NULL;
			}
			pItem = _buffer[_available_index];
			_available_index = (_available_index + 1) % _size;
			_pending_count += 1;
			return pItem;
		}

		T * get_pending(void)
		{
			if (_pending_count == 0)
			{
				return NULL;
			}

			T *pItem = _buffer[_pending_index];
			_pending_index = (_pending_index + 1) % _size;
			_pending_count -= 1;
			return pItem;
		}

	protected:
		T**			_buffer;
		uint32_t	_size;
		uint32_t	_pending_count;
		uint32_t	_available_index;
		uint32_t	_pending_index;
	};
};

#endif
