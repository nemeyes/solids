#ifndef _SLD_MF_REFCOUNT_OBJECT_H_
#define _SLD_MF_REFCOUNT_OBJECT_H_

namespace sld
{
	namespace lib
	{
		namespace mf
		{
			class refcount_object
			{
			public:
				refcount_object(void)
					: _ref_count(1)
				{}

				virtual ~refcount_object(void)
				{
					assert(_ref_count == 0);
				}

				ULONG AddRef(void)
				{
					return ::InterlockedIncrement(&_ref_count);
				}

				ULONG Release(void)
				{
					assert(_ref_count > 0);
					ULONG count = ::InterlockedDecrement(&_ref_count);
					if (count == 0)
						delete this;
					return count;
				}

			protected:
				long _ref_count;
			};
		};
	};
};

#endif