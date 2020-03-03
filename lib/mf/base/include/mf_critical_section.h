#ifndef _SLD_MF_CRITICAL_SECTION_H_
#define _SLD_MF_CRITICAL_SECTION_H_

namespace sld
{
	namespace lib
	{
		namespace mf
		{
			class critical_section
			{
			private:
				CRITICAL_SECTION _cs;
			public:
				critical_section(void)
					: _cs()
				{
					::InitializeCriticalSection(&_cs);
				}

				~critical_section(void)
				{
					::DeleteCriticalSection(&_cs);
				}

				_Acquires_lock_(this->_cs)
				void lock(void)
				{
					::EnterCriticalSection(&_cs);
				}

				_Releases_lock_(this->_cs)
				void unlock(void)
				{
					::LeaveCriticalSection(&_cs);
				}
			};

			class auto_lock
			{
			private:
				critical_section* _cs;
			public:
				_Acquires_lock_(this->_cs->_cs)
				auto_lock(critical_section* cs)
					: _cs(cs)
				{
					_cs->lock();
				}

				_Releases_lock_(this->_cs->_cs)
				~auto_lock(void)
				{
					_cs->unlock();
				}
			};
		};
	};
};

#endif