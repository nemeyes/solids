#ifndef _SLD_AUTOLOCK_H_
#define _SLD_AUTOLOCK_H_

#include <mutex>

namespace solids
{
	namespace lib
	{
		class autolock
		{
		public:
			autolock(CRITICAL_SECTION* lock)
				: _lock(lock)
			{
				::EnterCriticalSection(_lock);
			}

			~autolock(void)
			{
				::LeaveCriticalSection(_lock);
			}
		private:
			CRITICAL_SECTION* _lock;
		};

		class autolock2
		{
		public:
			autolock2(std::mutex* lock)
				: _lock(lock)
			{
				_lock->lock();
			}

			~autolock2(void)
			{
				_lock->unlock();
			}
		private:
			std::mutex* _lock;
		};

		class scopedlock
		{
		public:
			scopedlock(HANDLE lock)
				: _lock(lock)
			{
				if (_lock == NULL || _lock == INVALID_HANDLE_VALUE)
					return;
				::WaitForSingleObject(_lock, INFINITE);
			}

			~scopedlock(void)
			{
				if (_lock == NULL || _lock == INVALID_HANDLE_VALUE)
					return;

				::SetEvent(_lock);
			}

		private:
			HANDLE _lock;

		private:
			scopedlock(void);
			scopedlock(const scopedlock& clone);

		};

		class exclusive_scopedlock
		{
		public:
			exclusive_scopedlock(SRWLOCK* lock)
				: _lock(lock)
			{
				::AcquireSRWLockExclusive(_lock);
			}

			~exclusive_scopedlock(void)
			{
				ReleaseSRWLockExclusive(_lock);
			}

		private:
			SRWLOCK* _lock;

		private:
			exclusive_scopedlock(void);
			exclusive_scopedlock(const exclusive_scopedlock& clone);
		};

		class shared_scopedlock
		{
		public:
			shared_scopedlock(SRWLOCK* lock)
				: _lock(lock)
			{
				::AcquireSRWLockShared(_lock);
			}

			~shared_scopedlock(void)
			{
				::ReleaseSRWLockShared(_lock);
			}

		private:
			SRWLOCK* _lock;

		private:
			shared_scopedlock(void);
			shared_scopedlock(const shared_scopedlock& clone);
		};
	};
};




#endif
