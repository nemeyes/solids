#ifndef _SLD_MF_PLAIN_CONTROLLER_H_
#define _SLD_MF_PLAIN_CONTROLLER_H_

#if defined(EXPORT_SLD_MF_CONTROLLER_LIB)
#  define EXP_SLD_MF_CONTROLLER_CLASS __declspec(dllexport)
#else
#  define EXP_SLD_MF_CONTROLLER_CLASS __declspec(dllimport)
#endif

#include <sld.h>

namespace solids
{
	namespace lib
	{
		namespace mf
		{
			namespace control
			{
				namespace plain
				{
					class EXP_SLD_MF_CONTROLLER_CLASS controller
						: public solids::lib::base
					{
					public:
						class core;
					public:
						typedef struct _state_t
						{
							static const int32_t unknown = 0;
							static const int32_t closed = 1;
							static const int32_t ready = 2;
							static const int32_t open_pending = 3;
							static const int32_t started = 4;
							static const int32_t paused = 5;
							static const int32_t stopped = 6;
							static const int32_t closing = 7;
						} state_t;

						typedef struct _context_t
						{
							wchar_t url[MAX_PATH];
							HWND	hwnd;
							void *	userdata;
							BOOL	repeat;
							_context_t(void)
								: userdata(NULL)
								, repeat(FALSE)
							{

							}
						} context_t;

						controller(void);
						virtual ~controller(void);

						// Playback control
						int32_t open(solids::lib::mf::control::plain::controller::context_t * context);
						int32_t close(void);
						int32_t play(void);
						int32_t pause(void);
						int32_t stop(void);
						int32_t state(void) const;


					private:
						solids::lib::mf::control::plain::controller::core * _core;
					};
				};
			};
		};
	};
};

#endif