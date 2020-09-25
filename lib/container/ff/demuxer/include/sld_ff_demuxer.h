#pragma once

#if defined(EXP_SLD_FF_DEMUXER_LIB)
#define EXP_SLD_FF_DEMUXER_CLS __declspec(dllexport)
#else
#define EXP_SLD_FF_DEMUXER_CLS __declspec(dllimport)
#endif

#include <sld.h>

namespace solids
{
	namespace lib
	{
		namespace container
		{
			namespace ff
			{
				class EXP_SLD_FF_DEMUXER_CLS demuxer
					: public solids::lib::base
				{
					class core;
				public:
					typedef struct _type_t
					{
						static const int32_t normal = 0;
						static const int32_t step = 1;
					} type_t;

					demuxer(void);
					virtual ~demuxer(void);

					BOOL	is_running(void);
					BOOL	is_paused(void);

					int32_t play(const char* container, int32_t type = 0);
					int32_t resume(void);
					int32_t pause(void);
					int32_t stop(void);
					int32_t next(void);

					virtual void on_video_begin(int32_t codec, const uint8_t* extradata, int32_t extradataSize, int32_t width, int32_t height, double fps) = 0;
					virtual void on_video_recv(uint8_t* bytes, int32_t nbytes, int32_t nFrameIdx) = 0;
					virtual void on_video_end(void) = 0;

				private:
					solids::lib::container::ff::demuxer::core* _core;

				};
			};
		};
	};
};


