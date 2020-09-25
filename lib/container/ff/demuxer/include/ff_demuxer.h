#pragma once

#include "sld_ff_demuxer.h"

extern "C" {
	#include <libavcodec/avcodec.h>
	#include <libavformat/avformat.h>
	#include <libavutil/mem.h>
	#include <libavutil/opt.h>
	#include <libavutil/imgutils.h>
	#include <libswscale/swscale.h>
	#include <libavutil/mathematics.h>
	#include <libavutil/time.h>
};


namespace solids
{
	namespace lib
	{
		namespace container
		{
			namespace ff
			{

				class demuxer::core
				{
				public:
					core(solids::lib::container::ff::demuxer* front);
					virtual ~core(void);

					BOOL	is_running(void);
					BOOL	is_paused(void);

					int32_t play(const char* container, int32_t type);
					int32_t resume(void);
					int32_t pause(void);
					int32_t stop(void);
					int32_t next(void);

				private:
					unsigned static __stdcall process_cb(void* param);
					void	process(void);

				private:
					solids::lib::container::ff::demuxer* _front;
					int32_t				_type;
					AVFormatContext*	_format_ctx;
					int32_t				_stream_index;
					AVPacket			_packet;
					AVPacket			_packet_filtered;
					AVBSFContext*		_bsfc;
					BOOL				_bMP4;
					int32_t				_nCnt;

					char				_container[MAX_PATH];
					CRITICAL_SECTION	_lock;
					BOOL				_run;
					BOOL				_pause;
					HANDLE				_thread;
				};

			};
		};
	};
};

