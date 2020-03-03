#ifndef _FF_RTSP_CLIENT_H_
#define _FF_RTSP_CLIENT_H_

#include "sld_ff_rtsp_client.h"

extern "C" {
	#include <libavcodec/avcodec.h>
	#include <libavformat/avformat.h>
}

namespace solids
{
	namespace lib
	{
		namespace net
		{
			namespace rtsp
			{
				namespace ff
				{
					class client::core
					{
					public:
						typedef struct _stream_index_t
						{
							static const int32_t video = 0;
							static const int32_t audio = 1;
						} stream_index_t;

						core(solids::lib::net::rtsp::ff::client* front);
						~core(void);

						int32_t play(const char * url, int32_t transport, int32_t stimeout);
						int32_t stop(void);

					private:
						static unsigned __stdcall process_cb(void* param);
						void process(void);

					private:
						solids::lib::net::rtsp::ff::client* _front;

						AVFormatContext* _fmt_ctx;
						int32_t				_stream_index[2];

						char				_url[MAX_PATH];
						int32_t				_transport;
						int32_t				_session_timeout;
						HANDLE				_thread;
						BOOL				_run;
					};
				};
			};
		};
	};
};

#endif