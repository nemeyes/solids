#ifndef _H264_BUFFER_SINK_H_
#define _H264_BUFFER_SINK_H_

#include "h2645_buffer_sink.h"
#include "rtsp_client.h"

namespace solids
{
	namespace lib
	{
		namespace net
		{
			namespace rtsp
			{
				class client::h264_buffer_sink
					: public solids::lib::net::rtsp::client::h2645_buffer_sink
				{
				public:
					static solids::lib::net::rtsp::client::h264_buffer_sink * createNew(solids::lib::net::rtsp::client::core * front, UsageEnvironment & env, const char * sps, unsigned sps_size, const char * pps, unsigned pps_size, unsigned buffer_size = 100000);

				protected:
					h264_buffer_sink(solids::lib::net::rtsp::client::core * front, UsageEnvironment & env, const char * sps, unsigned sps_size, const char * pps, unsigned pps_size, unsigned buffer_size);
					virtual ~h264_buffer_sink(void);
				};
			};
		};
	};
};


#endif // H264_BUFFER_SINK_H

