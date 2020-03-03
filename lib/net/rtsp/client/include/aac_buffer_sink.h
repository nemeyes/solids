#ifndef _AAC_BUFFER_SINK_H_
#define _AAC_BUFFER_SINK_H_

#include "buffer_sink.h"
#include "rtsp_client.h"

namespace sld
{
	namespace lib
	{
		namespace net
		{
			namespace rtsp
			{
				class client::aac_buffer_sink
					: public sld::lib::net::rtsp::client::buffer_sink
				{
				public:
					static sld::lib::net::rtsp::client::aac_buffer_sink * createNew(sld::lib::net::rtsp::client::core * front, UsageEnvironment & env, unsigned buffer_size, int32_t channels, int32_t samplerate, char * configstr, int32_t configstr_size);

				protected:
					aac_buffer_sink(sld::lib::net::rtsp::client::core * front, UsageEnvironment & env, unsigned buffer_size, int32_t channels, int32_t samplerate, char * configstr, int32_t configstr_size);
					virtual ~aac_buffer_sink(void);

					virtual void after_getting_frame(unsigned frame_size, unsigned truncated_bytes, struct timeval presentation_time, unsigned duration_msec);
				};
			};
		};
	};
};

#endif