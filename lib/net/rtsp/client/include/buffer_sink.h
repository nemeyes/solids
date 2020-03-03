#ifndef _BUFFER_SINK_H_
#define _BUFFER_SINK_H_

#include <MediaSink.hh>
#include <UsageEnvironment.hh>
#include "rtsp_client.h"

namespace sld
{
	namespace lib
	{
		namespace net
		{
			namespace rtsp
			{
				class client::buffer_sink : public MediaSink
				{
				public:
					static sld::lib::net::rtsp::client::buffer_sink * createNew(sld::lib::net::rtsp::client::core * front, int32_t mt, int32_t codec, UsageEnvironment & env, unsigned buffer_size);

					virtual void add_data(unsigned char * data, unsigned size, struct timeval presentation_time, unsigned duration_msec);

				protected:
					buffer_sink(sld::lib::net::rtsp::client::core * front, int32_t mt, int32_t codec, UsageEnvironment & env, unsigned buffer_size);
					virtual ~buffer_sink(void);

				protected: //redefined virtual functions
					virtual Boolean continuePlaying(void);

				protected:
					static void after_getting_frame(void * param, unsigned frame_size, unsigned truncated_bytes, struct timeval presentation_time, unsigned duration_msec);
					virtual void after_getting_frame(unsigned frame_size, unsigned truncated_bytes, struct timeval presentation_time, unsigned duration_msec);


					sld::lib::net::rtsp::client::core * _front;
					unsigned char *		_buffer;
					unsigned			_buffer_size;
					int32_t				_mt;
					int32_t				_vcodec;
					int32_t				_acodec;
				};
			};
		};
	};
};

#endif // BUFFER_SINK_H

