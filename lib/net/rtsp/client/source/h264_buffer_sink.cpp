
#include "h264_buffer_sink.h"

sld::lib::net::rtsp::client::h264_buffer_sink::h264_buffer_sink(sld::lib::net::rtsp::client::core * front, UsageEnvironment & env, const char * sps, unsigned sps_size, const char * pps, unsigned pps_size, unsigned buffer_size)
	: sld::lib::net::rtsp::client::h2645_buffer_sink(front, sld::lib::net::rtsp::client::video_codec_t::avc, env, nullptr, 0, sps, sps_size, pps, pps_size, buffer_size)
{

}

sld::lib::net::rtsp::client::h264_buffer_sink::~h264_buffer_sink(void)
{

}

sld::lib::net::rtsp::client::h264_buffer_sink* sld::lib::net::rtsp::client::h264_buffer_sink::createNew(sld::lib::net::rtsp::client::core * front, UsageEnvironment & env, const char * sps, unsigned sps_size, const char * pps, unsigned pps_size, unsigned buffer_size)
{
	return new h264_buffer_sink(front, env, sps, sps_size, pps, pps_size, buffer_size);
}
