
#include "aac_buffer_sink.h"
#include <H264VideoRTPSource.hh>

solids::lib::net::rtsp::client::aac_buffer_sink::aac_buffer_sink(solids::lib::net::rtsp::client::core * front, UsageEnvironment & env, unsigned buffer_size, int32_t channels, int32_t samplerate, char * configstr, int32_t configstr_size)
	: solids::lib::net::rtsp::client::buffer_sink(front, solids::lib::net::rtsp::client::media_type_t::audio, solids::lib::net::rtsp::client::audio_codec_t::aac, env, buffer_size)
{
	if (_front)
	{
		_front->set_audio_channels(channels);
		_front->set_audio_samplerate(samplerate);
		_front->set_audio_extradata((uint8_t*)configstr, configstr_size);
	}
}

solids::lib::net::rtsp::client::aac_buffer_sink::~aac_buffer_sink(void)
{
}

solids::lib::net::rtsp::client::aac_buffer_sink* solids::lib::net::rtsp::client::aac_buffer_sink::createNew(solids::lib::net::rtsp::client::core * front, UsageEnvironment & env, unsigned buffer_size, int32_t channels, int32_t samplerate, char * configstr, int32_t configstr_size)
{
	return new aac_buffer_sink(front, env, buffer_size, channels, samplerate, configstr, configstr_size);
}


void solids::lib::net::rtsp::client::aac_buffer_sink::after_getting_frame(unsigned frame_size, unsigned truncated_bytes, struct timeval presentation_time, unsigned duration_msec)
{
	solids::lib::net::rtsp::client::buffer_sink::after_getting_frame(frame_size, truncated_bytes, presentation_time, duration_msec);
}
