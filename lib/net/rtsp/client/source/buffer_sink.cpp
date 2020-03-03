#include "buffer_sink.h"
#include <GroupsockHelper.hh>

solids::lib::net::rtsp::client::buffer_sink::buffer_sink(solids::lib::net::rtsp::client::core * front, int32_t mt, int32_t codec, UsageEnvironment & env, unsigned buffer_size)
    : MediaSink(env)
	, _front(front)
    , _buffer_size(buffer_size)
	, _mt(mt)
{
	if (_mt == solids::lib::net::rtsp::client::media_type_t::video)
		_vcodec = codec;
	if (_mt == solids::lib::net::rtsp::client::media_type_t::audio)
		_acodec = codec;
    _buffer = new unsigned char[buffer_size];
}

solids::lib::net::rtsp::client::buffer_sink::~buffer_sink(void)
{
	if (_buffer)
	{
		delete[] _buffer;
		_buffer = 0;
	}
}

solids::lib::net::rtsp::client::buffer_sink * solids::lib::net::rtsp::client::buffer_sink::createNew(solids::lib::net::rtsp::client::core * front, int32_t mt, int32_t codec, UsageEnvironment & env, unsigned buffer_size)
{
	return new buffer_sink(front, mt, codec, env, buffer_size);
}

Boolean solids::lib::net::rtsp::client::buffer_sink::continuePlaying(void)
{
    if( !fSource )
        return False;

    fSource->getNextFrame(_buffer, _buffer_size, after_getting_frame, this, onSourceClosure, this);
    return True;
}

void solids::lib::net::rtsp::client::buffer_sink::after_getting_frame(void * param, unsigned frame_size, unsigned truncated_bytes, struct timeval presentation_time, unsigned duration_msec)
{
	solids::lib::net::rtsp::client::buffer_sink * sink = static_cast<solids::lib::net::rtsp::client::buffer_sink*>(param);
	sink->after_getting_frame(frame_size, truncated_bytes, presentation_time, duration_msec);
}

void solids::lib::net::rtsp::client::buffer_sink::add_data(unsigned char * data, unsigned data_size, struct timeval presentation_time, unsigned duration_msec)
{
	long long pts = (presentation_time.tv_sec * 10000000i64) + (presentation_time.tv_usec * 10i64);
	if (_front)
	{
		if (_mt == solids::lib::net::rtsp::client::media_type_t::video)
		{
			_front->put_video_sample(_vcodec, data, data_size, pts);
		}
		else if (_mt == solids::lib::net::rtsp::client::media_type_t::audio)
		{
			_front->put_audio_sample(_acodec, data, data_size, pts);
		}
	}
}

void solids::lib::net::rtsp::client::buffer_sink::after_getting_frame(unsigned frame_size, unsigned truncated_bytes, struct timeval presentation_time, unsigned duration_msec)
{
	if (_front)
	{
		if (_mt == solids::lib::net::rtsp::client::media_type_t::video)
		{
			if (_vcodec == solids::lib::net::rtsp::client::video_codec_t::avc)
			{
				const unsigned char start_code[4] = { 0x00, 0x00, 0x00, 0x01 };
				if ((_buffer[0] == start_code[0]) && (_buffer[1] == start_code[1]) && (_buffer[2] == start_code[2]) && (_buffer[3] == start_code[3]))
					add_data(_buffer, frame_size, presentation_time, duration_msec);
				else
				{
					if (truncated_bytes > 0)
						::memmove(_buffer + 4, _buffer, frame_size - 4);
					else
					{
						truncated_bytes = (frame_size + 4) - _buffer_size;
						if (truncated_bytes > 0 && (frame_size + 4) > _buffer_size)
							::memmove(_buffer + 4, _buffer, frame_size - truncated_bytes);
						else
							::memmove(_buffer + 4, _buffer, frame_size);
					}
					::memmove(_buffer, start_code, sizeof(start_code));
					add_data(_buffer, frame_size + sizeof(start_code), presentation_time, duration_msec);
				}
			} 
			else if (_vcodec == solids::lib::net::rtsp::client::video_codec_t::hevc)
			{
				const unsigned char start_code[4] = { 0x00, 0x00, 0x00, 0x01 };
				if ((_buffer[0] == start_code[0]) && (_buffer[1] == start_code[1]) && (_buffer[2] == start_code[2]) && (_buffer[3] == start_code[3]))
					add_data(_buffer, frame_size, presentation_time, duration_msec);
				else
				{
					if (truncated_bytes > 0)
						::memmove(_buffer + 4, _buffer, frame_size - 4);
					else
					{
						truncated_bytes = (frame_size + 4) - _buffer_size;
						if (truncated_bytes > 0 && (frame_size + 4) > _buffer_size)
							::memmove(_buffer + 4, _buffer, frame_size - truncated_bytes);
						else
							::memmove(_buffer + 4, _buffer, frame_size);
					}
					::memmove(_buffer, start_code, sizeof(start_code));
					add_data(_buffer, frame_size + sizeof(start_code), presentation_time, duration_msec);
				}
			}
		}
		else if (_mt == solids::lib::net::rtsp::client::media_type_t::audio)
		{
			add_data(_buffer, frame_size, presentation_time, duration_msec);
		}
	}
    continuePlaying();
}