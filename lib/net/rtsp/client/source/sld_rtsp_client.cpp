#include <winsock2.h>
#include <windows.h>
#include <process.h>
#include "sld_rtsp_client.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "rtsp_client.h"

solids::lib::net::rtsp::client::client(void)
	: _ignore_sdp(TRUE)
{
	WSADATA wsd;
	WSAStartup( MAKEWORD(2,2), &wsd );
}

solids::lib::net::rtsp::client::~client(void)
{
	on_end_video();
	on_end_audio();

	WSACleanup();
}

int32_t solids::lib::net::rtsp::client::play(const char * url, const char * username, const char * password, int32_t transport_option, int32_t recv_option, int32_t recv_timeout, float scale, bool repeat)
{
    if( !url || strlen(url)<1 )
		return solids::lib::net::rtsp::client::err_code_t::generic_fail;

	memset(_url, 0x00, sizeof(_url));
	memset(_username, 0x00, sizeof(_username));
	memset(_password, 0x00, sizeof(_password));
	if (strlen(url)>0)
		strcpy_s(_url, url);
	if (username && strlen(username)>0)
		strcpy_s(_username, username);
	if (password && strlen(password)>0)
		strcpy_s(_password, password);
	_transport_option = transport_option;
	_recv_option = recv_option;
	_recv_timeout = recv_timeout;
	_scale = scale;
	_repeat = repeat;

	unsigned int thread_id;
	_thread = (HANDLE)::_beginthreadex(0, 0, solids::lib::net::rtsp::client::process_cb, this, 0, &thread_id);
	return solids::lib::net::rtsp::client::err_code_t::success;
}

int32_t solids::lib::net::rtsp::client::stop(void)
{
	if (!_kill )
	{
		_repeat = false;
		_kill = true;
		if (_live)
			_live->close();
	}

	/*
	if (_thread != NULL && _thread !=INVALID_HANDLE_VALUE)
	{
		if(::WaitForSingleObject(_thread, INFINITE)==WAIT_OBJECT_0)
		{
			::CloseHandle(_thread);
			_thread = INVALID_HANDLE_VALUE;
		}
	}
	*/
	return solids::lib::net::rtsp::client::err_code_t::success;
}

int32_t solids::lib::net::rtsp::client::pause(void)
{
	if (!_kill)
	{
		if (_live)
		{
			_live->start_pausing_session();
		}
	}
	return solids::lib::net::rtsp::client::err_code_t::success;
}

int32_t solids::lib::net::rtsp::client::width(void)
{
	if (_live)
		return _live->width();
	return 0;
}

int32_t solids::lib::net::rtsp::client::height(void)
{
	if (_live)
		return _live->height();
	return 0;
}

void solids::lib::net::rtsp::client::on_begin_video(int32_t codec, uint8_t * extradata, int32_t extradata_size, int32_t width, int32_t height)
{

}

void solids::lib::net::rtsp::client::on_recv_video(uint8_t* bytes, int32_t nbytes, long long pts, long long duration)
{

}

void solids::lib::net::rtsp::client::on_end_video(void)
{

}

void solids::lib::net::rtsp::client::on_begin_audio(int32_t codec, uint8_t* extradata, int32_t extradata_size, int32_t samplerate, int32_t channels)
{

}

void solids::lib::net::rtsp::client::on_recv_audio(uint8_t * bytes, int32_t nbytes, long long pts, long long duration)
{

}

void solids::lib::net::rtsp::client::on_end_audio(void)
{

}

void solids::lib::net::rtsp::client::process(void)
{
	do
	{
		TaskScheduler * sched = BasicTaskScheduler::createNew();
		UsageEnvironment * env = BasicUsageEnvironment::createNew(*sched);
		if (strlen(_username) > 0 && strlen(_password) > 0)
			_live = solids::lib::net::rtsp::client::core::createNew(this, *env, _url, _username, _password, _transport_option, _recv_option, _recv_timeout, _scale, 0, &_kill);
		else
			_live = solids::lib::net::rtsp::client::core::createNew(this, *env, _url, 0, 0, _transport_option, _recv_option, _recv_timeout, _scale, 0, &_kill);

		_kill = false;
		solids::lib::net::rtsp::client::core::continue_after_client_creation(_live);
		env->taskScheduler().doEventLoop((char*)&_kill);

		if (env)
		{
			env->reclaim();
			env = 0;
		}
		if (sched)
		{
			delete sched;
			sched = 0;
		}
	} while (_repeat);
}

#if defined(WIN32)
unsigned __stdcall solids::lib::net::rtsp::client::process_cb(void * param)
{
	solids::lib::net::rtsp::client * self = static_cast<solids::lib::net::rtsp::client*>(param);
	self->process();
	return 0;
}
#else
void* debuggerking::live_rtsp_client::process_cb(void * param)
{
	rtsp_client * self = static_cast<rtsp_client*>(param);
	self->process();
	return 0;
}
#endif


/*
const int32_t solids::lib::net::rtsp::client::find_nal_unit(uint8_t * bitstream, int32_t size, int * nal_start, int * nal_end)
{
	int32_t i;
	// find start
	*nal_start = 0;
	*nal_end = 0;

	i = 0;
	//( next_bits( 24 ) != 0x000001 && next_bits( 32 ) != 0x00000001 )
	while ((bitstream[i] != 0 || bitstream[i + 1] != 0 || bitstream[i + 2] != 0x01) &&
		(bitstream[i] != 0 || bitstream[i + 1] != 0 || bitstream[i + 2] != 0 || bitstream[i + 3] != 0x01))
	{
		i++; // skip leading zero
		if (i + 4 >= size)
		{
			return 0;
		} // did not find nal start
	}

	if (bitstream[i] != 0 || bitstream[i + 1] != 0 || bitstream[i + 2] != 0x01) // ( next_bits( 24 ) != 0x000001 )
	{
		i++;
	}

	if (bitstream[i] != 0 || bitstream[i + 1] != 0 || bitstream[i + 2] != 0x01)
	{
		return 0;
	}

	i += 3;
	*nal_start = i;

	//( next_bits( 24 ) != 0x000000 && next_bits( 24 ) != 0x000001 )
	while ((bitstream[i] != 0 || bitstream[i + 1] != 0 || bitstream[i + 2] != 0) &&
		(bitstream[i] != 0 || bitstream[i + 1] != 0 || bitstream[i + 2] != 0x01))
	{
		i++;
		// FIXME the next line fails when reading a nal that ends exactly at the end of the data
		if (i + 3 >= size)
		{
			*nal_end = size;
			return -1;
		} // did not find nal end, stream ended first
	}

	*nal_end = i;
	return (*nal_end - *nal_start);
}


const uint8_t * solids::lib::net::rtsp::client::find_start_code(const uint8_t * __restrict begin, const uint8_t * end, uint32_t * __restrict state)
{
	int i;
	if (begin >= end)
		return end;

	for (i = 0; i < 3; i++)
	{
		uint32_t tmp = *state << 8;
		*state = tmp + *(begin++);
		if (tmp == 0x100 || begin == end)
			return begin;
	}

	while (begin < end)
	{
		if (begin[-1] > 1)
			begin += 3;
		else if (begin[-2])
			begin += 2;
		else if (begin[-3] | (begin[-1] - 1))
			begin++;
		else
		{
			begin++;
			break;
		}
	}

	//begin = std::min(begin, end) - 4;
	*state = AV_RB32(begin);
	return begin + 4;
}
*/