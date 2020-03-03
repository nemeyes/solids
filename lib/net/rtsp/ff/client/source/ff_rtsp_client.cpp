#include "ff_rtsp_client.h"
#include <process.h>

sld::lib::net::rtsp::ff::client::core::core(sld::lib::net::rtsp::ff::client* front)
	: _front(front)
	, _fmt_ctx(nullptr)
	, _thread(INVALID_HANDLE_VALUE)
	, _run(FALSE)
{

}

sld::lib::net::rtsp::ff::client::core::~core(void)
{

}

int32_t sld::lib::net::rtsp::ff::client::core::play(const char * url, int32_t transport, int32_t stimeout)
{
	if (url && strlen(url)>0)
		strncpy_s(_url, url, strlen(url));
	_transport = transport;
	_session_timeout = stimeout;

	_run = TRUE;
	_thread = (HANDLE)::_beginthreadex(NULL, 0, sld::lib::net::rtsp::ff::client::core::process_cb, this, 0, NULL);
	return sld::lib::net::rtsp::ff::client::err_code_t::success;
}

int32_t sld::lib::net::rtsp::ff::client::core::stop(void)
{
	_run = FALSE;
	if (_thread != NULL && _thread != INVALID_HANDLE_VALUE)
	{
		if (::WaitForSingleObject(_thread, INFINITE) == WAIT_OBJECT_0)
		{
			::CloseHandle(_thread);
			_thread = INVALID_HANDLE_VALUE;
		}
	}
	return sld::lib::net::rtsp::ff::client::err_code_t::success;
}

void sld::lib::net::rtsp::ff::client::core::process(void)
{
	int ret = 0;
	_fmt_ctx = ::avformat_alloc_context();

	AVDictionary* dic = NULL;

	char stimeout[MAX_PATH] = { 0 };
	int32_t sessionTimeout = _session_timeout * 1000000;
	::_itoa_s(sessionTimeout, stimeout, 10);
	::av_dict_set(&dic, "stimeout", stimeout, 0);

	switch (_transport)
	{
	case sld::lib::net::rtsp::ff::client::transport_t::rtp_over_tcp:
		::av_dict_set(&dic, "rtsp_transport", "tcp", 0);
		break;
	case sld::lib::net::rtsp::ff::client::transport_t::rtp_over_udp:
		::av_dict_set(&dic, "rtsp_transport", "udp", 0);
		break;
	}
	av_dict_set(&dic, "buffer_size", "37748736", 0);		// INT_MAX 2147483647 // 655360 -> 36MB 37748736
	av_dict_set_int(&dic, "fifo_size", 37748736, 0);
	av_dict_set(&dic, "recv_buffer_size", "37748736", 0);
	av_dict_set(&dic, "send_buffer_size", "37748736", 0);

	ret = ::avformat_open_input(&_fmt_ctx, _url, NULL, &dic);
	::av_dict_free(&dic);
	if (ret < 0)
		return;

	ret = avformat_find_stream_info(_fmt_ctx, NULL);
	if (ret < 0)
		return;

	av_dump_format(_fmt_ctx, 0, _url, 0);

	_stream_index[core::stream_index_t::video] = ::av_find_best_stream(_fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
	_stream_index[core::stream_index_t::audio] = ::av_find_best_stream(_fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);

	AVCodecID videoCodec = AV_CODEC_ID_NONE;
	if (_stream_index[core::stream_index_t::video] >= 0)


	{
		videoCodec			= _fmt_ctx->streams[_stream_index[core::stream_index_t::video]]->codecpar->codec_id;
		int32_t videoWidth	= _fmt_ctx->streams[_stream_index[core::stream_index_t::video]]->codecpar->width;
		int32_t videoHeight = _fmt_ctx->streams[_stream_index[core::stream_index_t::video]]->codecpar->height;
		int32_t videoFPSNum = _fmt_ctx->streams[_stream_index[core::stream_index_t::video]]->r_frame_rate.num;
		int32_t videoFPSDen = _fmt_ctx->streams[_stream_index[core::stream_index_t::video]]->r_frame_rate.den;
		double videoFPS		= (double)videoFPSNum / (double)videoFPSDen;
		int32_t videoExtradataSize	= _fmt_ctx->streams[_stream_index[core::stream_index_t::video]]->codecpar->extradata_size;
		uint8_t * videoExtradata	= _fmt_ctx->streams[_stream_index[core::stream_index_t::video]]->codecpar->extradata;
		int32_t videoNum	= _fmt_ctx->streams[_stream_index[core::stream_index_t::video]]->time_base.num;
		int32_t videoDen	= _fmt_ctx->streams[_stream_index[core::stream_index_t::video]]->time_base.den;
		int32_t videoCodec2 = sld::lib::net::rtsp::ff::client::video_codec_t::avc;
		switch (videoCodec)
		{
		case AV_CODEC_ID_H264:
			videoCodec2 = sld::lib::net::rtsp::ff::client::video_codec_t::avc;
			break;
		case AV_CODEC_ID_HEVC:
			videoCodec2 = sld::lib::net::rtsp::ff::client::video_codec_t::hevc;
			break;
		}

		_front->on_begin_video(videoCodec2, videoExtradata, videoExtradataSize);
	}
	else
	{
		return;
	}

	if (_stream_index[core::stream_index_t::audio] >= 0)
	{
		int32_t audioCodec = _fmt_ctx->streams[_stream_index[core::stream_index_t::audio]]->codecpar->codec_id;
		int32_t audioSamplerate = _fmt_ctx->streams[_stream_index[core::stream_index_t::audio]]->codecpar->sample_rate;
		int32_t audioChannels = _fmt_ctx->streams[_stream_index[core::stream_index_t::audio]]->codecpar->channels;
		int32_t audioSampleformat = _fmt_ctx->streams[_stream_index[core::stream_index_t::audio]]->codecpar->format;
		int32_t audioExtradataSize = _fmt_ctx->streams[_stream_index[core::stream_index_t::audio]]->codecpar->extradata_size;
		uint8_t * audioExtradata = _fmt_ctx->streams[_stream_index[core::stream_index_t::audio]]->codecpar->extradata;
		int32_t audioNum = _fmt_ctx->streams[_stream_index[core::stream_index_t::audio]]->time_base.num;
		int32_t audioDen = _fmt_ctx->streams[_stream_index[core::stream_index_t::audio]]->time_base.den;

		int32_t audioCodec2 = sld::lib::net::rtsp::ff::client::audio_codec_t::aac;
		switch (audioCodec)
		{
		case AV_CODEC_ID_AAC:
			audioCodec2 = sld::lib::net::rtsp::ff::client::audio_codec_t::aac;
			break;
		case AV_CODEC_ID_MP3:
			audioCodec2 = sld::lib::net::rtsp::ff::client::audio_codec_t::mp3;
			break;
		case AV_CODEC_ID_AC3:
			audioCodec2 = sld::lib::net::rtsp::ff::client::audio_codec_t::ac3;
			break;
		case AV_CODEC_ID_OPUS:
			audioCodec2 = sld::lib::net::rtsp::ff::client::audio_codec_t::opus;
			break;
		}

		_front->on_begin_audio(audioCodec2, audioExtradata, audioExtradataSize, audioSamplerate, audioChannels);
	}

	BOOL bWaitFirstVideo = TRUE;
	int64_t videoStartTime = 0;
	while (_run)
	{
		AVPacket* pkt = av_packet_alloc();
		av_init_packet(pkt);

		if ((ret = av_read_frame(_fmt_ctx, pkt)) >= 0 && _run)
		{
			if (pkt->stream_index == _stream_index[core::stream_index_t::video])
			{
				if (pkt->size > 0)
				{
					if (bWaitFirstVideo)
					{
						if (videoCodec == AV_CODEC_ID_H264)
						{
							if (((pkt->data[4] & 0x1F) == 0x07) || ((pkt->data[4] & 0x1F) == 0x08) || ((pkt->data[4] & 0x1F) == 0x05))
							{
								videoStartTime = pkt->pts;
								bWaitFirstVideo = FALSE;
							}
							else
								continue;
						}
						else if (videoCodec == AV_CODEC_ID_HEVC)
						{
							/*
													if ((((pkt->data[4] >> 1) & 0x3F) == 0x20) ||
														(((pkt->data[4] >> 1) & 0x3F) == 0x21) ||
														(((pkt->data[4] >> 1) & 0x3F) == 0x22) ||
														(((pkt->data[4] >> 1) & 0x3F) == 0x13) ||
														(((pkt->data[4] >> 1) & 0x3F) == 0x14))
							*/
							char debug[MAX_PATH] = { 0 };
							_snprintf_s(debug, MAX_PATH, "%.2x %.2x %.2x %.2x %.2x\n", pkt->data[0], pkt->data[1], pkt->data[2], pkt->data[3], pkt->data[4]);
							::OutputDebugStringA(debug);

							if ((((pkt->data[3] >> 1) & 0x3F) == 0x20) ||
								(((pkt->data[3] >> 1) & 0x3F) == 0x21) ||
								(((pkt->data[3] >> 1) & 0x3F) == 0x22) ||
								(((pkt->data[3] >> 1) & 0x3F) == 0x13) ||
								(((pkt->data[3] >> 1) & 0x3F) == 0x14))
								bWaitFirstVideo = FALSE;
							else
								continue;
						}
					}
					if (!bWaitFirstVideo)
						_front->on_recv_video(pkt->data, pkt->size, ((pkt->pts - videoStartTime) * 10000000) / _fmt_ctx->streams[_stream_index[core::stream_index_t::video]]->time_base.den, (pkt->duration * 10000000) / _fmt_ctx->streams[_stream_index[core::stream_index_t::video]]->time_base.den);
				}
			}
			else if (pkt->stream_index == _stream_index[core::stream_index_t::audio])
			{
				if (pkt->size > 0)
				{
					_front->on_recv_audio(pkt->data, pkt->size, (pkt->pts * 10000000) / _fmt_ctx->streams[_stream_index[core::stream_index_t::audio]]->time_base.den, (pkt->duration * 10000000) / _fmt_ctx->streams[_stream_index[core::stream_index_t::audio]]->time_base.den);
				}
			}
		}

		av_packet_free(&pkt);
	}


	if (_stream_index[core::stream_index_t::video] >= 0)
		_front->on_end_video();
	if (_stream_index[core::stream_index_t::audio] >= 0)
		_front->on_end_audio();

	::avformat_close_input(&_fmt_ctx);
	::avformat_free_context(_fmt_ctx);
	_fmt_ctx = NULL;
}

unsigned sld::lib::net::rtsp::ff::client::core::process_cb(void* param)
{
	sld::lib::net::rtsp::ff::client::core * self = static_cast<sld::lib::net::rtsp::ff::client::core*>(param);
	self->process();
	return 0;
}