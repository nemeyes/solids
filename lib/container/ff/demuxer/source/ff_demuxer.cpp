#include "ff_demuxer.h"
#include <sld_locks.h>
#include <sld_stringhelper.h>
#include <process.h>


namespace solids
{
namespace lib
{
namespace container
{
namespace ff
{

	demuxer::core::core(solids::lib::container::ff::demuxer* front)
		: _front(front)
		, _format_ctx(NULL)
		, _stream_index(-1)
		, _bsfc(NULL)
		, _bMP4(FALSE)
		, _nCnt(0)
		, _run(FALSE)
		, _pause(FALSE)
		, _thread(INVALID_HANDLE_VALUE)
	{
		::InitializeCriticalSection(&_lock);
	}

	demuxer::core::~core(void)
	{
		::DeleteCriticalSection(&_lock);
	}

	BOOL demuxer::core::is_running(void)
	{
		return _run;
	}

	BOOL demuxer::core::is_paused(void)
	{
		return _pause;
	}

	int32_t demuxer::core::play(const char* container, int32_t type)
	{
		if (container == NULL || strlen(container) < 1)
			return solids::lib::container::ff::demuxer::err_code_t::invalid_parameter;

		_type = type;
		::strncpy_s(_container, container, MAX_PATH);
		_run = TRUE;
		if (_type == solids::lib::container::ff::demuxer::type_t::normal)
		{
			_thread = (HANDLE)::_beginthreadex(NULL, 0, solids::lib::container::ff::demuxer::core::process_cb, this, 0, NULL);
		}
		else
		{
			_nCnt = 0;
			av_init_packet(&_packet);
			_packet.data = NULL;
			_packet.size = 0;
			av_init_packet(&_packet_filtered);
			_packet_filtered.data = NULL;
			_packet_filtered.size = 0;

			if (avformat_open_input(&_format_ctx, _container, 0, NULL) != 0)
			{
				::OutputDebugStringA("[CESMMovieRTRun] avformat_open_input ERROR");
				return solids::lib::container::ff::demuxer::err_code_t::generic_fail;
			}
			if (avformat_find_stream_info(_format_ctx, NULL) < 0)
			{
				::OutputDebugStringA("[CESMMovieRTRun] avformat_find_stream_info ERROR");
				return solids::lib::container::ff::demuxer::err_code_t::generic_fail;
			}
			_stream_index = av_find_best_stream(_format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
			if (_stream_index == -1)
			{
				::OutputDebugStringA("[CESMMovieRTRun] videoStreamIdx ERROR");
				return solids::lib::container::ff::demuxer::err_code_t::generic_fail;
			}

			_bMP4 = FALSE;
			if (_stream_index >= 0)
			{
				AVCodecID videoCodec = _format_ctx->streams[_stream_index]->codecpar->codec_id;
				int32_t videoExtradataSize = _format_ctx->streams[_stream_index]->codecpar->extradata_size;
				uint8_t* videoExtradata = _format_ctx->streams[_stream_index]->codecpar->extradata;
				int32_t videoWidth = _format_ctx->streams[_stream_index]->codecpar->width;
				int32_t videoHeight = _format_ctx->streams[_stream_index]->codecpar->height;
				int32_t videoFPSNum = _format_ctx->streams[_stream_index]->r_frame_rate.num;
				int32_t videoFPSDen = _format_ctx->streams[_stream_index]->r_frame_rate.den;
				double videoFPS = (double)videoFPSNum / (double)videoFPSDen;
				int32_t videoCodec2 = solids::lib::container::ff::demuxer::video_codec_t::avc;
				_bMP4 = (videoCodec == AV_CODEC_ID_H264) || (videoCodec == AV_CODEC_ID_HEVC) && (!strcmp(_format_ctx->iformat->long_name, "QuickTime / MOV") ||
					!strcmp(_format_ctx->iformat->long_name, "FLV (Flash Video)") ||
					!strcmp(_format_ctx->iformat->long_name, "Matroska / WebM"));
				if (_bMP4)
				{
					const AVBitStreamFilter* bsf = NULL;
					if (videoCodec == AV_CODEC_ID_H264)
						bsf = av_bsf_get_by_name("h264_mp4toannexb");
					else if (videoCodec == AV_CODEC_ID_HEVC)
						bsf = av_bsf_get_by_name("hevc_mp4toannexb");
					if (!bsf)
					{
						::OutputDebugStringA("[CESMMovieRTRun] av_bsf_get_by_name() failed");
						return solids::lib::container::ff::demuxer::err_code_t::generic_fail;
					}
					av_bsf_alloc(bsf, &_bsfc);
					_bsfc->par_in = _format_ctx->streams[_stream_index]->codecpar;
					av_bsf_init(_bsfc);
				}

				switch (videoCodec)
				{
				case AV_CODEC_ID_H264:
					videoCodec2 = solids::lib::container::ff::demuxer::video_codec_t::avc;
					break;
				case AV_CODEC_ID_HEVC:
					videoCodec2 = solids::lib::container::ff::demuxer::video_codec_t::hevc;
					break;
				}

				if (_front)
					_front->on_video_begin(videoCodec2, videoExtradata, videoExtradataSize, videoWidth, videoHeight, videoFPS);
			}
		}

		return solids::lib::container::ff::demuxer::err_code_t::success;
	}

	int32_t demuxer::core::resume(void)
	{
		if (_type == solids::lib::container::ff::demuxer::type_t::normal)
			_pause = FALSE;

		return solids::lib::container::ff::demuxer::err_code_t::success;
	}

	int32_t demuxer::core::pause(void)
	{
		if (_type == solids::lib::container::ff::demuxer::type_t::normal)
			_pause = TRUE;

		return solids::lib::container::ff::demuxer::err_code_t::success;
	}

	int32_t demuxer::core::stop(void)
	{
		_run = FALSE;
		if (_type == solids::lib::container::ff::demuxer::type_t::normal)
		{
			if (_thread != NULL && _thread != INVALID_HANDLE_VALUE)
			{
				if (::WaitForSingleObject(_thread, INFINITE) == WAIT_OBJECT_0)
				{
					::CloseHandle(_thread);
					_thread = INVALID_HANDLE_VALUE;
				}
			}
			_pause = FALSE;
		}
		else
		{
			if (_packet.data)
				av_packet_unref(&_packet);

			if (_packet_filtered.data)
				av_packet_unref(&_packet_filtered);

			if (_front)
				_front->on_video_end();

			avformat_close_input(&_format_ctx);
		}

		return solids::lib::container::ff::demuxer::err_code_t::success;
	}

	int32_t demuxer::core::next(void)
	{
		if (_run)
		{
			int32_t e = 0;
			if (e = av_read_frame(_format_ctx, &_packet) >= 0)
			{
				if (_packet.stream_index == _stream_index)
				{
					if (_bMP4)
					{
						if (_packet_filtered.data)
							av_packet_unref(&_packet_filtered);
						av_bsf_send_packet(_bsfc, &_packet);
						av_bsf_receive_packet(_bsfc, &_packet_filtered);
						if (_packet_filtered.size > 0)
						{
							if (_front)
								_front->on_video_recv(_packet_filtered.data, _packet_filtered.size, _nCnt++);
						}
					}
					else
					{
						if (_packet.size > 0)
						{
							if (_front)
								_front->on_video_recv(_packet.data, _packet.size, _nCnt++);
						}
					}
				}
			}
		}

		return solids::lib::container::ff::demuxer::err_code_t::success;
	}

	void demuxer::core::process(void)
	{
		AVFormatContext* formatCtx = NULL;
		int32_t				streamIndex = -1;
		AVPacket			packet;
		AVPacket			packetFiltered;
		AVBSFContext* bsfc = NULL;

		av_init_packet(&packet);
		packet.data = NULL;
		packet.size = 0;
		av_init_packet(&packetFiltered);
		packetFiltered.data = NULL;
		packetFiltered.size = 0;

		if (avformat_open_input(&formatCtx, _container, 0, NULL) != 0)
		{
			::OutputDebugStringA("[CESMMovieRTRun] avformat_open_input ERROR");
			return;
		}
		if (avformat_find_stream_info(formatCtx, NULL) < 0)
		{
			::OutputDebugStringA("[CESMMovieRTRun] avformat_find_stream_info ERROR");
			return;
		}
		streamIndex = av_find_best_stream(formatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
		if (streamIndex == -1)
		{
			::OutputDebugStringA("[CESMMovieRTRun] videoStreamIdx ERROR");
			return;
		}

		BOOL bMP4 = FALSE;
		if (streamIndex >= 0)
		{
			AVCodecID videoCodec = formatCtx->streams[streamIndex]->codecpar->codec_id;
			int32_t videoExtradataSize = formatCtx->streams[streamIndex]->codecpar->extradata_size;
			uint8_t* videoExtradata = formatCtx->streams[streamIndex]->codecpar->extradata;
			int32_t videoWidth = formatCtx->streams[streamIndex]->codecpar->width;
			int32_t videoHeight = formatCtx->streams[streamIndex]->codecpar->height;
			int32_t videoFPSNum = formatCtx->streams[streamIndex]->r_frame_rate.num;
			int32_t videoFPSDen = formatCtx->streams[streamIndex]->r_frame_rate.den;
			double videoFPS = (double)videoFPSNum / (double)videoFPSDen;
			int32_t videoCodec2 = solids::lib::container::ff::demuxer::video_codec_t::avc;
			bMP4 = (videoCodec == AV_CODEC_ID_MPEG4) || (videoCodec == AV_CODEC_ID_H264) || (videoCodec == AV_CODEC_ID_HEVC) && (!strcmp(formatCtx->iformat->long_name, "QuickTime / MOV") ||
				!strcmp(formatCtx->iformat->long_name, "FLV (Flash Video)") ||
				!strcmp(formatCtx->iformat->long_name, "Matroska / WebM"));
			if (bMP4)
			{
				const AVBitStreamFilter* bsf = NULL;
				if (videoCodec == AV_CODEC_ID_MPEG4)
					bsf = av_bsf_get_by_name("mpeg4_unpack_bframes");
				else if (videoCodec == AV_CODEC_ID_H264)
					bsf = av_bsf_get_by_name("h264_mp4toannexb");
				else if (videoCodec == AV_CODEC_ID_HEVC)
					bsf = av_bsf_get_by_name("hevc_mp4toannexb");
			
				if (!bsf)
				{
					::OutputDebugStringA("[CESMMovieRTRun] av_bsf_get_by_name() failed");
					return;
				}
				av_bsf_alloc(bsf, &bsfc);
				bsfc->par_in = formatCtx->streams[streamIndex]->codecpar;
				av_bsf_init(bsfc);
			}

			switch (videoCodec)
			{
			case AV_CODEC_ID_MPEG4:
				videoCodec2 = solids::lib::container::ff::demuxer::video_codec_t::mp4v;
				break;
			case AV_CODEC_ID_H264:
				videoCodec2 = solids::lib::container::ff::demuxer::video_codec_t::avc;
				break;
			case AV_CODEC_ID_HEVC:
				videoCodec2 = solids::lib::container::ff::demuxer::video_codec_t::hevc;
				break;
			}

			if (_front)
				_front->on_video_begin(videoCodec2, videoExtradata, videoExtradataSize, videoWidth, videoHeight, videoFPS);
		}

		int32_t e = 0;
		int32_t nCnt = 0;
		while (_run && ((e = av_read_frame(formatCtx, &packet)) >= 0))
		{
			while (_pause && _run)
				::Sleep(10);

			if (packet.stream_index == streamIndex)
			{
				if (bMP4)
				{
					if (packetFiltered.data)
						av_packet_unref(&packetFiltered);
					av_bsf_send_packet(bsfc, &packet);
					av_bsf_receive_packet(bsfc, &packetFiltered);
					if (packetFiltered.size > 0)
					{
						if (_front)
							_front->on_video_recv(packetFiltered.data, packetFiltered.size, nCnt++);
					}
				}
				else
				{
					if (packet.size > 0)
					{
						if (_front)
							_front->on_video_recv(packet.data, packet.size, nCnt++);
					}
				}
			}
		}

		if (packet.data)
			av_packet_unref(&packet);

		if (packetFiltered.data)
			av_packet_unref(&packetFiltered);

		if (_front)
			_front->on_video_end();

		avformat_close_input(&formatCtx);
	}

	unsigned demuxer::core::process_cb(void* param)
	{
		demuxer::core* self = static_cast<demuxer::core*>(param);
		self->process();
		return 0;
	}

};
};
};
};

