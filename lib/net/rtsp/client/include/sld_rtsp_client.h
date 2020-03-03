#ifndef _SLD_RTSP_CLIENT_H_
#define _SLD_RTSP_CLIENT_H_

#if defined(EXPORT_SLD_RTSP_CLIENT_LIB)
#define EXP_SLD_RTSP_CLIENT_CLASS __declspec(dllexport)
#else
#define EXP_SLD_RTSP_CLIENT_CLASS __declspec(dllimport)
#endif

#include <sld.h>

namespace solids
{
	namespace lib
	{
		namespace net
		{
			namespace rtsp
			{
				class EXP_SLD_RTSP_CLIENT_CLASS client
					: public solids::lib::base
				{
				public:
					class core;
					class buffer_sink;
					class h2645_buffer_sink;
					class h265_buffer_sink;
					class h264_buffer_sink;
					class aac_buffer_sink;
				public:
					typedef struct _transport_option_t
					{
						static const int32_t rtp_over_udp = 0;
						static const int32_t rtp_over_tcp = 1;
						static const int32_t rtp_over_http = 2;
					} transport_option_t;

					client(void);
					virtual ~client(void);
					int32_t play(const char * url, const char * username, const char * password, int32_t transport_option, int32_t recv_option, int32_t recv_timeout, float scale = 1.f, bool repeat = true);
					int32_t stop(void);
					int32_t pause(void);

					virtual void on_begin_video(int32_t codec, uint8_t * extradata, int32_t extradata_size);
					virtual void on_recv_video(uint8_t * bytes, int32_t nbytes, long long pts, long long duration);
					virtual void on_end_video(void);

					virtual void on_begin_audio(int32_t codec, uint8_t* extradata, int32_t extradata_size, int32_t samplerate, int32_t channels);
					virtual void on_recv_audio(uint8_t* bytes, int32_t nbytes, long long pts, long long duration);
					virtual void on_end_audio(void);

				private:
					void process(void);
					static unsigned __stdcall process_cb(void * param);
					HANDLE _thread;

				private:
					solids::lib::net::rtsp::client::core * _live;

					char	_url[260];
					char	_username[260];
					char	_password[260];
					int32_t _transport_option;
					int32_t _recv_option;
					int32_t _recv_timeout;
					float	_scale;
					BOOL 	_repeat;
					bool	_kill;
					BOOL	_ignore_sdp;
				};
			};
		};
	};
};

#endif
