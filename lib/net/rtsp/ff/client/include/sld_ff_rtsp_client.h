#ifndef _SLD_FF_RTSP_CLIENT_H_
#define _SLD_FF_RTSP_CLIENT_H_

#if defined(EXPORT_SLD_RTSP_CLIENT_LIB)
#  define EXP_SLD_RTSP_CLIENT_CLASS __declspec(dllexport)
#else
#  define EXP_SLD_RTSP_CLIENT_CLASS __declspec(dllimport)
#endif

#include <sld.h>

namespace sld
{
	namespace lib
	{
		namespace net
		{
			namespace rtsp
			{
				namespace ff
				{
					class EXP_SLD_RTSP_CLIENT_CLASS client
						: public sld::lib::base
					{
					public:
						class core;
					public:
						typedef struct _transport_t
						{
							static const int32_t unknown = -1;
							static const int32_t rtp_over_udp = 0;
							static const int32_t rtp_over_tcp = 1;
						} transport_t;

						client(void);
						virtual ~client(void);

						int32_t play(const char * url, int32_t transport, int32_t stimeout);
						int32_t stop(void);

						virtual void on_begin_video(int32_t codec, uint8_t * extradata, int32_t extradata_size);
						virtual void on_recv_video(uint8_t* bytes, int32_t nbytes, long long pts, long long duration);
						virtual void on_end_video(void);

						virtual void on_begin_audio(int32_t codec, uint8_t* extradata, int32_t extradata_size, int32_t samplerate, int32_t channels);
						virtual void on_recv_audio(uint8_t* bytes, int32_t nbytes, long long pts, long long duration);
						virtual void on_end_audio(void);

					private:
						sld::lib::net::rtsp::ff::client::core* _core;
					};
				};
			};
		};
	};
};

#endif