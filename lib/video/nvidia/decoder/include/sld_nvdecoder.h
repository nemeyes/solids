#pragma once

#if defined(EXP_SLD_NVDECODER_LIB)
#define EXP_SLD_NVDECODER_CLS __declspec(dllexport)
#else
#define EXP_SLD_NVDECODER_CLS __declspec(dllimport)
#endif

#include <sld.h>

namespace solids
{
	namespace lib
	{
		namespace video
		{
			namespace nvidia
			{
				class EXP_SLD_NVDECODER_CLS decoder
					: public solids::lib::base
				{
				public:
					class core;
					class resizer;
					class converter;
				public:
					typedef struct _context_t
					{
						int deviceIndex;
						int width;
						int height;
						int codec;
						int colorspace;
						_context_t(void)
							: deviceIndex(0)
							, width(3840)
							, height(2160)
							, codec(solids::lib::video::nvidia::decoder::video_codec_t::avc)
							, colorspace(solids::lib::video::nvidia::decoder::colorspace_t::bgra)
						{}
					} context_t;


					decoder(void);
					virtual ~decoder(void);

					BOOL	is_initialized(void);
					void*	context(void);

					int32_t	initialize(solids::lib::video::nvidia::decoder::context_t* ctx);
					int32_t	release(void);
					int32_t	decode(uint8_t* bitstream, int32_t bitstreamSize, long long bitstreamTimestamp, uint8_t*** decoded, int32_t* numberOfDecoded, long long** timetstamp);

					size_t	get_pitch(void);
					size_t	get_pitch_resized(void);
					size_t	get_pitch_converted(void);
					size_t	get_pitch2(void);

				private:
					decoder(const decoder& clone);

				private:
					decoder::core * _core;
				};
			};
		};
	};
};

