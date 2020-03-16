#ifndef _SLD_BITSTREAM_PARSER_H_
#define _SLD_BITSTREAM_PARSER_H_

#if defined(EXPORT_SLD_BITSTREAM_PARSER_LIB)
#  define EXP_SLD_BITSTREAM_PARSER_CLASS __declspec(dllexport)
#else
#  define EXP_SLD_BITSTREAM_PARSER_CLASS __declspec(dllimport)
#endif

#include <sld.h>


namespace solids
{
	namespace lib
	{
		namespace video
		{
			namespace bitstream
			{
				class EXP_SLD_BITSTREAM_PARSER_CLASS parser
					: public solids::lib::base
				{
					class core;
				public:
					parser(void);
					~parser(void);

					static void parse_video_parameter_set(int32_t video_codec, const uint8_t * vps, int32_t vpssize);
					static void parse_seq_parameter_set(int32_t video_codec, const uint8_t * sps, int32_t spssize, int32_t & width, int32_t & height);
				};
			};
		};
	};
};














#endif