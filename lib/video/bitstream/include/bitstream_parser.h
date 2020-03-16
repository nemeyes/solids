#ifndef _SLD_BITSTREAM_PARSER_CORE_H_
#define _SLD_BITSTREAM_PARSER_CORE_H_

#include "sld_bitstream_parser.h"
#include "bitvector.h"

namespace solids
{
	namespace lib
	{
		namespace video
		{
			namespace bitstream
			{
				class parser::core
				{
				public:
					core(void);
					~core(void);

					static void parse_video_parameter_set(int32_t video_codec, const uint8_t * vps, int32_t vpssize);
					static void parse_seq_parameter_set(int32_t video_codec, const uint8_t * sps, int32_t spssize, int32_t & width, int32_t & height);


				private:
					static void	profile_tier_level(BitVector & bv, unsigned max_sub_layers_minus1);
				private:
					static uint8_t		_vps[1000];
					static int32_t		_vps_size;
					static uint8_t		_sps[1000];
					static int32_t		_sps_size;
				};
			};
		};
	};
};











#endif