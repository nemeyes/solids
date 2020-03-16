#include "sld_bitstream_parser.h"
#include "bitstream_parser.h"

solids::lib::video::bitstream::parser::parser(void)
{

}

solids::lib::video::bitstream::parser::~parser(void)
{

}

void solids::lib::video::bitstream::parser::parse_video_parameter_set(int32_t video_codec, const uint8_t * vps, int32_t vpssize)
{
	solids::lib::video::bitstream::parser::core::parse_video_parameter_set(video_codec, vps, vpssize);
}

void solids::lib::video::bitstream::parser::parse_seq_parameter_set(int32_t video_codec, const uint8_t * sps, int32_t spssize, int32_t & width, int32_t & height)
{
	solids::lib::video::bitstream::parser::core::parse_seq_parameter_set(video_codec, sps, spssize, width, height);
}