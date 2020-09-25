#pragma once

#include "sld_nvdecoder.h"

namespace solids
{
	namespace lib
	{
		namespace video
		{
			namespace nvidia
			{
				class decoder::converter
				{
				public:
					typedef struct _standard_t
					{
						static const int bt709 = 0;
						static const int bt601 = 2;
						static const int bt2020 = 4;
					} standard_t;

					static const int RESOLUTION_W4K = 3840;
					static const int RESOLUTION_H4K = 2160;
					static const int RESOLUTION_WFHD = 1920;
					static const int RESOLUTION_HFHD = 1080;

					static void convert_nv12_to_bgra32(unsigned char* nv12, int nv12Pitch, unsigned char* bgra, int bgraPicth, int width, int height);
					static void convert_nv12_to_i420(unsigned char* nv12, int nv12Pitch, unsigned char* i420, int i420Pitch, int width, int height);
					static void convert_nv12_to_yv12(unsigned char* nv12, int nv12Pitch, unsigned char* yv12, int yv12Pitch, int width, int height);
				private:
					static void setup_mat_yuv2rgb(int mat);
					static void constants(int mat, float& wr, float& wb, int& black, int& white, int& max);
				};
			};
		};
	};
};

