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

				class decoder::resizer
				{
				public:
					static void resize_nv12(unsigned char* dstNV12, int dstNV12Pitch, int dstNV12Width, int dstNV12Height, unsigned char* srcNV12, int srcNV12Picth, int srcNV12Width, int srcNV12Height);
				};

			};
		};
	};
};



