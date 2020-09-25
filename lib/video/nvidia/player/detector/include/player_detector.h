#pragma once

#include "sld_player_detector.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudalegacy.hpp>

#include <nppi_filtering_functions.h>

namespace solids
{
	namespace lib
	{
		namespace video
		{
			namespace nvidia
			{
				namespace player
				{

					class detector::core
					{
					public:
						core(solids::lib::video::nvidia::player::detector* front);
						~core(void);

						int32_t initialize(solids::lib::video::nvidia::player::detector::context_t* ctx);
						int32_t release(void);

						int32_t detect(uint8_t* input, int32_t inputStride, uint8_t** output, int32_t& outputStride);

					private:
						solids::lib::video::nvidia::player::detector* _front;
						solids::lib::video::nvidia::player::detector::context_t* _ctx;

#if defined(WITH_HOG)
						const int32_t			_win_width = 48;
						const int32_t			_cell_width = 8;
						const int32_t			_nbins = 9;
						const int32_t			_win_stride_width = 48;
						const int32_t			_win_stride_height = _win_stride_width;
						const int32_t			_block_width = _win_stride_width * 2;

						const int32_t			_block_stride_width = _block_width / 2;
						const int32_t			_block_stride_height = _block_width / 2;

						const int32_t			_hog_levels = cv::HOGDescriptor::DEFAULT_NLEVELS;
						const int32_t			_hog_group_thredshold = 16;

						cv::Ptr<cv::cuda::HOG>	_hog;
						cv::Size				_win_stride;
						cv::Size				_win_size;
						cv::Size				_block_size;
						cv::Size				_block_stride;
						cv::Size				_cell_size;
#elif defined(WITH_HARR)


#elif defined(WITH_MOG)
						cv::Ptr<cv::BackgroundSubtractor>	_mog;
						cv::Ptr<cv::BackgroundSubtractor>	_mog2;
						cv::cuda::GpuMat					_foregroundmask;
						cv::cuda::GpuMat					_foregroundimg;
						cv::cuda::GpuMat					_backgroundimg;
						BOOL								_first_frame;
						cv::Ptr<cv::cuda::Filter>			_filter[5];
#elif defined(WITH_GSOC)
						cv::Ptr<cv::BackgroundSubtractor>	_gsoc;
						cv::cuda::GpuMat					_target;
#else
						cv::cuda::GpuMat					_backgroundimg;
						cv::cuda::GpuMat					_target;
						cv::Ptr<cv::cuda::Filter>			_filter[5];
#endif
					};


				};
			};
		};
	};
};
