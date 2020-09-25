#include "player_detector.h"

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

	detector::core::core(solids::lib::video::nvidia::player::detector* front)
		: _front(front)
		, _ctx(NULL)
#if defined(WITH_HOG)
		, _win_stride(_win_stride_width, _win_stride_height)
		, _win_size(_win_width, _win_width * 2)
		, _block_size(_block_width, _block_width)
		, _block_stride(_block_stride_width, _block_stride_height)
		, _cell_size(_cell_width, _cell_width)
#elif defined(WITH_HARR)

#elif defined(WITH_MOG)
		, _first_frame(TRUE)
#else
#endif
	{


	}

	detector::core::~core(void)
	{

	}

	int32_t detector::core::initialize(solids::lib::video::nvidia::player::detector::context_t* ctx)
	{
		_ctx = ctx;

#if defined(WITH_HOG)
		_hog = cv::cuda::HOG::create();
		cv::Mat detector = _hog->getDefaultPeopleDetector();
		_hog->setSVMDetector(detector);
#elif defined(WITH_HARR)


#elif defined(WITH_MOG)
#if defined(WITH_LABELLING)
		_mog = cv::cuda::createBackgroundSubtractorMOG();
		//_mog2 = cv::cuda::createBackgroundSubtractorMOG2(500, 32.0, false);
		cv::Mat mc = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)/*, cv::Point(11, 11)*/);
		cv::Mat md = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)/*, cv::Point(4, 4)*/);
		cv::Mat mo = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)/*, cv::Point(11, 11)*/);
		_filter[0] = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 1);
		_filter[1] = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, md, cv::Point(-1, -1), 3);
		_filter[2] = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, mc, cv::Point(-1, -1), 7);
		_filter[3] = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, mo, cv::Point(-1, -1), 3);
		_filter[4] = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 7);
#else
		cv::Mat background = cv::imread("background.png", cv::IMREAD_GRAYSCALE);
		_backgroundimg.create(background.rows, background.cols, CV_8UC1);
		_backgroundimg.upload(background);

		_mog = cv::cuda::createBackgroundSubtractorMOG();
		//_mog2 = cv::cuda::createBackgroundSubtractorMOG2(500, 32.0, false);
		cv::Mat mc = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)/*, cv::Point(11, 11)*/);
		cv::Mat md = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)/*, cv::Point(4, 4)*/);
		cv::Mat mo = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)/*, cv::Point(11, 11)*/);
		_filter[0] = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(3, 3), 3);
		_filter[1] = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, md, cv::Point(-1, -1), 15);
		_filter[2] = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, mc, cv::Point(-1, -1), 10);
		//_filter[2] = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, mc, cv::Point(-1, -1), 5);
		_filter[3] = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, mo, cv::Point(-1, -1), 3);
		_filter[4] = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 7);
#endif
#elif defined(WITH_GSOC)
		_gsoc = cv::bgsegm::createBackgroundSubtractorGSOC();
		_target.create(_ctx->height, _ctx->width, CV_8UC4);
#else
		cv::Mat background = cv::imread("F:\\workspace\\Pulsar\\Pulsar\\Out\\x64\\Debug\\bin\\background.png", cv::IMREAD_GRAYSCALE);
		//cv::equalizeHist(background, background);
		_backgroundimg.create(background.rows, background.cols, CV_8UC1);
		_backgroundimg.upload(background);
		//cv::imshow("equal hist", background);


		_target.create(_ctx->height, _ctx->width, CV_8UC4);



		cv::Mat mc = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)/*, cv::Point(11, 11)*/);
		cv::Mat md = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)/*, cv::Point(4, 4)*/);
		cv::Mat mo = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)/*, cv::Point(11, 11)*/);
		_filter[0] = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 1);
		_filter[1] = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, md, cv::Point(-1, -1), 3);
		_filter[2] = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, mc, cv::Point(-1, -1), 3);
		_filter[3] = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, mo, cv::Point(-1, -1), 3);
#endif
		return solids::lib::video::nvidia::player::detector::err_code_t::success;
	}

	int32_t detector::core::release(void)
	{
#if defined(WITH_HOG)
		_hog.release();
#elif defined(WITH_HARR)


#elif defined(WITH_MOG)


#else

#endif
		return solids::lib::video::nvidia::player::detector::err_code_t::success;
	}

	int32_t detector::core::detect(uint8_t* input, int32_t inputStride, uint8_t** output, int32_t& outputStride)
	{
#if defined(WITH_HOG)
		cv::cuda::GpuMat frame(_ctx->height, _ctx->width, CV_8UC4, input, inputStride);
		cv::cuda::GpuMat gray(_ctx->height, _ctx->width, CV_8UC1);
		cv::cuda::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);

		std::vector<cv::Rect> found;

		_hog->setScaleFactor(1.01f);
		_hog->setNumLevels(_hog_levels);
		//_hog->setNumLevels(7);
		_hog->setWinStride(_win_stride);
		_hog->setGroupThreshold(_hog_group_thredshold);
		_hog->detectMultiScale(gray, found);
		//for (int32_t i = 0; i < (int32_t)found.size(); i++)
		//{
		//	cv::Rect r = found[i];
		//	cv::rectangle(img, r, cv::Scalar(255, 0, 0), 1);
		//	
		//}
		*output = frame.data;
		outputStride = (int32_t)frame.step;
#elif defined(WITH_HARR)


#elif defined(WITH_MOG)
#if defined(WITH_LABELLING)
		cv::cuda::GpuMat frame(_ctx->height, _ctx->width, CV_8UC4, input, inputStride);

		cv::cuda::GpuMat frame1(_ctx->height, _ctx->width, CV_8UC1);
		cv::cuda::GpuMat frame2(_ctx->height, _ctx->width, CV_8UC1);
		cv::cuda::cvtColor(frame, frame1, cv::COLOR_BGRA2GRAY);
		//cv::cuda::threshold(frame2, frame2, 1, 255, cv::THRESH_BINARY_INV);
		_filter[0]->apply(frame1, frame2);
		if (_first_frame)
		{
			_mog->apply(frame2, _foregroundmask, -1);
			//_mog2->apply(frame2, _foregroundmask);
			_first_frame = FALSE;
		}
		else
		{
			_mog->apply(frame2, _foregroundmask, -1);
			_mog->getBackgroundImage(_backgroundimg);
			//_mog2->apply(frame2, _foregroundmask);
			//_mog2->getBackgroundImage(_backgroundimg);

			_foregroundimg.create(frame2.size(), frame2.type());
			_foregroundimg.setTo(cv::Scalar::all(0));
			frame2.copyTo(_foregroundimg, _foregroundmask);

			_filter[1]->apply(_foregroundimg, frame2);
			_filter[2]->apply(frame2, frame2);
			//_filter[3]->apply(frame2, frame2);
			//_filter[4]->apply(frame2, frame2);
			cv::cuda::threshold(frame2, frame2, 1, 255, cv::THRESH_BINARY);

			//labelling
			{
				int32_t bufferSize;
				NppiSize sourceROI = { frame2.cols, frame2.rows };
				nppiLabelMarkersGetBufferSize_8u_C1R(sourceROI, &bufferSize);
				Npp8u* buffer = NULL;
				cudaMalloc((void**)&buffer, bufferSize);

				int32_t max;
				::nppiLabelMarkers_8u_C1IR(frame2.data, frame2.step, sourceROI, (Npp8u)1, NppiNorm::nppiNormInf, &max, buffer);
				int32_t bs;
				::nppiCompressMarkerLabelsGetBufferSize_8u_C1R(1, &bs);
				if (bs > bufferSize)
				{
					bufferSize = bs;
					::cudaFree(buffer);
					::cudaMalloc(&buffer, bufferSize);
				}
				::nppiCompressMarkerLabels_8u_C1IR(frame2.data, frame2.step, sourceROI, 200, &max, buffer);
				cv::cuda::multiply(frame2, cv::Scalar::all(20.0), frame2);

				cv::cuda::cvtColor(frame2, frame, cv::COLOR_GRAY2BGRA);

				::cudaFree(buffer);
			}


			//cv::cuda::cvtColor(frame2, frame, cv::COLOR_GRAY2BGRA);

			*output = frame.data;
			outputStride = (int32_t)frame.step;
		}
#else
		cv::cuda::GpuMat frame(_ctx->height, _ctx->width, CV_8UC4, input, inputStride);

		cv::cuda::GpuMat frame1(_ctx->height, _ctx->width, CV_8UC1);
		cv::cuda::GpuMat frame2(_ctx->height, _ctx->width, CV_8UC1);
		cv::cuda::cvtColor(frame, frame1, cv::COLOR_BGRA2GRAY);
		_filter[0]->apply(frame1, frame2);
		if (_first_frame)
		{
			_mog->apply(frame2, _foregroundmask, -1);
			_first_frame = FALSE;
		}
		else
		{
			_mog->apply(frame2, _foregroundmask, -1);
			_mog->getBackgroundImage(_backgroundimg);

			_foregroundimg.create(frame2.size(), frame2.type());
			_foregroundimg.setTo(cv::Scalar::all(0));
			frame2.copyTo(_foregroundimg, _foregroundmask);

			_filter[1]->apply(_foregroundimg, frame2);
			_filter[2]->apply(frame2, frame2);
			cv::cuda::threshold(frame2, frame2, 1, 255, cv::THRESH_BINARY);


			std::vector<cv::Vec4i> hierarchy;
			std::vector<std::vector<cv::Point> > contours;
			cv::Mat frame3(_ctx->height, _ctx->width, CV_8UC1);
			cv::Mat frame4(_ctx->height, _ctx->width, CV_8UC1, cv::Scalar(0, 0, 0));
			frame2.download(frame3);
			cv::findContours(frame3, contours, hierarchy, cv::RETR_LIST, 2);

			std::vector<cv::Point> result;
			std::vector<cv::Point> pts;
			for (size_t i = 0; i < contours.size(); i++)
				for (size_t j = 0; j < contours[i].size(); j++)
					pts.push_back(contours[i][j]);
			cv::convexHull(pts, result);

			/*
			// approximate contours
			std::vector<std::vector<cv::Point> > contours_poly(contours.size());
			for (int32_t i = 0; i < contours.size(); i++)
			{
				cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 10, true);
			}
			*/
			/*
			std::vector<cv::Point> merged_contour_points;
			for (int32_t i = 0; i < contours_poly.size(); i++)
			{
				for (int32_t j = 0; j < contours_poly[i].size(); j++)
				{
					merged_contour_points.push_back(contours_poly[i][j]);
				}
			}

			std::vector<cv::Point> hull;
			cv::convexHull(cv::Mat(merged_contour_points), hull);
			cv::Mat hull_points(hull);
			cv::RotatedRect rotated_bounding_rect = minAreaRect(hull_points);
			cv::Point2f vertices2f[4];
			rotated_bounding_rect.points(vertices2f);
			cv::Point vertices[4];
			for (int i = 0; i < 4; ++i)
			{
				vertices[i] = vertices2f[i];
			}
			cv::fillConvexPoly(frame4, vertices, 4, cv::Scalar(255.0, 255.0, 255.0));
			*/
			cv::RNG rng(12345);
			for (int32_t x = 0; x < contours.size(); x++)
			{
				cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				cv::drawContours(frame4, contours, x, color, 2, 8, hierarchy, 0, cv::Point());
			}

			frame2.upload(frame4);
			//cv::polylines(frame4, pts, true, cv::Scalar(0, 0, 255));
			frame2.upload(frame4);
			cv::cuda::cvtColor(frame2, frame, cv::COLOR_GRAY2BGRA);

			*output = frame.data;
			outputStride = (int32_t)frame.step;
		}
#endif
#elif defined(WITH_GSOC)
		cv::cuda::GpuMat frame(_ctx->height, _ctx->width, CV_8UC4, input, inputStride);
		cv::cuda::GpuMat gpuInput(_ctx->height, _ctx->width, CV_8UC1);
		cv::cuda::GpuMat gpuOutput(_ctx->height, _ctx->width, CV_8UC1);
		cv::Mat cpuInput(_ctx->height, _ctx->width, CV_8UC1);
		cv::Mat cpuOutput(_ctx->height, _ctx->width, CV_8UC1);


		cv::cuda::cvtColor(frame, gpuInput, cv::COLOR_BGRA2GRAY);
		gpuInput.download(cpuInput);

		_gsoc->apply(cpuInput, cpuOutput);

		gpuOutput.upload(cpuOutput);

		cv::cuda::cvtColor(gpuOutput, _target, cv::COLOR_GRAY2BGRA);

		*output = _target.data;
		outputStride = (int32_t)_target.step;

#else
		cv::cuda::GpuMat frame(_ctx->height, _ctx->width, CV_8UC4, input, inputStride);
		cv::cuda::GpuMat intermediate1(_ctx->height, _ctx->width, CV_8UC1);
		cv::cuda::GpuMat intermediate2(_ctx->height, _ctx->width, CV_8UC1);
		cv::cuda::GpuMat intermediate3(_ctx->height, _ctx->width, CV_8UC1);
		cv::cuda::cvtColor(frame, intermediate1, cv::COLOR_BGRA2GRAY);
		_backgroundimg.copyTo(intermediate2);

		//_filter[0]->apply(intermediate1, intermediate1);
		//_filter->apply(intermediate2, intermediate2);
		cv::cuda::subtract(intermediate1, intermediate2, intermediate3);

		cv::cuda::threshold(intermediate3, intermediate3, 130, 255, cv::THRESH_BINARY);
		//
		_filter[3]->apply(intermediate3, intermediate3);
		_filter[1]->apply(intermediate3, intermediate3);
		_filter[2]->apply(intermediate3, intermediate3);

		cv::cuda::cvtColor(intermediate3, _target, cv::COLOR_GRAY2BGRA);
		*output = _target.data;
		outputStride = (int32_t)_target.step;
#endif

		return solids::lib::video::nvidia::player::detector::err_code_t::success;
	}

};
};
};
};
};