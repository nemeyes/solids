#include "pose_estimator.h"


#define MAX_WORKSPACE (1 << 30) // 1G workspace memory
//#define _Only_Pose_Esimation
static Logger gLogger;

namespace solids
{
namespace lib
{
namespace video
{
namespace nvidia
{
namespace pose
{

	estimator::core::core(solids::lib::video::nvidia::pose::estimator* front)
		: _front(front)
		, _ctx(NULL)
	{
	}

	estimator::core::~core(void)
	{
	}

	int32_t estimator::core::initialize(solids::lib::video::nvidia::pose::estimator::context_t* ctx)
	{
		_ctx = ctx;
		std::cout << "Loading OpenPose Inference Engine ... " << std::endl;


		cudaSetDevice(0);
		std::fstream file;

		confThreshold = 0.3f;
		nmsThreshold = 0.3f;
		std::string engineFilePath = "trt_pose_fp16.engine";
		//std::string engineFilePath = "pose_higher_hrnet_w32_512.engine";
		file.open(engineFilePath, std::ios::binary | std::ios::in);
		if (!file.is_open())
		{
			std::cout << "read engine file: " << engineFilePath << " failed" << std::endl;
			return 0;
		}
		file.seekg(0, std::ios::end);
		int length = file.tellg();
		file.seekg(0, std::ios::beg);
		std::unique_ptr<char[]> data(new char[length]);
		file.read(data.get(), length);

		file.close();

		auto runtime = UniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
		assert(runtime != nullptr);
		
		engine = UniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(data.get(), length, nullptr));
		assert(engine != nullptr);

		std::cout << "Done" << std::endl;

		context = UniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
		assert(context);

		const int numBindingPerProfile = engine->getNbBindings() / engine->getNbOptimizationProfiles();
		std::cout << "Number of binding profiles: " << numBindingPerProfile << std::endl;

		initEngine();
		Openpose openpose(outputDims[0]);
		m_openpose = openpose;
		return solids::lib::video::nvidia::pose::estimator::err_code_t::success;
	}

	int32_t estimator::core::release(void)
	{
		return solids::lib::video::nvidia::pose::estimator::err_code_t::success;
	}

	int32_t estimator::core::estimate(uint8_t* input, int32_t inputStride, uint8_t* srcBBox, int32_t bboxSize, uint8_t** output, int32_t& outputStride)
	{
		cv::cuda::GpuMat img = cv::cuda::GpuMat(_ctx->height, _ctx->width, CV_8UC4, input, inputStride);
		cv::cuda::GpuMat img2, img3;
		
		std::vector<cv::Rect> bboxes;
		if(bboxSize == 0)
			return solids::lib::video::nvidia::pose::estimator::err_code_t::success;

		// TODO: srcBBox 해제해주기
		
#ifdef _Only_Pose_Esimation
		// TODO: GPUMat Crop Detected Image
		cv::Mat mImg1;
		//cv::cuda::cvtColor(img, img2, cv::COLOR_RGBA2RGB);
		cv::cuda::cvtColor(img, img2, cv::COLOR_BGRA2RGB);
		
		std::chrono::system_clock::time_point st = std::chrono::system_clock::now();
		img2.download(mImg1);
	
		int elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - st).count();
		std::cout << "download elapsed Time : " << elapsedTime << std::endl;
		
		st = std::chrono::system_clock::now();
		//-------------------------------------------------------------
		cudaMemcpy((void*)img2.ptr(), mImg1.data, mImg1.step[0] * mImg1.rows, cudaMemcpyHostToDevice);
		resizeAndNorm((void*)img2.ptr(), (float*)cudaBuffers[0], _ctx->width, _ctx->height, inputWidthSize, inputHeightSize, cudaStream);

		context->enqueue(1, cudaBuffers.data(), cudaStream, nullptr);

		cudaMemcpy(cpuCmapBuffer.data(), (float*)cudaBuffers[1], cpuCmapBuffer.size() * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuPafBuffer.data(), (float*)cudaBuffers[2], cpuPafBuffer.size() * sizeof(float), cudaMemcpyDeviceToHost);
		//--------------------------------------------------------------
		//cudaMemcpyAsync(cpuCmapBuffer.data(), (float*)cudaBuffers[1], cpuCmapBuffer.size() * sizeof(float), cudaMemcpyDeviceToHost, cudaStream);
		//cudaMemcpyAsync(cpuPafBuffer.data(), (float*)cudaBuffers[2], cpuPafBuffer.size() * sizeof(float), cudaMemcpyDeviceToHost, cudaStream);
		//cudaStreamSynchronize(cudaStream);
		//--------------------------------------------------------------
		elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - st).count();
		std::cout << "cudamemcpy & inference elapsed Time : " << elapsedTime << std::endl;

		st = std::chrono::system_clock::now();
		m_openpose.detect(cpuCmapBuffer, cpuPafBuffer, mImg1);
		elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - st).count();
		std::cout << "detect elapsed Time : " << elapsedTime << std::endl;

		st = std::chrono::system_clock::now();
		img2.upload(mImg1);

		elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - st).count();
		std::cout << "upload elapsed Time : " << elapsedTime << std::endl;

		//cv::cuda::cvtColor(img2, img, cv::COLOR_RGB2RGBA);
		cv::cuda::cvtColor(img2, img, cv::COLOR_RGB2BGRA);
		
#else
	for (int i = 0; i < bboxSize / sizeof(cv::Rect); ++i)
	{
		// get bbox
		cv::Rect tmpRect;
		::memmove(&tmpRect, srcBBox + (i * sizeof(cv::Rect)), sizeof(cv::Rect));
		// TODO: BBox 예외처리해주기... 이거는 Object Detection에서 해주기
		// 1) image size 넘치는경우
		// 2) 좌표의 값이 -가 되는 경우
		// 3) Person의 BBox만 보내주자
		if (tmpRect.x < 0) tmpRect.x = 0;
		if (tmpRect.y < 0) tmpRect.y = 0;
		tmpRect.width = (tmpRect.width <= _ctx->width) ? tmpRect.width : _ctx->width;
		tmpRect.height = (tmpRect.height <= _ctx->height) ? tmpRect.height : _ctx->height;
		cv::Rect roi{ tmpRect.x, tmpRect.y, tmpRect.width - tmpRect.x, tmpRect.height - tmpRect.y };

		// TODO: GPUMat Crop Detected Image
		cv::Mat mImg, mImg1, mImg1_2, mImg2, mImg3;
		//cv::cuda::cvtColor(img, img2, cv::COLOR_RGBA2RGB);
		cv::cuda::cvtColor(img, img2, cv::COLOR_BGRA2RGB);

		std::chrono::system_clock::time_point st = std::chrono::system_clock::now();
		img2.download(mImg1);
		cv::cuda::GpuMat cropimg3 = (cv::cuda::GpuMat(img2, roi)).clone();
		cropimg3.download(mImg2);
		//mImg1.copyTo(mImg2);
		//mImg2 = mImg2(roi);
		//img3.upload(mImg2);
		int elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - st).count();
		std::cout << "download elapsed Time : " << elapsedTime << std::endl;

		st = std::chrono::system_clock::now();
		//-------------------------------------------------------------
		//cudaMemcpy((void*)img2.ptr(), mImg1.data, mImg1.step[0] * mImg1.rows, cudaMemcpyHostToDevice);
		//resizeAndNorm((void*)img2.ptr(), (float*)cudaBuffers[0], _ctx->width, _ctx->height, inputWidthSize, inputHeightSize, cudaStream);
		cudaMemcpy((void*)cropimg3.ptr(), mImg2.data, mImg2.step[0] * mImg2.rows, cudaMemcpyHostToDevice);
		resizeAndNorm((void*)cropimg3.ptr(), (float*)cudaBuffers[0], cropimg3.cols, cropimg3.rows, inputWidthSize, inputHeightSize, cudaStream);

		context->enqueue(1, cudaBuffers.data(), cudaStream, nullptr);

		cudaMemcpy(cpuCmapBuffer.data(), (float*)cudaBuffers[1], cpuCmapBuffer.size() * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuPafBuffer.data(), (float*)cudaBuffers[2], cpuPafBuffer.size() * sizeof(float), cudaMemcpyDeviceToHost);
		//--------------------------------------------------------------
		//cudaMemcpyAsync(cpuCmapBuffer.data(), (float*)cudaBuffers[1], cpuCmapBuffer.size() * sizeof(float), cudaMemcpyDeviceToHost, cudaStream);
		//cudaMemcpyAsync(cpuPafBuffer.data(), (float*)cudaBuffers[2], cpuPafBuffer.size() * sizeof(float), cudaMemcpyDeviceToHost, cudaStream);
		//cudaStreamSynchronize(cudaStream);
		//--------------------------------------------------------------
		elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - st).count();
		std::cout << "cudamemcpy & inference elapsed Time : " << elapsedTime << std::endl;

		st = std::chrono::system_clock::now();
		m_openpose.detect(cpuCmapBuffer, cpuPafBuffer, mImg2);
		elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - st).count();
		std::cout << "detect elapsed Time : " << elapsedTime << std::endl;

		st = std::chrono::system_clock::now();
		// crop image patch to original image
		mImg2.copyTo(mImg1(roi));
		img2.upload(mImg1);

		elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - st).count();
		std::cout << "upload elapsed Time : " << elapsedTime << std::endl;
		//cv::cuda::cvtColor(img2, img, cv::COLOR_RGB2RGBA);
		cv::cuda::cvtColor(img2, img, cv::COLOR_RGB2BGRA);

	}
#endif

		//-----------detect inference----------------------
		*output = (uint8_t*)img.ptr();

		//--------------------------------------------------------------
		// Debug
		//img.download(mImg2); // Debug
		//cv::cuda::GpuMat imgOut = cv::cuda::GpuMat(_ctx->height, _ctx->width, CV_8UC4, *output, inputStride);
		//imgOut.download(mImg3);
		//--------------------------------------------------------------

		outputStride = inputStride;
		return solids::lib::video::nvidia::pose::estimator::err_code_t::success;
	}

	std::size_t estimator::core::getSizeByDim(const nvinfer1::Dims& dims)
	{
		std::size_t size = 1;
		for (std::size_t i = 0; i < dims.nbDims; ++i)
		{
			size *= dims.d[i];
		}
		return size;
	}

	void estimator::core::initEngine()
	{
		cudaBuffers.resize(engine->getNbBindings());
		for (size_t i = 0; i < engine->getNbBindings(); ++i)
		{
			auto bindingSize = getSizeByDim(engine->getBindingDimensions(i)) * 1 * sizeof(float);
			cudaMalloc(&cudaBuffers[i], bindingSize);
			if (engine->bindingIsInput(i))
			{
				inputDims.emplace_back(engine->getBindingDimensions(i));
			}
			else
			{
				outputDims.emplace_back(engine->getBindingDimensions(i));
			}
			std::cout << "Binding Name: " << engine->getBindingName(i) << std::endl;
		}
		if (inputDims.empty() || outputDims.empty())
		{
			std::cerr << "Expect at least one input and one output for network";
		}

		batchSize = inputDims[0].d[0];
		numChannels = inputDims[0].d[1];
		inputHeightSize = inputDims[0].d[2];
		inputWidthSize = inputDims[0].d[3];

		std::cout << "output[0] -> cmap" << std::endl;
		std::cout << "outputDims[0]: " << std::endl;
		std::size_t size = 1;
		for (std::size_t i = 0; i < outputDims[0].nbDims; ++i)
		{
			std::cout << "out[0]: " << outputDims[0].d[i] << std::endl;
			size *= outputDims[0].d[i];
		}
		std::cout << "out[0].size: " << size << std::endl;

		std::cout << "output[1] -> paf" << std::endl;
		std::cout << "outputDims[1]: " << std::endl;
		size = 1;
		for (std::size_t i = 0; i < outputDims[1].nbDims; ++i)
		{
			std::cout << "out[1]: " << outputDims[1].d[i] << std::endl;
			size *= outputDims[1].d[i];
		}
		std::cout << "out[1].size: " << size << std::endl;

		cpuCmapBuffer.resize(getSizeByDim(outputDims[0]) * batchSize);
		cpuPafBuffer.resize(getSizeByDim(outputDims[1]) * batchSize);

		std::cout << "Model input shape: " <<
			batchSize << "x" <<
			numChannels << "x" <<
			inputWidthSize << "x" <<
			inputHeightSize << std::endl;

		cudaFrame = safeCudaMalloc(4096 * 4096 * 3 * sizeof(uchar)); // max input image shape

		cudaStreamCreate(&cudaStream);
	}

};
};
};
};
};

