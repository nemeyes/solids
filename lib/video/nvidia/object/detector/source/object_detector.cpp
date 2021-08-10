#include "object_detector.h"

#define MAX_WORKSPACE (1 << 30) // 1G workspace memory


namespace solids
{
namespace lib
{
namespace video
{
namespace nvidia
{
namespace object
{
	detector::core::core(solids::lib::video::nvidia::object::detector* front)
		: _front(front)
		, _ctx(NULL)
	{
	}

	detector::core::~core(void)
	{
	}

	void detector::core::initEngine()
	{
		// TODO: Functionalization (ex. input & output Dimension, etc...)
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

		// TODO: dynamic batchsize...
		batchSize = inputDims[0].d[0];
		numChannels = inputDims[0].d[1];
		inputHeightSize = inputDims[0].d[2];
		inputWidthSize = inputDims[0].d[3];

		// TODO: Debug Mode print info
		std::cout << "output[0] -> Objectness Score & Box Cordinate" << std::endl;
		std::cout << "outputDims[0]: " << std::endl;
		std::size_t size = 1;
		for (std::size_t i = 0; i < outputDims[0].nbDims; ++i)
		{
			std::cout << "out[0]: " << outputDims[0].d[i] << std::endl;
			size *= outputDims[0].d[i];
		}
		std::cout << "out[0].size: " << size << std::endl;

		std::cout << "output[1] -> Class Confidence" << std::endl;
		std::cout << "outputDims[1]: " << std::endl;
		size = 1;
		for (std::size_t i = 0; i < outputDims[1].nbDims; ++i)
		{
			std::cout << "out[1]: " << outputDims[1].d[i] << std::endl;
			size *= outputDims[1].d[i];
		}
		std::cout << "out[1].size: " << size << std::endl;

		cpuBboxBuffer.resize(getSizeByDim(outputDims[0]) * batchSize);
		cpuConfidenceBuffer.resize(getSizeByDim(outputDims[1]) * batchSize);

		std::cout << "Model input shape: " <<
			batchSize << "x" <<
			numChannels << "x" <<
			inputWidthSize << "x" <<
			inputHeightSize << std::endl;
		
		cudaFrame = safeCudaMalloc(4096 * 4096 * 3 * sizeof(uchar)); // max input image shape

		cudaStreamCreate(&cudaStream);
	}

	int32_t detector::core::initialize(solids::lib::video::nvidia::object::detector::context_t* ctx)
	{
		_ctx = ctx; 
		std::cout << "Loading Object Detecte Inference Eninge..." << std::endl;

		// Select GPU Device Num
		cudaSetDevice(0);

		// TODO: File seek function
		std::string engineFilePath = "D:\\Download\\TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.0.cudnn7.6\\TensorRT-7.0.0.11\\bin\\yolov4_1_3_288_288_static_Upsample1fp16.engine";
		std::fstream file;
		file.open(engineFilePath, std::ios::binary | std::ios::in);
		if (!file.is_open())
		{
			std::cout << "read engine file : " << engineFilePath << " failed" << std::endl;
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

		std::cout << "Engine Initialize Done" << std::endl;

		context = UniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
		assert(context);

		const int numBindingPerProfile = engine->getNbBindings() / engine->getNbOptimizationProfiles();
		std::cout << "Number of binding profiles: " << numBindingPerProfile << std::endl;

		initEngine();
		return 0;
	}

	int32_t detector::core::release(void)
	{
		return solids::lib::video::nvidia::object::detector::err_code_t::success;
	}

	int32_t detector::core::detect(uint8_t* input, int32_t inputStride, uint8_t** output, int32_t& outputStride)
	{
		cv::cuda::GpuMat srcGpuImg = cv::cuda::GpuMat(_ctx->height, _ctx->width, CV_8UC4, input, inputStride);
		cv::Mat rgbImg;
		cv::cuda::GpuMat rgbGpuImg;
		int32_t classesNum = 80;
		cv::Size frameSize = srcGpuImg.size();
		cv::cuda::cvtColor(srcGpuImg, srcGpuImg, cv::COLOR_BGRA2BGR);
		srcGpuImg.download(rgbImg);
		// Pre-Processing
#ifdef __host_memory
		srcGpuImg.download(rgbImg);
		cv::cvtColor(rgbImg, rgbImg, cv::COLOR_BGRA2RGB);
		cv::resize(rgbImg, rgbImg, cv::Size(inputWidthSize, inputHeightSize));
		rgbGpuImg.convertTo(rgbGpuImg, CV_32FC3, 1.f / 255.f);
		std::vector<float>data;
		std::vector<cv::cuda::GpuMat>channeles(3);
		cv::split(rgbImg, channeles);
		float* ptr1 = (float*)(channeles[0].data);
		float* ptr2 = (float*)(channeles[1].data);
		float* ptr3 = (float*)(channeles[2].data);
		data.insert(data.end(), ptr3, ptr3 + inputWidthSize * inputHeightSize);
		data.insert(data.end(), ptr1, ptr1 + inputWidthSize * inputHeightSize);
		data.insert(data.end(), ptr2, ptr2 + inputWidthSize * inputHeightSize);
		cudaMemcpy(cudaBuffers[0], data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
#else
		// GPU Operation
		
		cudaMemcpy((void*)srcGpuImg.ptr(), rgbImg.data, rgbImg.step[0] * rgbImg.rows, cudaMemcpyHostToDevice);
		resizeAndNorm((void*)srcGpuImg.ptr(), (float*)cudaBuffers[0], _ctx->width, _ctx->height, inputWidthSize, inputHeightSize, cudaStream);
#endif
		// Inference
		context->enqueue(1, cudaBuffers.data(), cudaStream, nullptr);

		// Inference result device to host
		cudaMemcpy(cpuBboxBuffer.data(), (float*)cudaBuffers[1], cpuBboxBuffer.size() * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuConfidenceBuffer.data(), (float*)cudaBuffers[2], cpuConfidenceBuffer.size() * sizeof(float), cudaMemcpyDeviceToHost);

		// Post-Processing
		int32_t stIdx = 0;
		std::vector<cv::Rect>* bboxes = new std::vector<cv::Rect>;
		std::vector<float> scores;
		std::vector<int> classes;

		// TODO: NMS 관련 코드 수정하기(중첩되는 부분이 정상적으로처리가 안되는 부분이 존재... NMSThreshold값을 낮추기...)
		// TODO: Functionalization [PostProcessing]
		for (int32_t i = 0; i < cpuConfidenceBuffer.size() / classesNum; ++i)
		{
			std::vector<float> tmpVector;

			// Step1 : Score Extract and get max score index(=confidence idx)
			int32_t endIdx = stIdx + classesNum;
			tmpVector = std::vector<float>(cpuConfidenceBuffer.begin() + stIdx, cpuConfidenceBuffer.begin() + endIdx);

			// Get Max Score & Index
			auto result = std::max_element(tmpVector.begin(), tmpVector.end());
			int32_t targetIdx = std::distance(tmpVector.begin(), result);// +stIdx;
			int label = targetIdx;
			targetIdx += stIdx;

			// Step2 : Get Confidence index
			float confidence = cpuConfidenceBuffer[targetIdx];
			stIdx = endIdx;

			// Step3 : Confidence Threshold compare and make box info
			if (confidence < confThreshold)
				continue;

			// Find target coordinate and Get target index
			int32_t targetBoxStartIdx = (targetIdx / 80) * 4;

			// Get Boundary Box Info
			int32_t x = int(cpuBboxBuffer[targetBoxStartIdx] * frameSize.width);      // Stary X
			int32_t y = int(cpuBboxBuffer[targetBoxStartIdx + 1] * frameSize.height); // Start Y
			int32_t w = int(cpuBboxBuffer[targetBoxStartIdx + 2] * frameSize.width);  // End X
			int32_t h = int(cpuBboxBuffer[targetBoxStartIdx + 3] * frameSize.height); // End Y
			//int32_t x = int(centerx - w / 2);
			//int32_t y = int(centery - h / 2);
			cv::Rect bbox = { x, y, w, h };
			bboxes->push_back(bbox);
			scores.push_back(confidence);
			classes.push_back(label);
		}

		// Step4 : Calculate NMS with NMSBoxes(opencv cv::dnn::NMSBoxes)
		std::vector<int32_t> indices;
		cv::dnn::NMSBoxes(*bboxes, scores, confThreshold, nmsThreshold, indices); // Remote to overlap box 

		// Step5 : draw BBoxes or Send to BBox from Pose Estimation
		*output = (uint8_t*)bboxes->data();
		outputStride = bboxes->size() * sizeof(cv::Rect);

		return solids::lib::video::nvidia::object::detector::err_code_t::success;
	}

	// TODO: TensorRT Common Function
	std::size_t detector::core::getSizeByDim(const nvinfer1::Dims& dims)
	{
		std::size_t size = 1;
		for (std::size_t i = 0; i < dims.nbDims; ++i)
		{
			size *= dims.d[i];
		}
		return size;
	}


}
}
}
}
}