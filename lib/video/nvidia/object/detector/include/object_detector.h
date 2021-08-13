#pragma once

#include "sld_object_detector.h"
#include "resize.h"

#include "NvInfer.h"
#include <cuda_runtime.h>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <iostream>
#include <fstream>
#include <assert.h>

// TODO: uchar instead type;
typedef unsigned char uchar;

// TODO: Object Detect Util Function
//static cv::Scalar obj_id_to_color(int obj_id) {
//	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
//	int const offset = obj_id * 123457 % 6;
//	int const color_scale = 150 + (obj_id * 123457) % 100;
//	cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
//	color *= color_scale;
//	return color;
//}


// TODO: TensorRT Common Function
inline void* safeCudaMalloc(size_t memSize)
{
	void* deviceMem;
	cudaMalloc(&deviceMem, memSize);
	if (deviceMem == nullptr)
	{
		std::cerr << "Out of memory" << std::endl;
		exit(1);
	}
	return deviceMem;
}

// TODO: TensorRT Common Structure
struct InferDeleter
{
	template <typename T>
	void operator()(T* obj) const
	{
		if (obj)
		{
			obj->destroy();
		}
	}
};

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
					typedef struct _infer_deleter_t
					{
						template<typename T>
						void operator()(T* obj) const
						{
							if (obj)
								obj->destroy();
						}
					}infer_deleter_t;

					typedef struct _network_params_t
					{
						int32_t batch_size{ 1 };
						int32_t dla_core{ -1 };
						bool	int8{ false };
						bool	fp16{ false };
						
						// TODO: 향후 network 관련 parameter 넣어주기...
						// sample 
						// std::vector<std::string> data_dirs;
						// std::vector<std::string> input_tensor_names;
						// std::vector<std::string> output_tensor_names;
					}network_params_t;

					typedef struct _onnx_params_t
					{
					std:: string onnx_filename;
					}onnx_params;

					template<typename T>
					using rtunique_ptr = std::unique_ptr<T, infer_deleter_t>;

					class Logger : public nvinfer1::ILogger
					{
					public:

						void log(Severity severity, const char* msg) override
						{
							if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR))
							{
								std::cerr << msg << std::endl;
							}
						}
					};

					class detector::core
						// TODO: TensorRT 관련 class 상속받기
					{
					template <typename T>
					using UniquePtr = std::unique_ptr<T, InferDeleter>;
					public:
						core(solids::lib::video::nvidia::object::detector* front);
						~core(void);

						int32_t initialize(solids::lib::video::nvidia::object::detector::context_t* ctx);
						int32_t release(void);

						int32_t detect(uint8_t* input, int32_t inputStride, uint8_t** output, int32_t& outputStride);

					private:
						solids::lib::video::nvidia::object::detector* _front;
						solids::lib::video::nvidia::object::detector::context_t* _ctx;
						
						solids::lib::video::nvidia::object::_onnx_params_t _params;

					template <typename T>
					using UniquePtr = std::unique_ptr<T, InferDeleter>;

					public:
						int batchSize{-1};
						int numClasses{-1};
						int numChannels{-1};
						int inputHeightSize{-1};
						int inputWidthSize{-1};

						std::vector<float> cpuBboxBuffer;
						std::vector<float> cpuConfidenceBuffer;

						std::vector<nvinfer1::Dims> inputDims;
						std::vector<nvinfer1::Dims> outputDims;

						// TODO: Yolov4 Class 만들자...
						// Yolov4 m_yolov4;

					private:
						std::size_t getSizeByDim(const nvinfer1::Dims& dims);

						UniquePtr<nvinfer1::ICudaEngine> engine;
						UniquePtr<nvinfer1::IExecutionContext> context;
						std::vector<void*> cudaBuffers;
						void* cudaFrame{NULL};
						void initEngine();
						Logger gLogger;
						cudaStream_t cudaStream;
						
						float confThreshold = 0.4f;
						float nmsThreshold = 0.1f;
					};
					
				}
			}
		}
	}
}