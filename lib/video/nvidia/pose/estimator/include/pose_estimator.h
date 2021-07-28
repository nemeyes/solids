#pragma once

#include "sld_pose_estimator.h"
#include "resize.h"
#include "TensorrtPoseNet.h"
#include "Openpose.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <opencv2/cudaimgproc.hpp>

#include <nppi_filtering_functions.h>

//struct InferDeleter
//{
//	template <typename T>
//	void operator()(T* obj) const
//	{
//		if (obj)
//		{
//			obj->destroy();
//		}
//	}
//};

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
					typedef struct _infer_deleter_t
					{
						template<typename T>
						void operator()(T* obj) const
						{
							if (obj)
								obj->destroy();
						}
					} infer_deleter_t;

					typedef struct _network_params_t
					{
						int32_t batch_size{ 1 };
						int32_t dla_core{ -1 };
						bool	int8{ false };
						bool	fp16{ false };
						std::vector<std::string> data_dirs;
						std::vector<std::string> input_tensor_names;
						std::vector<std::string> output_tensor_names;
					} network_params_t;

					typedef struct _onnx_params_t
					{
						std::string onnx_filename;
					} onnx_params_t;

					template<typename T>
					using rtunique_ptr = std::unique_ptr<T, infer_deleter_t>;



					class estimator::core
					{
					template <typename T>
					using UniquePtr = std::unique_ptr<T, InferDeleter>;
					public:
						core(solids::lib::video::nvidia::pose::estimator* front);
						~core(void);

						int32_t initialize(solids::lib::video::nvidia::pose::estimator::context_t* ctx);
						int32_t release(void);

						int32_t estimate(uint8_t* input, int32_t inputStride, uint8_t** output, int32_t& outputStride);


						//std::vector<nvinfer1::Dims> inputDims;
						//std::vector<nvinfer1::Dims> outputDims;
						//std::vector<float> cpuCmapBuffer;
						//std::vector<float> cpuPafBuffer;
					private:
						solids::lib::video::nvidia::pose::estimator* _front;
						solids::lib::video::nvidia::pose::estimator::context_t* _ctx;
						
						solids::lib::video::nvidia::pose::onnx_params_t _params;


						//std::size_t getSizeByDim(const nvinfer1::Dims& dims);
						//std::vector<void*> cudaBuffers;
						//UniquePtr<nvinfer1::IExecutionContext> context;
						//cudaStream_t cudaStream;

					template <typename T>
					using UniquePtr = std::unique_ptr<T, InferDeleter>;

					public:
						// The dimensions of the input and output to the network

						int batchSize;
						int numClasses;
						int numChannels;
						int inputHeightSize;
						int inputWidthSize;

						std::vector<float> cpuCmapBuffer;
						std::vector<float> cpuPafBuffer;

						std::vector<nvinfer1::Dims> inputDims;
						std::vector<nvinfer1::Dims> outputDims;

						Openpose m_openpose;
					private:
						std::size_t getSizeByDim(const nvinfer1::Dims& dims);


						UniquePtr<nvinfer1::ICudaEngine> engine;
						UniquePtr<nvinfer1::IExecutionContext> context;
						cudaStream_t cudaStream;

						std::vector<void*> cudaBuffers;
						void* cudaFrame;
						void initEngine();
						float confThreshold = 0.4f;
						float nmsThreshold = 0.4f;
					};
				};
			};
		};
	};
};
