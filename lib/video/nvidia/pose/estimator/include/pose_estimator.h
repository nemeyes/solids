#pragma once

#include "sld_pose_estimator.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>


#include <nppi_filtering_functions.h>

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
					public:
						core(solids::lib::video::nvidia::pose::estimator* front);
						~core(void);

						int32_t initialize(solids::lib::video::nvidia::pose::estimator::context_t* ctx);
						int32_t release(void);

						int32_t estimate(uint8_t* input, int32_t inputStride, uint8_t** output, int32_t& outputStride);

					private:
						solids::lib::video::nvidia::pose::estimator* _front;
						solids::lib::video::nvidia::pose::estimator::context_t* _ctx;

						solids::lib::video::nvidia::pose::onnx_params_t _params;

					};
				};
			};
		};
	};
};
