#include <cuda_runtime.h>
#include "resizer.h"

namespace solids
{
namespace lib
{
namespace video
{
namespace nvidia
{
	template<class T>
	__device__ static T clamp(T x, T lower, T upper)
	{
		return x < lower ? lower : (x > upper ? upper : x);
	}

	template<typename yuv>
	static __global__ void resize(cudaTextureObject_t texY, cudaTextureObject_t texUV, unsigned char* dst, unsigned char* dstUV, int pitch, int width, int height, float fxScale, float fyScale)
	{
		int ix = blockIdx.x * blockDim.x + threadIdx.x,
			iy = blockIdx.y * blockDim.y + threadIdx.y;

		if (ix >= width / 2 || iy >= height / 2)
			return;

		int x = ix * 2, y = iy * 2;
		typedef decltype(yuv::x) YuvUnit;
		const int MAX = 1 << (sizeof(YuvUnit) * 8);

		yuv data;


		data.x = (YuvUnit)clamp((float)(tex2D<float>(texY, x / fxScale, y / fyScale) * MAX), 0.0f, 255.0f);
		data.y = (YuvUnit)clamp((float)(tex2D<float>(texY, (x + 1) / fxScale, y / fyScale) * MAX), 0.0f, 255.0f);
		*(yuv*)(dst + y * pitch + x * sizeof(YuvUnit)) = data;

		y++;

		data.x = (YuvUnit)clamp((float)(tex2D<float>(texY, x / fxScale, y / fyScale) * MAX), 0.0f, 255.0f);
		data.y = (YuvUnit)clamp((float)(tex2D<float>(texY, (x + 1) / fxScale, y / fyScale) * MAX), 0.0f, 255.0f);
		*(yuv*)(dst + y * pitch + x * sizeof(YuvUnit)) = data;


		float2 uv = tex2D<float2>(texUV, ix / fxScale, (height + iy) / fyScale + 0.5f);
		data.x = (YuvUnit)clamp((float)(uv.x * MAX), 0.0f, 255.0f);
		data.y = (YuvUnit)clamp((float)(uv.y * MAX), 0.0f, 255.0f);
		*(yuv*)(dstUV + iy * pitch + ix * 2 * sizeof(YuvUnit)) = data;
	}

	static void resize(unsigned char* dst, unsigned char* dstChroma, int dstPitch, int dstWidth, int dstHeight, unsigned char* src, int srcPitch, int srcWidth, int srcHeight)
	{
		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypePitch2D;
		resDesc.res.pitch2D.devPtr = src;
		resDesc.res.pitch2D.desc = cudaCreateChannelDesc<decltype(uchar2::x)>();
		resDesc.res.pitch2D.width = srcWidth;
		resDesc.res.pitch2D.height = srcHeight;
		resDesc.res.pitch2D.pitchInBytes = srcPitch;

		cudaTextureDesc texDesc = {};
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeNormalizedFloat;
		texDesc.normalizedCoords = 0;

		cudaTextureObject_t texY = 0;
		cudaCreateTextureObject(&texY, &resDesc, &texDesc, NULL);

		resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar2>();
		resDesc.res.pitch2D.width = srcWidth / 2;
		resDesc.res.pitch2D.height = srcHeight * 3 / 2;

		cudaTextureObject_t texUV = 0;
		cudaCreateTextureObject(&texUV, &resDesc, &texDesc, NULL);

		resize<uchar2> << <dim3((dstWidth + 31) / 32, (dstHeight + 31) / 32), dim3(16, 16) >> > (texY, texUV, dst, dstChroma, dstPitch, dstWidth, dstHeight, 1.0f * (float)dstWidth / (float)srcWidth, 1.0f * (float)dstHeight / (float)srcHeight);

		cudaDestroyTextureObject(texY);
		cudaDestroyTextureObject(texUV);
	}

	void decoder::resizer::resize_nv12(unsigned char* dstNV12, int dstNV12Pitch, int dstNV12Width, int dstNV12Height, unsigned char* srcNV12, int srcNV12Pitch, int srcNV12Width, int srcNV12Height)
	{
		unsigned char* dstNV12Chroma = dstNV12 + (dstNV12Pitch * dstNV12Height);
		return resize(dstNV12, dstNV12Chroma, dstNV12Pitch, dstNV12Width, dstNV12Height, srcNV12, srcNV12Pitch, srcNV12Width, srcNV12Height);
	}
};
};
};
};

