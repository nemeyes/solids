#include "colorspace_converter.h"
#include <cuda_runtime.h>

namespace solids
{
namespace lib
{
namespace video
{
namespace nvidia
{

	__constant__ float mat_yuv2rgb[3][3];

	template<class T>
	__device__ static T clamp(T x, T lower, T upper)
	{
		return x < lower ? lower : (x > upper ? upper : x);
	}

	template<class rgbs, class yuvs>
	__device__ inline rgbs yuv_to_rgb_pixel(yuvs y, yuvs u, yuvs v)
	{
		const int32_t low = 1 << (sizeof(yuvs) * 8 - 4);
		const int32_t mid = 1 << (sizeof(yuvs) * 8 - 1);
		float fy = (int32_t)y - low;
		float fu = (int32_t)u - mid;
		float fv = (int32_t)v - mid;
		const float maxf = (1 << sizeof(yuvs) * 8) - 1.0f;
		yuvs r = (yuvs)clamp(mat_yuv2rgb[0][0] * fy + mat_yuv2rgb[0][1] * fu + mat_yuv2rgb[0][2] * fv, 0.0f, maxf);
		yuvs g = (yuvs)clamp(mat_yuv2rgb[1][0] * fy + mat_yuv2rgb[1][1] * fu + mat_yuv2rgb[1][2] * fv, 0.0f, maxf);
		yuvs b = (yuvs)clamp(mat_yuv2rgb[2][0] * fy + mat_yuv2rgb[2][1] * fu + mat_yuv2rgb[2][2] * fv, 0.0f, maxf);

		rgbs rgb = {};
		const int32_t shift = abs((int)sizeof(yuvs) - (int)sizeof(rgb.c.r)) * 8;
		if (sizeof(yuvs) >= sizeof(rgb.c.r))
		{
			rgb.c.r = r >> shift;
			rgb.c.g = g >> shift;
			rgb.c.b = b >> shift;
		}
		else
		{
			rgb.c.r = r << shift;
			rgb.c.g = g << shift;
			rgb.c.b = b << shift;
		}
		return rgb;
	}

	union bgra32
	{
		uint32_t d;
		uchar4 v;
		struct
		{
			uint8_t b, g, r, a;
		} c;
	};

	////////////////////////device////////////////////
	__global__ static void yuv_to_rgb_kernel(uint8_t* pYUV, int32_t yuvPitch, uint8_t* pRGB, int32_t rgbPitch, int32_t width, int32_t height)
	{
		int32_t x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
		int32_t y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
		if (x + 1 >= width || y + 1 >= height)
			return;

		uint8_t* pSrc = pYUV + x * sizeof(uchar2) / 2 + y * yuvPitch;
		uint8_t* pDst = pRGB + x * sizeof(bgra32) + y * rgbPitch;

		uchar2 l0 = *(uchar2*)pSrc;
		uchar2 l1 = *(uchar2*)(pSrc + yuvPitch);
		uchar2 ch = *(uchar2*)(pSrc + (height - y / 2) * yuvPitch);

		uint2 dst;
		dst.x = yuv_to_rgb_pixel<bgra32>(l0.x, ch.x, ch.y).d;
		dst.y = yuv_to_rgb_pixel<bgra32>(l0.y, ch.x, ch.y).d;
		*(uint2*)pDst = dst;

		dst.x = yuv_to_rgb_pixel<bgra32>(l1.x, ch.x, ch.y).d;
		dst.y = yuv_to_rgb_pixel<bgra32>(l1.y, ch.x, ch.y).d;
		*(uint2*)(pDst + rgbPitch) = dst;
	}

	__global__ static void nv12_to_i420_kernel(uint8_t* pNV12Chroma, int32_t nv12ChromaPitch, uint8_t* pI420Chroma, int32_t i420ChromaPitch, int32_t chromaWidth, int32_t chromaHeight)
	{
		int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
		int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

		uint8_t u = *(pNV12Chroma + (2 * x) + y * nv12ChromaPitch);
		uint8_t v = *(pNV12Chroma + (2 * x + 1) + y * nv12ChromaPitch);

		uint8_t* pI420U = pI420Chroma;
		uint8_t* pI420V = pI420U + (i420ChromaPitch * chromaHeight);
		*(pI420U + x + y * i420ChromaPitch) = u;
		*(pI420V + x + y * i420ChromaPitch) = v;
	}

	__global__ static void nv12_to_yv12_kernel(uint8_t* pNV12Chroma, int32_t nv12ChromaPitch, uint8_t* pYV12Chroma, int32_t yv12ChromaPitch, int32_t chromaWidth, int32_t chromaHeight)
	{
		int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
		int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

		uint8_t u = *(pNV12Chroma + (2 * x) + y * nv12ChromaPitch);
		uint8_t v = *(pNV12Chroma + (2 * x + 1) + y * nv12ChromaPitch);

		uint8_t* pYV12V = pYV12Chroma;
		uint8_t* pYV12U = pYV12V + (yv12ChromaPitch * chromaHeight);
		*(pYV12V + x + y * yv12ChromaPitch) = v;
		*(pYV12U + x + y * yv12ChromaPitch) = u;
	}

	void decoder::converter::convert_nv12_to_bgra32(unsigned char* nv12, int nv12Pitch, unsigned char* bgra, int bgraPicth, int width, int height)
	{
		if ((width * height) >= (RESOLUTION_W4K * RESOLUTION_H4K))
			setup_mat_yuv2rgb(solids::lib::video::nvidia::decoder::converter::standard_t::bt2020);
		else if ((width * height) >= (RESOLUTION_WFHD * RESOLUTION_HFHD))
			setup_mat_yuv2rgb(solids::lib::video::nvidia::decoder::converter::standard_t::bt709);
		else
			setup_mat_yuv2rgb(solids::lib::video::nvidia::decoder::converter::standard_t::bt601);
		yuv_to_rgb_kernel << <dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2), dim3(32, 2) >> > (nv12, nv12Pitch, bgra, bgraPicth, width, height);
	}

	void decoder::converter::convert_nv12_to_i420(unsigned char* nv12, int nv12Pitch, unsigned char* i420, int i420Pitch, int width, int height)
	{
		cudaError_t cerr = cudaMemcpy2D(i420, i420Pitch, nv12, nv12Pitch, width, height, cudaMemcpyDeviceToDevice);
		int chromaWidth = width >> 1;
		int chromaheight = height >> 1;
		unsigned char* nv12Chroma = nv12 + nv12Pitch * height;
		unsigned char* i420Chroma = i420 + i420Pitch * height;
		int nv12ChromaPitch = nv12Pitch;
		int i420ChromaPitch = i420Pitch >> 1;

		dim3 threadPerBlock(4, 4);
		dim3 blocks(chromaWidth / threadPerBlock.x, chromaheight / threadPerBlock.y);
		nv12_to_i420_kernel << <blocks, threadPerBlock >> > (nv12Chroma, nv12ChromaPitch, i420Chroma, i420ChromaPitch, chromaWidth, chromaheight);
	}


	void decoder::converter::convert_nv12_to_yv12(unsigned char* nv12, int nv12Pitch, unsigned char* yv12, int yv12Pitch, int width, int height)
	{
		cudaError_t cerr = cudaMemcpy2D(yv12, yv12Pitch, nv12, nv12Pitch, width, height, cudaMemcpyDeviceToDevice);
		int chromaWidth = width >> 1;
		int chromaheight = height >> 1;
		unsigned char* nv12Chroma = nv12 + nv12Pitch * height;
		unsigned char* yv12Chroma = yv12 + yv12Pitch * height;
		int nv12ChromaPitch = nv12Pitch;
		int yv12ChromaPitch = yv12Pitch >> 1;

		dim3 threadPerBlock(4, 4);
		dim3 blocks(chromaWidth / threadPerBlock.x, chromaheight / threadPerBlock.y);
		nv12_to_yv12_kernel << <blocks, threadPerBlock >> > (nv12Chroma, nv12ChromaPitch, yv12Chroma, yv12ChromaPitch, chromaWidth, chromaheight);
	}

	void decoder::converter::setup_mat_yuv2rgb(int imat)
	{
		float wr;
		float wb;
		int black;
		int white;
		int max;
		constants(imat, wr, wb, black, white, max);
		float mat[3][3] = {
			1.0f, 0.0f, (1.0f - wr) / 0.5f,
			1.0f, -wb * (1.0f - wb) / 0.5f / (1 - wb - wr), -wr * (1 - wr) / 0.5f / (1 - wb - wr),
			1.0f, (1.0f - wb) / 0.5f, 0.0f,
		};
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
				mat[i][j] = (float)(1.0 * max / (white - black) * mat[i][j]);
		}
		::cudaMemcpyToSymbol(mat_yuv2rgb, mat, sizeof(mat));
	}

	void decoder::converter::constants(int mat, float& wr, float& wb, int& black, int& white, int& max)
	{
		// Default is BT709
		wr = 0.2126f;
		wb = 0.0722f;
		black = 16;
		white = 235;
		max = 255;
		if (mat == solids::lib::video::nvidia::decoder::converter::converter::standard_t::bt601)
		{
			wr = 0.2990f;
			wb = 0.1140f;
		}
		else if (mat == solids::lib::video::nvidia::decoder::converter::standard_t::bt2020)
		{
			wr = 0.2627f;
			wb = 0.0593f;
			// 10-bit only
			black = 64 << 6;
			white = 940 << 6;
			max = (1 << 16) - 1;
		}
	}



};
};
};
};

