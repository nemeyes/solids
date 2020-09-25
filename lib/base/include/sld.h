#ifndef _SLD_H_
#define _SLD_H_

//#include <winsock2.h>
#include <windows.h>
//#include <winrt\Windows.Foundation.h>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <sstream>
#include <functional>
#include <algorithm>
#include <array>
#include <map>
#include <stack>
#include <vector>
#include <tuple>
#include <codecvt>
#include <dxgi1_2.h>
#include <dxgi1_3.h>
#include <dxgi1_4.h>
#include <dxgi1_5.h>
#include <dxgi1_6.h>
#include <d3d11.h>
#include <d3d11_1.h>
#include <d3d11_2.h>
#include <d3d11_3.h>
#include <d3d11_4.h>


//#if defined(_DEBUG) && !defined(WITH_DISABLE_VLD)
//#include <vld.h>
//#endif

#define SLD_MAX(A, B)                           (((A) > (B)) ? (A) : (B))
#define SLD_MIN(A, B)                           (((A) < (B)) ? (A) : (B))

#define SLD_ALIGN16(value)                      (((value + 15) >> 4) << 4) // round up to a multiple of 16
#define SLD_ALIGN32(value)                      (((value + 31) >> 5) << 5) // round up to a multiple of 32
#define SLD_ALIGN(value, alignment)             (alignment) * ( (value) / (alignment) + (((value) % (alignment)) ? 1 : 0))

#define SLD_BREAK_ON_ERROR(P)                   {if (MFX_ERR_NONE != (P)) break;}
#define SLD_SAFE_DELETE_ARRAY(P)                {if (P) {delete[] P; P = NULL;}}
#define SLD_SAFE_RELEASE(X)                     {if (X) { X->Release(); X = NULL; }}
#define SLD_SAFE_FREE(X)                        {if (X) { free(X); X = NULL; }}
#define SLD_SAFE_DELETE(P)                      {if (P) {delete P; P = NULL;}}

namespace solids
{
	namespace lib
	{
		template <class T> void safe_release(T*& pt)
		{
			if (pt != NULL)
			{
				pt->Release();
				pt = NULL;
			}
		}

		class base
		{
		public:
			typedef struct _err_code_t
			{
				static const int32_t unknown = -1;
				static const int32_t success = 0;
				static const int32_t generic_fail = 1;
				static const int32_t invalid_parameter = 2;
				static const int32_t invalid_file_path = 3;
				static const int32_t unsupported_media_file = 4;
			} err_code_t;

			typedef struct _protocol_t
			{
				static const int32_t file = 0;
				static const int32_t rtmp = 1;
				static const int32_t rtsp = 2;
				static const int32_t hls = 3;
				static const int32_t dash = 4;
				static const int32_t hds = 5;
			} protocol_t;

			typedef struct _media_type_t
			{
				static const int32_t unknown = 0x00;
				static const int32_t video = 0x01;
				static const int32_t audio = 0x02;
			} media_type_t;

			typedef struct _video_codec_t
			{
				static const int32_t unknown = -1;
				static const int32_t vp6 = 0;
				static const int32_t avc = 1;
				static const int32_t mp4v = 2;
				static const int32_t hevc = 3;
				static const int32_t vp8 = 4;
				static const int32_t vp9 = 5;
			} video_codec_t;

			typedef struct _colorspace_t
			{
				static const int nv12 = 0;
				static const int bgra = 1;
				static const int yv12 = 2;
				static const int i420 = 3;
			} colorspace_t;

			typedef struct _audio_codec_t
			{
				static const int32_t unknown = -1;
				static const int32_t mp3 = 0;
				static const int32_t alaw = 1;
				static const int32_t mlaw = 2;
				static const int32_t aac = 3;
				static const int32_t ac3 = 4;
				static const int32_t opus = 5;
			} audio_codec_t;

			typedef struct _audio_sample_t
			{
				static const int32_t unknown = -1;
				static const int32_t fmt_u8 = 0;
				static const int32_t fmt_s16 = 1;
				static const int32_t fmt_s32 = 2;
				static const int32_t fmt_flt = 3;
				static const int32_t fmt_dbl = 4;
				static const int32_t fmt_s64 = 5;
				static const int32_t fmt_u8p = 6;
				static const int32_t fmt_s16p = 7;
				static const int32_t fmt_s32p = 8;
				static const int32_t fmt_fltp = 9;
				static const int32_t fmt_dblp = 10;
				static const int32_t fmt_s64p = 11;
			} audio_sample_t;

			typedef struct _render_mode_t
			{
				static const int32_t unknown = -1;
				static const int32_t stretch = 0;
				static const int32_t original = 1;
			} render_mode_t;
		};
	};
};

#endif