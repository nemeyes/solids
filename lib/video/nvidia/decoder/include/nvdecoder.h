#pragma once

#include "sld_nvdecoder.h"
#include <nvcuvid.h>
#include <cuda.h>
#include <vector>
#include <map>

namespace solids
{
	namespace lib
	{
		namespace video
		{
			namespace nvidia
			{
				class decoder::core
				{
				public:
					core(solids::lib::video::nvidia::decoder* front);
					virtual ~core(void);

					BOOL	is_initialized(void);
					void*	context(void);

					int32_t	initialize(solids::lib::video::nvidia::decoder::context_t* ctx);
					int32_t	release(void);

					int32_t	decode(uint8_t* bitstream, int32_t bitstreamSize, long long bitstreamTimestamp, uint8_t*** nv12, int32_t* numberOfDecoded, long long** timetstamp);
					size_t	get_pitch(void);
					size_t	get_pitch_resized(void);
					size_t	get_pitch_converted(void);
					size_t	get_pitch2(void);
				private:
					int32_t process_video_sequence(CUVIDEOFORMAT* format);
					int32_t process_picture_decode(CUVIDPICPARAMS* picture);
					int32_t process_picture_display(CUVIDPARSERDISPINFO* display);
					int32_t get_number_of_decode_surfaces(cudaVideoCodec codec, int32_t width, int32_t height);

					static int __stdcall process_video_sequence(void* user_data, CUVIDEOFORMAT* format);
					static int __stdcall process_picture_decode(void* user_data, CUVIDPICPARAMS* picture);
					static int __stdcall process_picture_display(void* user_data, CUVIDPARSERDISPINFO* display);

				private:
					core(const solids::lib::video::nvidia::decoder::core& clone);

				private:
					solids::lib::video::nvidia::decoder*				_front;
					BOOL												_initialized;
					solids::lib::video::nvidia::decoder::context_t*		_context;
					CRITICAL_SECTION			_lock;
					CRITICAL_SECTION			_lock2;
					CUcontext					_cu_context;
					CUvideoctxlock				_cu_ctx_lock;
					CUvideoparser				_cu_parser;
					CUvideodecoder				_cu_decoder;

					int32_t						_cu_width;
					int32_t						_cu_height;
					int32_t						_cu_surface_height;
					cudaVideoCodec				_cu_codec;
					cudaVideoChromaFormat		_cu_chroma_format;
					int32_t						_cu_bitdepth_minus8;
					CUVIDEOFORMAT				_cu_format;
					size_t						_cu_pitch;
					size_t						_cu_pitch_resized;
					size_t						_cu_pitch_converted;
					size_t						_cu_pitch2;

					std::vector<uint8_t*>		_frame;
					std::vector<uint8_t*>		_frame_resized;
					std::vector<uint8_t*>		_frame_converted;
					std::vector<uint8_t*>		_frame2;
					std::vector<long long>		_timestamp;
					CRITICAL_SECTION			_frame_lock;
					int32_t						_ndecoded_frame;
					int32_t						_ndecoded_frame_returned;
				};
			};
		};
	};
};

