#include "Nvdecoder.h"
#include <sld_locks.h>
#include "colorspace_converter.h"
#include "resizer.h"

namespace solids
{
	namespace lib
	{
		namespace video
		{
			namespace nvidia
			{
				decoder::core::core(solids::lib::video::nvidia::decoder* front)
					: _front(front)
					, _initialized(FALSE)
					, _cu_context(NULL)
					, _cu_parser(NULL)
					, _cu_decoder(NULL)
					, _cu_width(0)
					, _cu_height(0)
					, _cu_surface_height(0)
					, _cu_codec(cudaVideoCodec_NumCodecs)
					, _cu_pitch(0)
					, _cu_pitch_resized(0)
					, _cu_pitch_converted(0)
					, _cu_pitch2(0)
				{
					::memset(&_cu_format, 0x00, sizeof(_cu_format));
					::InitializeCriticalSection(&_lock);
					::InitializeCriticalSection(&_lock2);
					::InitializeCriticalSection(&_frame_lock);
				}

				decoder::core::~core(void)
				{
					::DeleteCriticalSection(&_frame_lock);
					::DeleteCriticalSection(&_lock2);
					::DeleteCriticalSection(&_lock);
					_initialized = FALSE;
				}

				BOOL decoder::core::is_initialized(void)
				{
					return _initialized;
				}

				void* decoder::core::context(void)
				{
					return (void*)&_cu_context;
				}

				int32_t decoder::core::initialize(solids::lib::video::nvidia::decoder::context_t* ctx)
				{
					_context = ctx;
					int32_t ngpu = 0;
					CUresult cret = ::cuInit(0);
					cret = ::cuDeviceGetCount(&ngpu);
					if ((_context->deviceIndex < 0) || (_context->deviceIndex >= ngpu))
						return -1;
					CUdevice cuDevice;
					cret = ::cuDeviceGet(&cuDevice, _context->deviceIndex);
					cret = ::cuCtxCreate(&_cu_context, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice);
					cret = ::cuvidCtxLockCreate(&_cu_ctx_lock, _cu_context);

					CUVIDPARSERPARAMS videoParserParameters = {};
					switch (_context->codec)
					{
					case solids::lib::video::nvidia::decoder::video_codec_t::mp4v:
						videoParserParameters.CodecType = cudaVideoCodec_MPEG4;
						break;
					case solids::lib::video::nvidia::decoder::video_codec_t::avc:
						videoParserParameters.CodecType = cudaVideoCodec_H264;
						break;
					case solids::lib::video::nvidia::decoder::video_codec_t::hevc:
						videoParserParameters.CodecType = cudaVideoCodec_HEVC;
						break;
					}
					videoParserParameters.ulMaxNumDecodeSurfaces = 1;
					videoParserParameters.ulMaxDisplayDelay = 0;
					videoParserParameters.pUserData = this;
					videoParserParameters.pfnSequenceCallback = solids::lib::video::nvidia::decoder::core::process_video_sequence;
					videoParserParameters.pfnDecodePicture = solids::lib::video::nvidia::decoder::core::process_picture_decode;
					videoParserParameters.pfnDisplayPicture = solids::lib::video::nvidia::decoder::core::process_picture_display;
					{
						solids::lib::autolock lock(&_lock);
						::cuvidCreateVideoParser(&_cu_parser, &videoParserParameters);
					}
					_initialized = TRUE;
					return solids::lib::video::nvidia::decoder::err_code_t::success;
				}

				int32_t decoder::core::release(void)
				{
					solids::lib::autolock lock2(&_lock2);
					CUresult cret;
					if (_cu_parser)
					{
						cret = ::cuvidDestroyVideoParser(_cu_parser);
						_cu_parser = NULL;
					}
					if (_cu_decoder)
					{
						solids::lib::autolock lock(&_lock);
						::cuCtxPushCurrent(_cu_context);
						cret = ::cuvidDestroyDecoder(_cu_decoder);
						::cuCtxPopCurrent(NULL);
						_cu_decoder = NULL;
					}

					{
						solids::lib::autolock lock(&_frame_lock);
						std::vector<uint8_t*>::iterator iter;
						for (iter = _frame.begin(); iter != _frame.end(); iter++)
						{
							solids::lib::autolock lock2(&_lock);
							::cuCtxPushCurrent(_cu_context);
							::cuMemFree((CUdeviceptr)(*iter));
							::cuCtxPopCurrent(NULL);
						}
						_frame.clear();

						if (_context != nullptr && ((_context->width != _cu_width) || (_context->height != _cu_height)))
						{
							solids::lib::autolock lock(&_frame_lock);
							for (iter = _frame_resized.begin(); iter != _frame_resized.end(); iter++)
							{
								solids::lib::autolock lock2(&_lock);
								::cuCtxPushCurrent(_cu_context);
								::cuMemFree((CUdeviceptr)(*iter));
								::cuCtxPopCurrent(NULL);
							}
							_frame_resized.clear();
						}

						if (_context != nullptr && (_context->colorspace != solids::lib::video::nvidia::decoder::colorspace_t::nv12))
						{
							solids::lib::autolock lock(&_frame_lock);
							for (iter = _frame_converted.begin(); iter != _frame_converted.end(); iter++)
							{
								solids::lib::autolock lock2(&_lock);
								::cuCtxPushCurrent(_cu_context);
								::cuMemFree((CUdeviceptr)(*iter));
								::cuCtxPopCurrent(NULL);
							}
							_frame_converted.clear();
						}
						_frame2.clear();
					}
					::cuvidCtxLockDestroy(_cu_ctx_lock);
					::cuCtxDestroy(_cu_context);
					_initialized = FALSE;
					return solids::lib::video::nvidia::decoder::err_code_t::success;
				}

				int32_t decoder::core::decode(uint8_t* bitstream, int32_t bitstreamSize, long long bitstreamTimestamp, uint8_t*** nv12, int32_t* numberOfDecoded, long long** timetstamp)
				{
					solids::lib::autolock lock2(&_lock2);

					if (!_cu_parser)
						return solids::lib::video::nvidia::decoder::err_code_t::generic_fail;

					_ndecoded_frame = 0;
					CUVIDSOURCEDATAPACKET packet = { 0 };
					packet.payload = bitstream;
					packet.payload_size = bitstreamSize;
					packet.flags = CUVID_PKT_TIMESTAMP | CUVID_PKT_ENDOFPICTURE; //
					packet.timestamp = bitstreamTimestamp;

					if (!bitstream || (bitstreamSize < 1))
					{
						packet.flags |= CUVID_PKT_ENDOFSTREAM;
					}
					{
						solids::lib::autolock lock(&_lock);
						::cuvidParseVideoData(_cu_parser, &packet);
					}

					if (_ndecoded_frame > 0)
					{
						_frame2.clear();
						if (nv12)
						{
							int32_t index = 0;
							solids::lib::autolock lock(&_frame_lock);
							std::vector<uint8_t*>::iterator iter;
							for (iter = _frame.begin(); iter != (_frame.begin() + _ndecoded_frame); iter++, index++)
							{
								if (_context->colorspace == solids::lib::video::nvidia::decoder::colorspace_t::bgra)
								{
									if ((_context->width != _cu_width) || (_context->height != _cu_height))
									{
										solids::lib::video::nvidia::decoder::resizer::resize_nv12((uint8_t*)_frame_resized[index], (int32_t)_cu_pitch_resized, _context->width, _context->height, (*iter), (int32_t)_cu_pitch, _cu_width, _cu_height);
										solids::lib::video::nvidia::decoder::converter::convert_nv12_to_bgra32((uint8_t*)_frame_resized[index], (int32_t)_cu_pitch_resized, (uint8_t*)_frame_converted[index], (int32_t)_cu_pitch_converted, _context->width, _context->height);
									}
									else
									{
										solids::lib::video::nvidia::decoder::converter::convert_nv12_to_bgra32((*iter), (int32_t)_cu_pitch, (uint8_t*)_frame_converted[index], (int32_t)_cu_pitch_converted, _context->width, _context->height);
									}
									_cu_pitch2 = _cu_pitch_converted;
									_frame2.push_back(_frame_converted[index]);
								}
								else if (_context->colorspace == solids::lib::video::nvidia::decoder::colorspace_t::i420)
								{
									if ((_context->width != _cu_width) || (_context->height != _cu_height))
									{
										solids::lib::video::nvidia::decoder::resizer::resize_nv12((uint8_t*)_frame_resized[index], (int32_t)_cu_pitch_resized, _context->width, _context->height, (*iter), (int32_t)_cu_pitch, _cu_width, _cu_height);
										solids::lib::video::nvidia::decoder::converter::convert_nv12_to_i420((uint8_t*)_frame_resized[index], (int32_t)_cu_pitch_resized, (uint8_t*)_frame_converted[index], (int32_t)_cu_pitch_converted, _context->width, _context->height);
									}
									else
									{
										solids::lib::video::nvidia::decoder::converter::convert_nv12_to_i420((*iter), (int32_t)_cu_pitch, (uint8_t*)_frame_converted[index], (int32_t)_cu_pitch_converted, _context->width, _context->height);
									}
									_cu_pitch2 = _cu_pitch_converted;
									_frame2.push_back(_frame_converted[index]);
								}
								else if (_context->colorspace == solids::lib::video::nvidia::decoder::colorspace_t::yv12)
								{
									if ((_context->width != _cu_width) || (_context->height != _cu_height))
									{
										solids::lib::video::nvidia::decoder::resizer::resize_nv12((uint8_t*)_frame_resized[index], (int32_t)_cu_pitch_resized, _context->width, _context->height, (*iter), (int32_t)_cu_pitch, _cu_width, _cu_height);
										solids::lib::video::nvidia::decoder::converter::convert_nv12_to_yv12((uint8_t*)_frame_resized[index], (int32_t)_cu_pitch_resized, (uint8_t*)_frame_converted[index], (int32_t)_cu_pitch_converted, _context->width, _context->height);
									}
									else
									{
										solids::lib::video::nvidia::decoder::converter::convert_nv12_to_yv12((*iter), (int32_t)_cu_pitch, (uint8_t*)_frame_converted[index], (int32_t)_cu_pitch_converted, _context->width, _context->height);
									}
									_cu_pitch2 = _cu_pitch_converted;
									_frame2.push_back(_frame_converted[index]);
								}
								else
								{
									if ((_context->width != _cu_width) || (_context->height != _cu_height))
									{
										solids::lib::video::nvidia::decoder::resizer::resize_nv12((uint8_t*)_frame_resized[index], (int32_t)_cu_pitch_resized, _context->width, _context->height, (*iter), (int32_t)_cu_pitch, _cu_width, _cu_height);
										_cu_pitch2 = _cu_pitch_resized;
										_frame2.push_back(_frame_resized[index]);
									}
									else
									{
										_cu_pitch2 = _cu_pitch;
										_frame2.push_back((*iter));
									}
								}
							}
							*nv12 = &_frame2[0];
						}
						if (timetstamp)
						{
							*timetstamp = &_timestamp[0];
						}
					}
					if (numberOfDecoded)
					{
						*numberOfDecoded = _ndecoded_frame;
					}
					return solids::lib::video::nvidia::decoder::err_code_t::success;
				}

				size_t decoder::core::get_pitch(void)
				{
					return _cu_pitch;
				}

				size_t decoder::core::get_pitch2(void)
				{
					return _cu_pitch2;
				}

				size_t decoder::core::get_pitch_resized(void)
				{
					return _cu_pitch_resized;
				}

				size_t decoder::core::get_pitch_converted(void)
				{
					return _cu_pitch_converted;
				}

				int32_t decoder::core::process_video_sequence(CUVIDEOFORMAT* format)
				{
					int32_t numberOfDecodeSurfaces = format->min_num_decode_surfaces;// get_number_of_decode_surfaces(format->codec, format->coded_width, format->coded_height);
					if (_cu_width && _cu_height)
					{
						if ((format->coded_width == _cu_format.coded_width) && (format->coded_height == _cu_format.coded_height))
							return numberOfDecodeSurfaces;
						return numberOfDecodeSurfaces; // this error means current cuda device isn't support dynamic resolution change
					}

					_cu_codec = format->codec;
					_cu_chroma_format = format->chroma_format;
					_cu_bitdepth_minus8 = format->bit_depth_luma_minus8;
					_cu_format = *format;

					CUVIDDECODECREATEINFO videoDecodeCreateInfo = { 0 };
					videoDecodeCreateInfo.CodecType = _cu_codec;
					videoDecodeCreateInfo.ChromaFormat = _cu_chroma_format;
					if (_cu_chroma_format == cudaVideoChromaFormat_420)
						videoDecodeCreateInfo.OutputFormat = _cu_bitdepth_minus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
					else if (_cu_chroma_format == cudaVideoChromaFormat_422)
						videoDecodeCreateInfo.OutputFormat = _cu_bitdepth_minus8 ? cudaVideoSurfaceFormat_YUV444_16Bit : cudaVideoSurfaceFormat_YUV444;
					videoDecodeCreateInfo.bitDepthMinus8 = format->bit_depth_luma_minus8;
					if(format->progressive_sequence)
						videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
					else
						videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
					videoDecodeCreateInfo.ulNumOutputSurfaces = 2;
					videoDecodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
					videoDecodeCreateInfo.ulNumDecodeSurfaces = numberOfDecodeSurfaces;
					videoDecodeCreateInfo.vidLock = _cu_ctx_lock;
					videoDecodeCreateInfo.ulWidth = format->coded_width;
					videoDecodeCreateInfo.ulHeight = format->coded_height;
					videoDecodeCreateInfo.display_area.left = format->display_area.left;
					videoDecodeCreateInfo.display_area.right = format->display_area.right;
					videoDecodeCreateInfo.display_area.top = format->display_area.top;
					videoDecodeCreateInfo.ulTargetWidth = format->coded_width;
					videoDecodeCreateInfo.ulTargetHeight = format->coded_height;

					_cu_width = format->display_area.right - format->display_area.left;
					_cu_height = format->display_area.bottom - format->display_area.top;
					_cu_surface_height = videoDecodeCreateInfo.ulTargetHeight;

					::cuCtxPushCurrent(_cu_context);
					CUresult cret = ::cuvidCreateDecoder(&_cu_decoder, &videoDecodeCreateInfo);
					::cuCtxPopCurrent(NULL);
					return numberOfDecodeSurfaces;
				}

				int32_t decoder::core::process_picture_decode(CUVIDPICPARAMS* picture)
				{
					if (!_cu_decoder)
						return -1;
					::cuvidDecodePicture(_cu_decoder, picture);

					return 1;
				}

				int32_t decoder::core::process_picture_display(CUVIDPARSERDISPINFO* display)
				{
					CUVIDPROCPARAMS videoProcessingParameters = {};
					videoProcessingParameters.progressive_frame = display->progressive_frame;
					videoProcessingParameters.second_field = display->repeat_first_field + 1;
					videoProcessingParameters.top_field_first = display->top_field_first;
					videoProcessingParameters.unpaired_field = display->repeat_first_field < 0;

					CUdeviceptr dpSrcFrame = 0;
					uint32_t srcPitch = 0;
					::cuvidMapVideoFrame(_cu_decoder, display->picture_index, &dpSrcFrame, &srcPitch, &videoProcessingParameters);
					uint8_t* pDecodedFrame = NULL;
					{
						solids::lib::autolock lock(&_frame_lock);
						if (size_t(++_ndecoded_frame) > _frame.size())
						{
							//_nframe_alloc++;
							uint8_t* pFrame = NULL;
							::cuCtxPushCurrent(_cu_context);
							::cuMemAllocPitch((CUdeviceptr*)&pFrame, &_cu_pitch, _cu_width * (_cu_bitdepth_minus8 ? 2 : 1), (_cu_height >> 1) * 3, 16);
							::cuCtxPopCurrent(NULL);
							_frame.push_back(pFrame);

							if ((_context->width != _cu_width) || (_context->height != _cu_height))
							{
								::cuCtxPushCurrent(_cu_context);
								::cuMemAllocPitch((CUdeviceptr*)&pFrame, &_cu_pitch_resized, _context->width * (_cu_bitdepth_minus8 ? 2 : 1), (_context->height >> 1) * 3, 16);
								::cuCtxPopCurrent(NULL);
								_frame_resized.push_back(pFrame);
							}
							if (_context->colorspace != solids::lib::video::nvidia::decoder::colorspace_t::nv12)
							{
								if (_context->colorspace == solids::lib::video::nvidia::decoder::colorspace_t::bgra)
								{
									::cuCtxPushCurrent(_cu_context);
									::cuMemAllocPitch((CUdeviceptr*)&pFrame, &_cu_pitch_converted, 4 * _context->width, _context->height, 16);
									::cuCtxPopCurrent(NULL);
									_frame_converted.push_back(pFrame);
								}
								else
								{
									size_t cuPitchConverted = 0;
									::cuCtxPushCurrent(_cu_context);
									CUresult cret = ::cuMemAllocPitch((CUdeviceptr*)&pFrame, &cuPitchConverted, _context->width, (_context->height >> 1) * 3, 16);
									::cuCtxPopCurrent(NULL);

									_frame_converted.push_back(pFrame);
									_cu_pitch_converted = cuPitchConverted;
								}
							}
						}
						pDecodedFrame = _frame[_ndecoded_frame - 1];
					}

					::cuCtxPushCurrent(_cu_context);
					CUDA_MEMCPY2D m = { 0 };
					m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
					m.srcDevice = dpSrcFrame;
					m.srcPitch = srcPitch;
					m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
					m.dstDevice = (CUdeviceptr)(pDecodedFrame);
					m.dstPitch = _cu_pitch ? _cu_pitch : _cu_width * (_cu_bitdepth_minus8 ? 2 : 1);
					m.WidthInBytes = _cu_width * (_cu_bitdepth_minus8 ? 2 : 1);
					m.Height = _cu_height;
					::cuMemcpy2DAsync(&m, 0);

					m.srcDevice = (CUdeviceptr)((uint8_t*)dpSrcFrame + m.srcPitch * _cu_surface_height);
					m.dstDevice = (CUdeviceptr)(m.dstHost = pDecodedFrame + m.dstPitch * _cu_height);
					m.Height = _cu_height >> 1;
					::cuMemcpy2DAsync(&m, 0);
					::cuStreamSynchronize(0);
					::cuCtxPopCurrent(NULL);

					if (int32_t(_timestamp.size()) < _ndecoded_frame)
						_timestamp.resize(_frame.size());
					_timestamp[_ndecoded_frame - 1] = display->timestamp;

					::cuvidUnmapVideoFrame(_cu_decoder, dpSrcFrame);

					return 1;
				}

				int32_t decoder::core::get_number_of_decode_surfaces(cudaVideoCodec codec, int32_t width, int32_t height)
				{
					if (codec == cudaVideoCodec_H264)
						return 20;
					if (codec == cudaVideoCodec_HEVC)
					{
						// ref HEVC spec: A.4.1 General tier and level limits
						// currently assuming level 6.2, 8Kx4K
						int32_t MaxLumaPS = 35651584;
						int32_t MaxDpbPicBuf = 6;
						int32_t PicSizeInSamplesY = (int32_t)(width * height);
						int32_t MaxDpbSize;
						if (PicSizeInSamplesY <= (MaxLumaPS >> 2))
							MaxDpbSize = MaxDpbPicBuf * 4;
						else if (PicSizeInSamplesY <= (MaxLumaPS >> 1))
							MaxDpbSize = MaxDpbPicBuf * 2;
						else if (PicSizeInSamplesY <= ((3 * MaxLumaPS) >> 2))
							MaxDpbSize = (MaxDpbPicBuf * 4) / 3;
						else
							MaxDpbSize = MaxDpbPicBuf;
						return (std::min)(MaxDpbSize, 16) + 4;
					}
					return 8;
				}

				int32_t decoder::core::process_video_sequence(void* user_data, CUVIDEOFORMAT* format)
				{
					return (static_cast<decoder::core*>(user_data))->process_video_sequence(format);
				}

				int32_t decoder::core::process_picture_decode(void* user_data, CUVIDPICPARAMS* picture)
				{
					return (static_cast<decoder::core*>(user_data))->process_picture_decode(picture);
				}

				int32_t decoder::core::process_picture_display(void* user_data, CUVIDPARSERDISPINFO* display)
				{
					return (static_cast<decoder::core*>(user_data))->process_picture_display(display);
				}
			};
		};
	};
};

