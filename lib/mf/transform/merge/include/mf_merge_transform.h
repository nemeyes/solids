#ifndef _SLD_MERGE_TRANSFORM_H_
#define _SLD_MERGE_TRANSFORM_H_

#include <mf_base.h>

#define MIN_INPUT_STREAM_COUNT 1
#define MAX_INPUT_STREAM_COUNT 9
#define BASE_OUTPUT_STREAM_ID 0

namespace solids
{
	namespace lib
	{
		namespace mf
		{
			namespace transform
			{
				namespace merge
				{
					class transform
						: solids::lib::mf::base
						, solids::lib::mf::refcount_object
						, public IMFTransform
						, public IMergeTransformContext
					{
						enum Error {
							NO_INDEX = -1
						};

						typedef struct _direction_t
						{
							static const int32_t unknonw = -1;
							static const int32_t input = 0;
							static const int32_t output = 1;
						} direction_t;

						typedef struct _output_type_t
						{
							static const int32_t unknown = -1;
							static const int32_t audio = 0;
							static const int32_t video = 1;
						} output_type_t;

					public:
						transform(void);
						~transform(void);

						static HRESULT CreateInstance(REFIID riid, void** ppv);

						// IUnknown
						STDMETHODIMP QueryInterface(REFIID iid, void** ppv);
						STDMETHODIMP_(ULONG) AddRef(void);
						STDMETHODIMP_(ULONG) Release(void);

						//IMFTransform
						STDMETHODIMP GetStreamLimits(DWORD * inputMinimum, DWORD * inputMaximum, DWORD * outputMinimum, DWORD * outputMaximum);
						STDMETHODIMP GetStreamCount(DWORD * inputStreams, DWORD * outputStreams);
						STDMETHODIMP GetStreamIDs(DWORD inputIDSize, DWORD * inputIDs, DWORD outputIDSize, DWORD * outputIDs);
						STDMETHODIMP GetInputStreamInfo(DWORD inputStreamID, MFT_INPUT_STREAM_INFO * streamInfo);
						STDMETHODIMP GetOutputStreamInfo(DWORD outputStreamID, MFT_OUTPUT_STREAM_INFO * streamInfo);
						STDMETHODIMP GetAttributes(IMFAttributes ** attributes);
						STDMETHODIMP GetInputStreamAttributes(DWORD inputStreamID, IMFAttributes ** attributes);
						STDMETHODIMP GetOutputStreamAttributes(DWORD outputStreamID, IMFAttributes ** attributes);
						STDMETHODIMP DeleteInputStream(DWORD streamID);
						STDMETHODIMP AddInputStreams(DWORD streamSize, DWORD * streamIDs);
						STDMETHODIMP GetInputAvailableType(DWORD inputStreamID, DWORD typeIndex, IMFMediaType ** ppmt);
						STDMETHODIMP GetOutputAvailableType(DWORD outputStreamID, DWORD typeIndex, IMFMediaType** ppmt);
						STDMETHODIMP SetInputType(DWORD id, IMFMediaType * mt, DWORD flags);
						STDMETHODIMP SetOutputType(DWORD id, IMFMediaType* mt, DWORD flags);
						STDMETHODIMP GetInputCurrentType(DWORD id, IMFMediaType ** ppmt);
						STDMETHODIMP GetOutputCurrentType(DWORD id, IMFMediaType ** ppmt);
						STDMETHODIMP GetInputStatus(DWORD id, DWORD * flags);
						STDMETHODIMP GetOutputStatus(DWORD * flags);
						STDMETHODIMP SetOutputBounds(LONGLONG lowerBound, LONGLONG upperBound);
						STDMETHODIMP ProcessEvent(DWORD isid, IMFMediaEvent * evt);
						STDMETHODIMP ProcessMessage(MFT_MESSAGE_TYPE msg, ULONG_PTR param);
						STDMETHODIMP ProcessInput(DWORD isid, IMFSample * sample, DWORD flags);
						STDMETHODIMP ProcessOutput(DWORD flags, DWORD outputBufferCount, MFT_OUTPUT_DATA_BUFFER * outputSamples, DWORD * status);

						//IMergeTransformContext
						STDMETHODIMP SetEnableID(UINT dwStreamID);
						STDMETHODIMP SetSeletedOnly(BOOL enable);

					private:
						HRESULT		is_valid_input_stream(DWORD isid) const;
						HRESULT		is_type_acceptable(int32_t dir, DWORD id, IMFMediaType * mt) const;
						HRESULT		set_media_type(const int32_t dir, const DWORD id, IMFMediaType * mt);
						DWORD		get_input_stream_index(DWORD id) const;

						void		set_active_video_info(IMFSample** sample);

					private:
						critical_section		_lock;

						DWORD					_selected_id;
						BOOL					_selected_only;
						int32_t					_overlapped_count;
						BOOL					_overlapped;

						DWORD					_input_stream_count;
						DWORD *					_input_stream_ids;

						BOOL					_is_input_type_set[MAX_INPUT_STREAM_COUNT];
						BOOL					_is_output_type_set;
						IMFMediaType *			_input_type[MAX_INPUT_STREAM_COUNT];
						BOOL					_input_updated[MAX_INPUT_STREAM_COUNT];
						IMFMediaType *			_output_type;
						IMFMediaBuffer *		_input_buffer[MAX_INPUT_STREAM_COUNT];

						LONGLONG				_stream_duration;
						LONGLONG				_prev_pts;
						LONGLONG				_curr_pts;

						BOOL					_is_first_sample;
						double *				_ratios;
					};
				};
			};
		};
	};
};

#endif