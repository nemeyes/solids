
//#include "mf_mv_base.h"

#include <d3d11.h>
#include <dxgi1_2.h>

struct VIEW_SESSION_INFO {
	ID3D11Texture2D* buffer;
	ID3D11ShaderResourceView* shader_resource_view;
	ID3D11VideoProcessorEnumerator* video_processor_enum;
	ID3D11VideoProcessor*           video_processor;
	UINT src_height;
	UINT src_width;
	UINT dst_height;
	UINT dst_width;
	FLOAT	position[4];
	FLOAT	active_video_ratio[2];	// width, height
	INT		control[4];	//left, right, up, down
	VIEW_SESSION_INFO()
		:src_height(0), src_width(0), dst_height(0), dst_width(0)
		, video_processor_enum(NULL), video_processor(NULL), buffer(NULL), shader_resource_view(NULL) 
	{
		active_video_ratio[0] = 1;
		active_video_ratio[1] = 1;
	};
};