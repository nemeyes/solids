#ifndef _SLD_MF_BASE_H_
#define _SLD_MF_BASE_H_

#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>
#include <atlbase.h>
#include <float.h>
#include <tchar.h>
#include <math.h>
#include <strsafe.h>
#include <mmsystem.h>
#include <strmif.h>
#include <assert.h>
#include <stdint.h>
#include <vector>
#include <list>
#include <queue>
#include <map>
#include <assert.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mferror.h>
#include <mfreadwrite.h>
#include <evr.h>
#include <dcomp.h>
#include <wmcontainer.h>
#include <initguid.h>
#include <sstream>
#include <dxgi1_2.h>
#include <dxgi1_3.h>
#include <d3d11.h>

#define BREAK_ON_FAIL(value)            if(FAILED(value)) break;
#define BREAK_ON_NULL(value, newHr)     if(value == NULL) { hr = newHr; break; }
#define RETURN_ON_FAIL(value)			if(FAILED(value)) return value;

//const IID	BORROWED_IID_IMFDXGIDeviceManager	= { 0xeb533d5d, 0x2db6, 0x40f8, { 0x97, 0xa9, 0x49, 0x46, 0x92, 0x01, 0x4f, 0x07 } };
//const GUID	BORROWED_MF_SA_D3D11_AWARE			= { 0x206b4fc8, 0xfcf9, 0x4c51, { 0xaf, 0xe3, 0x97, 0x64, 0x36, 0x9e, 0x33, 0xa0 } };

namespace sld
{
	namespace lib
	{
		namespace mf
		{
			template <class T>
			struct noop
			{
				void operator()(T& t)
				{
				}
			};

			inline void safe_close_handle(HANDLE & h)
			{
				if (h != NULL)
				{
					::CloseHandle(h);
					h = NULL;
				}
			}

			template <class T> inline void safe_delete(T *& pt)
			{
				delete pt;
				pt = NULL;
			}

			template <class T> inline void safe_delete_array(T *& pt)
			{
				delete[] pt;
				pt = NULL;
			}

			template <class T> inline void safe_release(T *& pt)
			{
				if (pt != NULL)
				{
					pt->Release();
					pt = NULL;
				}
			}

			template <class T> inline double ticks2msecs(const T & t)
			{
				return t / 10000.0;
			}

			template <class T> inline T msecs2ticks(const T & t)
			{
				return t * 10000;
			}

			// returns the greatest common divisor of A and B
			inline int gcd(int A, int B)
			{
				int Temp;

				if (A < B)
				{
					Temp = A;
					A = B;
					B = Temp;
				}

				while (B != 0)
				{
					Temp = A % B;
					A = B;
					B = Temp;
				}

				return A;
			}

			inline float mf_offset_to_float(const MFOffset & offset)
			{
				return (float)offset.value + ((float)offset.value / 65536.0f);
			}

			inline RECT mf_video_area_to_rect(const MFVideoArea area)
			{
				float left = mf_offset_to_float(area.OffsetX);
				float top = mf_offset_to_float(area.OffsetY);

				RECT rc =
				{
					int(left + 0.5f),
					int(top + 0.5f),
					int(left + area.Area.cx + 0.5f),
					int(top + area.Area.cy + 0.5f)
				};

				return rc;
			}

			inline MFOffset make_offset(float v)
			{
				MFOffset offset;
				offset.value = short(v);
				offset.fract = WORD(65536 * (v - offset.value));
				return offset;
			}

			inline MFVideoArea make_area(float x, float y, DWORD width, DWORD height)
			{
				MFVideoArea area;
				area.OffsetX = make_offset(x);
				area.OffsetY = make_offset(y);
				area.Area.cx = width;
				area.Area.cy = height;
				return area;
			}

			class com_auto_release
			{
			public:
				void operator()(IUnknown *p)
				{
					if (p)
						p->Release();
				}
			};

			class mem_delete
			{
			public:
				void operator()(void*p)
				{
					if (p)
						delete p;
				}
			};

			class base
			{
			public:
				static long get_obj_count(void)
				{
					return _obj_count;
				}

			protected:
				base(void)
				{
					InterlockedIncrement(&_obj_count);
				}

				virtual ~base(void)
				{
					InterlockedDecrement(&_obj_count);
				}

			private:
				static volatile long _obj_count;
			};
		};
	};
};

#include <mf_critical_section.h>
#include <mf_refcount_object.h>
#include <mf_registry.h>
#include <mf_async_operation.h>
#include <mf_async_callback.h>
#include <mf_linked_list.h>
#include <mf_buffer.h>
#include <mf_growable_array.h>
#include <mf_linked_list.h>
#include <mf_tiny_map.h>
#include <mf_attributes.h>
#include <mf_clsid.h>
#include <mf_thread_safe_queue.h>
#include <mf_scheduler.h>
#include <mf_marker.h>

#endif