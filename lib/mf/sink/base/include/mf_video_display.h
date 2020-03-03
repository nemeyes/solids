#ifndef _SLD_MF_VIDEO_DISPLAY_H_
#define _SLD_MF_VIDEO_DISPLAY_H_

#include <mf_base.h>

#define MONITOR_INFO_PRIMARY_MONITOR	0x0001
#define MAX_MONITORS					16

namespace sld
{
	namespace lib
	{
		namespace mf
		{
			namespace sink
			{
				typedef struct _monitor_t
				{
					UINT			id;
					HMONITOR		monitor;
					TCHAR			device[32];
					LARGE_INTEGER	version;
					DWORD			vendor_id;
					DWORD			device_id;
					DWORD			subsys_id;
					DWORD			revision;
					SIZE			phys_mon_dim;
					DWORD			refresh_rate;
					IUnknown *		dd;
				} monitor_t;

				class monitors
				{
				public:
					monitors(void);
					virtual ~monitors(void);

					virtual HRESULT						initialize_display_system(_In_ HWND hwnd);
					virtual void						terminate_display_system(void);
					sld::lib::mf::sink::monitor_t *	find_monitor(_In_ HMONITOR hMon);
					HRESULT								match_guid(UINT id, _Out_ DWORD * match_id);
					sld::lib::mf::sink::monitor_t &	operator[](int32_t i);
					DWORD								count(void) const;
					static BOOL __stdcall				monitor_enum_proc(_In_ HMONITOR hMon, _In_opt_ HDC hDC, _In_ LPRECT pRect, LPARAM dwData);
					virtual BOOL						init_monitor(_In_ HMONITOR hMon, BOOL fXclMode);

				protected:
					BOOL								get_monitor_info(UINT uDevID, _Out_ sld::lib::mf::sink::monitor_t * lpmi, _In_ HMONITOR hm);
					virtual void						term_monitor_info(_Inout_ sld::lib::mf::sink::monitor_t * pmi);

				private:
					DWORD								_nmonitor;
					sld::lib::mf::sink::monitor_t	_monitor[MAX_MONITORS];
				};


				typedef struct _monitor_enum_proc_info_t 
				{
					HWND hwnd;
					sld::lib::mf::sink::monitors * monitor_array;
				} monitor_enum_proc_info_t;
			};
		};
	};
};








#endif