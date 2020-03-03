#include "mf_video_display.h"
#include <strsafe.h>

#ifndef DEFAULT_DENSITY_LIMIT
#define DEFAULT_DENSITY_LIMIT       60
#endif
#ifndef WIDTH
#define WIDTH(x) ((x)->right - (x)->left)
#endif
#ifndef HEIGHT
#define HEIGHT(x) ((x)->bottom - (x)->top)
#endif

/*
struct DDRAWINFO
{
	DWORD								count;
	DWORD								pmi_size;
	HRESULT								hr_cb;
	const GUID*							guid;
	solids::lib::mf::sink::monitor_t *	pmi;
	HWND								hwnd;
};
*/

void solids::lib::mf::sink::monitors::term_monitor_info(_Inout_ solids::lib::mf::sink::monitor_t* pmi)
{
	::ZeroMemory(pmi, sizeof(solids::lib::mf::sink::monitor_t));
}

BOOL solids::lib::mf::sink::monitors::get_monitor_info(UINT uDevID, _Out_ solids::lib::mf::sink::monitor_t * lpmi, _In_ HMONITOR hm)
{
	MONITORINFOEX miInfoEx;
	miInfoEx.cbSize = sizeof(miInfoEx);

	lpmi->monitor = NULL;
	lpmi->id = 0;
	lpmi->phys_mon_dim.cx = 0;
	lpmi->phys_mon_dim.cy = 0;
	lpmi->refresh_rate = DEFAULT_DENSITY_LIMIT;

	if (GetMonitorInfo(hm, &miInfoEx))
	{
		HRESULT hr = StringCchCopy(lpmi->device, sizeof(lpmi->device) / sizeof(lpmi->device[0]), miInfoEx.szDevice);
		if (FAILED(hr))
			return FALSE;

		lpmi->monitor = hm;
		lpmi->id = uDevID;
		lpmi->phys_mon_dim.cx = WIDTH(&miInfoEx.rcMonitor);
		lpmi->phys_mon_dim.cy = HEIGHT(&miInfoEx.rcMonitor);

		int j = 0;
		DISPLAY_DEVICE ddm;
		ddm.cb = sizeof(ddm);
		while (EnumDisplayDevices(lpmi->device, j, &ddm, 0))
		{
			if (ddm.StateFlags & DISPLAY_DEVICE_ATTACHED_TO_DESKTOP)
			{
				DEVMODE     dm;
				::ZeroMemory(&dm, sizeof(dm));
				dm.dmSize = sizeof(dm);
				if (EnumDisplaySettings(lpmi->device, ENUM_CURRENT_SETTINGS, &dm))
					lpmi->refresh_rate = dm.dmDisplayFrequency == 0 ? lpmi->refresh_rate : dm.dmDisplayFrequency;

				// Remove registry snooping for monitor dimensions, as this is not supported by LDDM.
				// if (!FindMonitorDimensions(ddMonitor.DeviceID, &lpmi->physMonDim.cx, &lpmi->physMonDim.cy))
				{
					lpmi->phys_mon_dim.cx = WIDTH(&miInfoEx.rcMonitor);
					lpmi->phys_mon_dim.cy = HEIGHT(&miInfoEx.rcMonitor);
				}
			}
			j++;
		}
		return TRUE;
	}
	return FALSE;
}

BOOL solids::lib::mf::sink::monitors::init_monitor(_In_ HMONITOR hMon, BOOL fXclMode)
{
	if (get_monitor_info(_nmonitor, &_monitor[_nmonitor], hMon))
	{
		_monitor[_nmonitor].dd = (IUnknown*)1; // make checks for pDD succeed.
		_nmonitor++;
	}

	if (MAX_MONITORS >= _nmonitor)
		return TRUE;

	return FALSE;
}

BOOL CALLBACK solids::lib::mf::sink::monitors::monitor_enum_proc(_In_ HMONITOR hMon, _In_opt_ HDC hDC, _In_ LPRECT pRect, LPARAM dwData)
{
	solids::lib::mf::sink::monitor_enum_proc_info_t * info = (solids::lib::mf::sink::monitor_enum_proc_info_t*)dwData;
	if (!info)
		return TRUE;

	return info->monitor_array->init_monitor(hMon, FALSE);
}

HRESULT solids::lib::mf::sink::monitors::initialize_display_system(_In_ HWND hwnd)
{
	HRESULT hr = S_OK;

	solids::lib::mf::sink::monitor_enum_proc_info_t info;

	info.hwnd = hwnd;
	info.monitor_array = this;

	EnumDisplayMonitors(NULL, NULL, &monitor_enum_proc, (LPARAM)&info);

	if (_nmonitor == 0)
	{
		hr = HRESULT_FROM_WIN32(GetLastError());
		return(hr);
	}

	return(hr);
}

solids::lib::mf::sink::monitor_t * solids::lib::mf::sink::monitors::find_monitor(_In_ HMONITOR hMon)
{
	for (DWORD i = 0; i < _nmonitor; i++)
	{
		if (hMon == _monitor[i].monitor)
		{
			return &_monitor[i];
		}
	}
	return NULL;
}

HRESULT solids::lib::mf::sink::monitors::match_guid(UINT uDevID, _Out_ DWORD * pdwMatchID)
{
	HRESULT hr = S_OK;
	*pdwMatchID = 0;
	for (DWORD i = 0; i < _nmonitor; i++)
	{
		UINT uMonDevID = _monitor[i].id;
		if (uDevID == uMonDevID)
		{
			*pdwMatchID = i;
			hr = S_OK;
			return(hr);
		}
	}

	hr = S_FALSE;
	return(hr);
}

solids::lib::mf::sink::monitor_t & solids::lib::mf::sink::monitors::operator[](int32_t i)
{
	return _monitor[i];
}
DWORD solids::lib::mf::sink::monitors::count(void) const
{
	return _nmonitor;
}

void solids::lib::mf::sink::monitors::terminate_display_system(void)
{
	for (DWORD i = 0; i < _nmonitor; i++)
	{
		term_monitor_info(&_monitor[i]);
	}
	_nmonitor = 0;
}

solids::lib::mf::sink::monitors::monitors(void)
	: _nmonitor(0)
{
	::ZeroMemory(_monitor, sizeof(_monitor));
}

solids::lib::mf::sink::monitors::~monitors(void)
{
	terminate_display_system();
}