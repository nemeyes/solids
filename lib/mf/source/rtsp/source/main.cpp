#include <mf_base.h>

#include "mf_rtsp_source_class_factory.h"

// {A756D7C8-E1B6-4848-B9C8-E4A936E3A8D6}
DEFINE_GUID(CLSID_PULSARMFRTSPSource,
	0xa756d7c8, 0xe1b6, 0x4848, 0xb9, 0xc8, 0xe4, 0xa9, 0x36, 0xe3, 0xa8, 0xd6);

HMODULE gModule = NULL;
volatile long sld::lib::mf::base::_obj_count = 0;
volatile long sld::lib::mf::source::rtsp::factory::_lock_count = 0;

const TCHAR * gSchemeHandlerDesc = TEXT("Pulsar RTSP Soure Scheme Handler");
const TCHAR * gSchemeExt = TEXT("plsrtsp:");

STDAPI DllCanUnloadNow(void)
{
	return ((sld::lib::mf::base::get_obj_count() == 0) && (sld::lib::mf::source::rtsp::factory::is_locked() == FALSE)) ? S_OK : S_FALSE;
}

STDAPI DllGetClassObject(_In_ REFCLSID clsid, _In_ REFIID iid, _Outptr_ void ** ppv)
{
	if (clsid != CLSID_PULSARMFRTSPSource)
		return CLASS_E_CLASSNOTAVAILABLE;

	sld::lib::mf::source::rtsp::factory* fctr = new sld::lib::mf::source::rtsp::factory();
	if (!fctr)
		return E_OUTOFMEMORY;

	fctr->AddRef();
	HRESULT hr = fctr->QueryInterface(iid, ppv);

	sld::lib::mf::safe_release(fctr);

	return hr;
}

BOOL APIENTRY DllMain(HANDLE module, DWORD reason, LPVOID reserved)
{
	switch (reason)
	{
	case DLL_PROCESS_ATTACH:
	{
		gModule = static_cast<HMODULE>(module);
		break;
	}
	default:
	{
		break;
	}
	}
	return TRUE;
}

STDAPI DllRegisterServer(void)
{
	HRESULT hr = S_OK;
	do
	{
		hr = sld::lib::mf::register_object(gModule, CLSID_PULSARMFRTSPSource, L"Pulsar MF RTSP Source", L"Both");
		if (FAILED(hr))
			break;
		hr = sld::lib::mf::register_scheme_handler(CLSID_PULSARMFRTSPSource, gSchemeExt, gSchemeHandlerDesc);
	} while (FALSE);
	return hr;
}

STDAPI DllUnregisterServer(void)
{
	HRESULT hr = S_OK;
	do
	{
		hr = sld::lib::mf::unregister_object(CLSID_PULSARMFRTSPSource);
		if (FAILED(hr))
			break;
		hr = sld::lib::mf::unregister_scheme_handler(CLSID_PULSARMFRTSPSource, gSchemeExt);
	} while (FALSE);
	return hr;
}