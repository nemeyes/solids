#include <mf_base.h>
#include "mf_mv_activate.h"
#include "mf_mv_class_factory.h"
#include "mf_mv_media_sink.h"

HMODULE g_module = NULL;
volatile long sld::mf::base::_obj_count = 0;
volatile long sld::mf::sink::video::mv_class_factory::_lock_count = 0;

STDAPI create_cap_mf_multiview_renderer(REFIID riid, void** ppvObject)
{
    return sld::mf::sink::video::mv_media_sink::create_instance(riid, ppvObject);
}

STDAPI create_cap_mf_multiview_renderer_activate(HWND hwnd, IMFActivate** ppActivate)
{
    return sld::mf::sink::video::activate::create_instance(hwnd, ppActivate);
}

STDAPI DllCanUnloadNow(void)
{
	return ((sld::mf::base::get_obj_count() == 0) && (sld::mf::sink::video::mv_class_factory::is_locked() == FALSE)) ? S_OK : S_FALSE;
}

STDAPI DllGetClassObject(_In_ REFCLSID clsid, _In_ REFIID iid, _Outptr_ void ** ppv)
{
	if (clsid != CLSID_MultiviewRenderer)
		return CLASS_E_CLASSNOTAVAILABLE;

	sld::mf::sink::video::mv_class_factory * factory = new sld::mf::sink::video::mv_class_factory();
	if (!factory)
		return E_OUTOFMEMORY;

	factory->AddRef();
	HRESULT hr = factory->QueryInterface(iid, ppv);

	sld::mf::safe_release(factory);

	return hr;
}
BOOL APIENTRY DllMain(HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
        case DLL_PROCESS_ATTACH:
        {
            g_module = static_cast<HMODULE>(hModule);
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
		hr = sld::mf::register_object(g_module, CLSID_MultiviewRenderer, L"Amadeus MF Multiview Renderer", L"Both");
		if (FAILED(hr)) break;

		hr = sld::mf::register_object(g_module, CLSID_MultiviewRendererActivate, L"Amadeus MF Multiview Renderer Activate", L"Both");
		if (FAILED(hr)) break;

		hr = MFTRegister(CLSID_MultiviewRenderer, MFT_CATEGORY_OTHER, L"Amadeus MF Multiview Renderer", 0, 0, NULL, 0, NULL, NULL);
	} while (FALSE);

	return hr;
}

STDAPI DllUnregisterServer(void)
{
	HRESULT hr = sld::mf::unregister_object(CLSID_MultiviewRenderer);
	HRESULT hr1 = MFTUnregister(CLSID_MultiviewRenderer);

	if (SUCCEEDED(hr))
		hr = hr1;

	return hr; 
}
