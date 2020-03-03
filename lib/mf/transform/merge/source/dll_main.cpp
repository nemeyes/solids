
#include "mf_base.h"
#include "mf_class_factory.h"
#include "mf_merge_transform.h"
#include "mf_merge_transform_activate.h"

HMODULE g_module = NULL;
volatile long sld::lib::mf::base::_obj_count = 0;
volatile long sld::lib::mf::transform::merge::factory::_lock_count = 0;

STDAPI CreateMFMergeTransformActivate(IMFActivate ** activate)
{
	return sld::lib::mf::transform::merge::activate::create_instance(activate);
}

STDAPI DllCanUnloadNow(void)
{
	return ((sld::lib::mf::base::get_obj_count() == 0) && (sld::lib::mf::transform::merge::factory::is_locked() == FALSE)) ? S_OK : S_FALSE;
}

STDAPI DllGetClassObject(_In_ REFCLSID clsid, _In_ REFIID iid, _Outptr_ void ** ppv)
{
	if (clsid != CLSID_MF_MERGE_TRANSFORM)
		return CLASS_E_CLASSNOTAVAILABLE;

	sld::lib::mf::transform::merge::factory* fctr = new sld::lib::mf::transform::merge::factory();
	if (!fctr)
		return E_OUTOFMEMORY;

	//factory->AddRef();
	HRESULT hr = fctr->QueryInterface(iid, ppv);
	//amadeus::mf::safe_release(factory);
	return hr;
}

BOOL APIENTRY DllMain(HANDLE module, DWORD ul_reason_for_call, LPVOID reserved)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
	{
		g_module = static_cast<HMODULE>(module);
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
#if defined(_DEBUG)
		hr = sld::lib::mf::register_object(g_module, CLSID_MF_MERGE_TRANSFORM, TEXT("MFT Merge Debug"), TEXT("Both"));
		hr = sld::lib::mf::register_object(g_module, CLSID_MF_MERGE_TRANSFORM_ACTIVATE, TEXT("MFT Merge Debug Activate"), TEXT("Both"));
#else
		hr = sld::lib::mf::register_object(g_module, CLSID_MF_MERGE_TRANSFORM, TEXT("MFT Merge"), TEXT("Both"));
		hr = sld::lib::mf::register_object(g_module, CLSID_MF_MERGE_TRANSFORM_ACTIVATE, TEXT("MFT Merge Activate"), TEXT("Both"));
#endif
		hr = MFTRegister(
			CLSID_MF_MERGE_TRANSFORM,
			MFT_CATEGORY_MULTIPLEXER,
#if defined(_DEBUG)
			(LPWSTR)L"MFT Merge Debug",
#else
			(LPWSTR)L"MFT Merge",
#endif
			0, 0, NULL, 0, NULL, NULL);

	} while (FALSE);
	return hr;
}

STDAPI DllUnregisterServer(void)
{
	if(sld::lib::mf::unregister_object(CLSID_MF_MERGE_TRANSFORM)==S_OK)
		return MFTUnregister(CLSID_MF_MERGE_TRANSFORM);
	return S_FALSE;
}