#include <mf_base.h>
#include "mf_d3d11_activate.h"
#include "mf_d3d11_class_factory.h"
#include "mf_d3d11_media.h"

HMODULE g_hModule = NULL;
volatile long sld::lib::mf::base::_obj_count = 0;
volatile long sld::lib::mf::sink::video::plain::factory::_lock_count = 0;

HRESULT CreateObjectKeyName(const GUID& guid, _Out_writes_(cchMax) TCHAR* pszName, DWORD cchMax)
{
    pszName[0] = _T('\0');

    // convert CLSID to string
    OLECHAR pszCLSID[CHARS_IN_GUID];
    HRESULT hr = StringFromGUID2(guid, pszCLSID, CHARS_IN_GUID);
    if (SUCCEEDED(hr))
    {
        // create a string of the form "CLSID\{clsid}"
        hr = StringCchPrintf(pszName, cchMax - 1, TEXT("CLSID\\%ls"), pszCLSID);
    }
    return hr;
}

HRESULT SetKeyValue(HKEY hKey, const TCHAR* pszName, const TCHAR* pszValue)
{
    size_t cch = 0;
    DWORD cbData = 0;
    HRESULT hr = StringCchLength(pszValue, MAXLONG, &cch);
    if (SUCCEEDED(hr))
    {
        cbData = (DWORD)(sizeof(TCHAR) * (cch + 1)); // add 1 for the NULL character
        hr = __HRESULT_FROM_WIN32(RegSetValueEx(hKey, pszName, 0, REG_SZ, reinterpret_cast<const BYTE*>(pszValue), cbData));
    }
    return hr;
}

HRESULT RegisterObject(GUID guid, const TCHAR* pszDescription, const TCHAR* pszThreadingModel)
{
    HRESULT hr = S_OK;
    TCHAR pszTemp[MAX_PATH];
    HKEY hKey = NULL;
    HKEY hSubkey = NULL;
    DWORD dwRet = 0;

    do
    {
        hr = CreateObjectKeyName(guid, pszTemp, MAX_PATH);
        if (FAILED(hr))
        {
            break;
        }

        hr = __HRESULT_FROM_WIN32(RegCreateKeyEx(HKEY_CLASSES_ROOT, pszTemp, 0, NULL, REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, NULL, &hKey, NULL));
        if (FAILED(hr))
        {
            break;
        }

        hr = SetKeyValue(hKey, NULL, pszDescription);
        if (FAILED(hr))
        {
            break;
        }

        hr = __HRESULT_FROM_WIN32(RegCreateKeyEx(hKey, L"InprocServer32", 0, NULL, REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, NULL, &hSubkey, NULL));
        if (FAILED(hr))
        {
            break;
        }

        dwRet = GetModuleFileName(g_hModule, pszTemp, MAX_PATH);
        if (dwRet == 0)
        {
            hr = __HRESULT_FROM_WIN32(GetLastError());
            break;
        }
        if (dwRet == MAX_PATH)
        {
            hr = E_FAIL; // buffer too small
            break;
        }

        hr = SetKeyValue(hSubkey, NULL, pszTemp);
        if (FAILED(hr))
        {
            break;
        }

        hr = SetKeyValue(hSubkey, L"ThreadingModel", pszThreadingModel);
    } while (FALSE);

    if (hSubkey != NULL)
    {
        RegCloseKey(hSubkey);
    }

    if (hKey != NULL)
    {
        RegCloseKey(hKey);
    }

    return hr;
}

HRESULT UnregisterObject(GUID guid)
{
    HRESULT hr = S_OK;
    TCHAR pszTemp[MAX_PATH];

    do
    {
        hr = CreateObjectKeyName(guid, pszTemp, MAX_PATH);
        if (FAILED(hr))
        {
            break;
        }

        hr = __HRESULT_FROM_WIN32(RegDeleteTree(HKEY_CLASSES_ROOT, pszTemp));
    } while (FALSE);

    return hr;
}

// DLL Exports

STDAPI CreateMFD3D11Renderer(REFIID riid, void** ppvObject)
{
    return sld::lib::mf::sink::video::plain::media::create_instance(riid, ppvObject);
}

STDAPI CreateMFD3D11RendererActivate(HWND hwnd, IMFActivate** ppActivate)
{
    return sld::lib::mf::sink::video::plain::activate::create_instance(hwnd, ppActivate);
}

STDAPI DllCanUnloadNow(void)
{
    return (sld::lib::mf::base::get_obj_count() == 0 && sld::lib::mf::sink::video::plain::factory::is_locked() == FALSE) ? S_OK : S_FALSE;
}

STDAPI DllGetClassObject(_In_ REFCLSID clsid, _In_ REFIID riid, _Outptr_ void** ppvObject)
{
    if (clsid != CLSID_MF_D3D11_RENDERER)
    {
        return CLASS_E_CLASSNOTAVAILABLE;
    }

    sld::lib::mf::sink::video::plain::factory * pFactory = new sld::lib::mf::sink::video::plain::factory();
    if (pFactory == NULL)
    {
        return E_OUTOFMEMORY;
    }

    pFactory->AddRef();

    HRESULT hr = pFactory->QueryInterface(riid, ppvObject);

    sld::lib::mf::safe_release(pFactory);
    
    return hr;
}

BOOL APIENTRY DllMain(HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    {
        g_hModule = static_cast<HMODULE>(hModule);
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
        hr = RegisterObject(CLSID_MF_D3D11_RENDERER, L"Pulsar MF D3D11 Renderer", L"Both");
        if (FAILED(hr))
        {
            break;
        }

        hr = RegisterObject(CLSID_MF_D3D11_RENDERER_ACTIVATE, L"Pulsar MF D3D11 Renderer Activate", L"Both");
        if (FAILED(hr))
        {
            break;
        }

        hr = MFTRegister(
            CLSID_MF_D3D11_RENDERER,    // CLSID
            MFT_CATEGORY_OTHER,         // Category
            (LPWSTR)L"Pulsar MF D3D11 Renderer",     // Friendly name
            0,                          // Reserved, must be zero.
            0,
            NULL,
            0,
            NULL,
            NULL
        );
    } while (FALSE);

    return hr;
}

STDAPI DllUnregisterServer(void)
{
    HRESULT hr = UnregisterObject(CLSID_MF_D3D11_RENDERER);
    HRESULT hrTemp = MFTUnregister(CLSID_MF_D3D11_RENDERER);
    if (SUCCEEDED(hr))
    {
        hr = hrTemp;
    }
    return hr;
}
