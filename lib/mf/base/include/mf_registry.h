
#ifndef _SLD_MF_REGISTRY_H_
#define _SLD_MF_REGISTRY_H_

namespace sld
{
	namespace lib
	{
		namespace mf
		{
			HRESULT inline create_registry_key(HKEY hKey, LPCTSTR subkey, HKEY* phKey)
			{
				assert(phKey != NULL);

				LONG lreturn = RegCreateKeyEx(
					hKey,                 // parent key
					subkey,               // name of subkey
					0,                    // reserved
					NULL,                 // class string (can be NULL)
					REG_OPTION_NON_VOLATILE,
					KEY_ALL_ACCESS,
					NULL,                 // security attributes
					phKey,
					NULL                  // receives the "disposition" (is it a new or existing key)
				);

				return HRESULT_FROM_WIN32(lreturn);
			}

			HRESULT inline create_object_keyname(const GUID& guid, _Out_writes_(cchMax) TCHAR* pszName, DWORD cchMax)
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

			HRESULT inline set_keyvalue(HKEY hKey, const TCHAR* pszName, const TCHAR* pszValue)
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

			HRESULT inline register_object(HMODULE module, GUID guid, const TCHAR* pszDescription, const TCHAR* pszThreadingModel)
			{
				HRESULT hr = S_OK;
				TCHAR pszTemp[MAX_PATH];
				HKEY hKey = NULL;
				HKEY hSubkey = NULL;
				DWORD dwRet = 0;

				do
				{
					hr = create_object_keyname(guid, pszTemp, MAX_PATH);
					if (FAILED(hr))
					{
						break;
					}

					hr = __HRESULT_FROM_WIN32(RegCreateKeyEx(HKEY_CLASSES_ROOT, pszTemp, 0, NULL, REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, NULL, &hKey, NULL));
					if (FAILED(hr))
					{
						break;
					}

					hr = set_keyvalue(hKey, NULL, pszDescription);
					if (FAILED(hr))
					{
						break;
					}

					hr = __HRESULT_FROM_WIN32(RegCreateKeyEx(hKey, L"InprocServer32", 0, NULL, REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, NULL, &hSubkey, NULL));
					if (FAILED(hr))
					{
						break;
					}

					dwRet = GetModuleFileName(module, pszTemp, MAX_PATH);
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

					hr = set_keyvalue(hSubkey, NULL, pszTemp);
					if (FAILED(hr))
					{
						break;
					}

					hr = set_keyvalue(hSubkey, L"ThreadingModel", pszThreadingModel);
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

			HRESULT inline unregister_object(GUID guid)
			{
				HRESULT hr = S_OK;
				TCHAR pszTemp[MAX_PATH];

				do
				{
					hr = create_object_keyname(guid, pszTemp, MAX_PATH);
					if (FAILED(hr))
					{
						break;
					}

					hr = __HRESULT_FROM_WIN32(RegDeleteTree(HKEY_CLASSES_ROOT, pszTemp));
				} while (FALSE);

				return hr;
			}

			static const TCHAR* REGKEY_MF_SCHEME_HANDLERS = TEXT("Software\\Microsoft\\Windows Media Foundation\\SchemeHandlers");
			inline HRESULT register_scheme_handler(const GUID& guid, const TCHAR* scheme_ext, const TCHAR* desc)
			{
				HRESULT hr = S_OK;
				HKEY    hKey = NULL;
				HKEY    hSubKey = NULL;
				OLECHAR szCLSID[CHARS_IN_GUID];
				size_t  cchDescription = 0;
				hr = StringCchLength(desc, STRSAFE_MAX_CCH, &cchDescription);

				if (SUCCEEDED(hr))
					hr = StringFromGUID2(guid, szCLSID, CHARS_IN_GUID);

				if (SUCCEEDED(hr))
					hr = create_registry_key(HKEY_LOCAL_MACHINE, REGKEY_MF_SCHEME_HANDLERS, &hKey);

				if (SUCCEEDED(hr))
					hr = create_registry_key(hKey, scheme_ext, &hSubKey);

				if (SUCCEEDED(hr))
					hr = RegSetValueEx(hSubKey, szCLSID, 0, REG_SZ, (BYTE*)desc, static_cast<DWORD>((cchDescription + 1) * sizeof(TCHAR)));

				if (hSubKey != NULL)
					RegCloseKey(hSubKey);

				if (hKey != NULL)
					RegCloseKey(hKey);

				return hr;
			}

			inline HRESULT unregister_scheme_handler(const GUID& guid, const TCHAR* scheme_ext)
			{
				TCHAR szKey[MAX_PATH];
				OLECHAR szCLSID[CHARS_IN_GUID];
				DWORD result = 0;
				HRESULT hr = S_OK;
				if (SUCCEEDED(hr = StringCchPrintf(szKey, MAX_PATH, TEXT("%s\\%s"), REGKEY_MF_SCHEME_HANDLERS, scheme_ext)))
				{
					if (SUCCEEDED(hr = StringFromGUID2(guid, szCLSID, CHARS_IN_GUID)))
					{
						result = RegDeleteKeyValue(HKEY_LOCAL_MACHINE, szKey, szCLSID);
						if (result != ERROR_SUCCESS)
						{
							hr = HRESULT_FROM_WIN32(result);
						}
					}
				}
				return hr;
			}

			static const TCHAR* REGKEY_MF_BYTESTREAM_HANDLERS = TEXT("Software\\Microsoft\\Windows Media Foundation\\ByteStreamHandlers");
			inline HRESULT register_bytestream_handler(const GUID& guid, const TCHAR* sFileExtension, const TCHAR* sDescription)
			{
				HRESULT hr = S_OK;
				HKEY hKey = NULL;
				HKEY hSubKey = NULL;

				OLECHAR szCLSID[CHARS_IN_GUID];

				size_t  cchDescription = 0;

				hr = StringCchLength(sDescription, STRSAFE_MAX_CCH, &cchDescription);

				if (SUCCEEDED(hr)) {
					hr = StringFromGUID2(guid, szCLSID, CHARS_IN_GUID);
				}

				if (SUCCEEDED(hr)) {
					hr = create_registry_key(HKEY_LOCAL_MACHINE, REGKEY_MF_BYTESTREAM_HANDLERS, &hKey);
				}

				if (SUCCEEDED(hr)) {
					hr = create_registry_key(hKey, sFileExtension, &hSubKey);
				}

				if (SUCCEEDED(hr)) {
					hr = RegSetValueEx(hSubKey, szCLSID, 0, REG_SZ, (BYTE*)sDescription, static_cast<DWORD>((cchDescription + 1) * sizeof(TCHAR)));
				}

				if (hSubKey != NULL) {
					RegCloseKey(hSubKey);
				}

				if (hKey != NULL) {
					RegCloseKey(hKey);
				}

				return hr;
			}

			inline HRESULT unregister_bytestream_handler(const GUID& guid, const TCHAR* sFileExtension)
			{
				TCHAR szKey[MAX_PATH];
				OLECHAR szCLSID[CHARS_IN_GUID];

				DWORD result = 0;
				HRESULT hr = S_OK;

				hr = StringCchPrintf(szKey, MAX_PATH, TEXT("%s\\%s"), REGKEY_MF_BYTESTREAM_HANDLERS, sFileExtension);

				if (SUCCEEDED(hr = StringFromGUID2(guid, szCLSID, CHARS_IN_GUID))) {

					result = RegDeleteKeyValue(HKEY_LOCAL_MACHINE, szKey, szCLSID);

					if (result != ERROR_SUCCESS) {
						hr = HRESULT_FROM_WIN32(result);
					}
				}
				return hr;
			}
		};
	};
};









#endif