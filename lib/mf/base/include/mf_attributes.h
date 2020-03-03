#ifndef _MF_ATTRIBUTES_H_
#define _MF_ATTRIBUTES_H_

namespace solids
{
	namespace lib
	{
		namespace mf
		{
			template <class TBase = IMFAttributes>
			class attributes : public TBase
			{
			protected:
				attributes(void)
					: _attr(NULL)
				{}

				attributes(HRESULT& hr, UINT32 initial_size = 0)
					: _attr(NULL)
				{
					hr = initialize(initial_size);
				}

				attributes(HRESULT& hr, IUnknown* unk)
					: _attr(NULL)
				{
					hr = initialize(unk);
				}

				virtual ~attributes(void)
				{
					if (_attr)
						_attr->Release();
				}

				HRESULT initialize(UINT32 initial_size = 0)
				{
					if (_attr == NULL)
						return MFCreateAttributes(&_attr, initial_size);
					else
						return S_OK;
				}

			public:

				// IMFAttributes methods

				STDMETHODIMP GetItem(__RPC__in REFGUID guidKey, __RPC__inout_opt PROPVARIANT* pValue)
				{
					assert(_attr);
					return _attr->GetItem(guidKey, pValue);
				}

				STDMETHODIMP GetItemType(__RPC__in REFGUID guidKey, __RPC__out MF_ATTRIBUTE_TYPE* pType)
				{
					assert(_attr);
					return _attr->GetItemType(guidKey, pType);
				}

				STDMETHODIMP CompareItem(__RPC__in REFGUID guidKey, __RPC__in REFPROPVARIANT Value, __RPC__out BOOL* pbResult)
				{
					assert(_attr);
					return _attr->CompareItem(guidKey, Value, pbResult);
				}

				STDMETHODIMP Compare(__RPC__in_opt IMFAttributes* pTheirs, MF_ATTRIBUTES_MATCH_TYPE MatchType, __RPC__out BOOL* pbResult)
				{
					assert(_attr);
					return _attr->Compare(pTheirs, MatchType, pbResult);
				}

				STDMETHODIMP GetUINT32(__RPC__in REFGUID guidKey, __RPC__out UINT32* punValue)
				{
					assert(_attr);
					return _attr->GetUINT32(guidKey, punValue);
				}

				STDMETHODIMP GetUINT64(__RPC__in REFGUID guidKey, __RPC__out UINT64* punValue)
				{
					assert(_attr);
					return _attr->GetUINT64(guidKey, punValue);
				}

				STDMETHODIMP GetDouble(__RPC__in REFGUID guidKey, __RPC__out double* pfValue)
				{
					assert(_attr);
					return _attr->GetDouble(guidKey, pfValue);
				}

				STDMETHODIMP GetGUID(__RPC__in REFGUID guidKey, __RPC__out GUID* pguidValue)
				{
					assert(_attr);
					return _attr->GetGUID(guidKey, pguidValue);
				}

				STDMETHODIMP GetStringLength(__RPC__in REFGUID guidKey, __RPC__out UINT32* pcchLength)
				{
					assert(_attr);
					return _attr->GetStringLength(guidKey, pcchLength);
				}

				STDMETHODIMP GetString(__RPC__in REFGUID guidKey, __RPC__out_ecount_full(cchBufSize) LPWSTR pwszValue, UINT32 cchBufSize, __RPC__inout_opt UINT32* pcchLength)
				{
					assert(_attr);
					return _attr->GetString(guidKey, pwszValue, cchBufSize, pcchLength);
				}

				STDMETHODIMP GetAllocatedString(__RPC__in REFGUID guidKey, __RPC__deref_out_ecount_full_opt((*pcchLength + 1)) LPWSTR* ppwszValue, __RPC__out UINT32* pcchLength)
				{
					assert(_attr);
					return _attr->GetAllocatedString(guidKey, ppwszValue, pcchLength);
				}

				STDMETHODIMP GetBlobSize(__RPC__in REFGUID guidKey, __RPC__out UINT32* pcbBlobSize)
				{
					assert(_attr);
					return _attr->GetBlobSize(guidKey, pcbBlobSize);
				}

				STDMETHODIMP GetBlob(__RPC__in REFGUID guidKey, __RPC__out_ecount_full(cbBufSize) UINT8* pBuf, UINT32 cbBufSize, __RPC__inout_opt UINT32* pcbBlobSize)
				{
					assert(_attr);
					return _attr->GetBlob(guidKey, pBuf, cbBufSize, pcbBlobSize);
				}

				STDMETHODIMP GetAllocatedBlob(__RPC__in REFGUID guidKey, __RPC__deref_out_ecount_full_opt(*pcbSize) UINT8** ppBuf, __RPC__out UINT32* pcbSize)
				{
					assert(_attr);
					return _attr->GetAllocatedBlob(guidKey, ppBuf, pcbSize);
				}

				STDMETHODIMP GetUnknown(__RPC__in REFGUID guidKey, __RPC__in REFIID riid, __RPC__deref_out_opt LPVOID* ppv)
				{
					assert(_attr);
					return _attr->GetUnknown(guidKey, riid, ppv);
				}

				STDMETHODIMP SetItem(__RPC__in REFGUID guidKey, __RPC__in REFPROPVARIANT Value)
				{
					assert(_attr);
					return _attr->SetItem(guidKey, Value);
				}

				STDMETHODIMP DeleteItem(__RPC__in REFGUID guidKey)
				{
					assert(_attr);
					return _attr->DeleteItem(guidKey);
				}

				STDMETHODIMP DeleteAllItems(void)
				{
					assert(_attr);
					return _attr->DeleteAllItems();
				}

				STDMETHODIMP SetUINT32(__RPC__in REFGUID guidKey, UINT32 unValue)
				{
					assert(_attr);
					return _attr->SetUINT32(guidKey, unValue);
				}

				STDMETHODIMP SetUINT64(__RPC__in REFGUID guidKey, UINT64 unValue)
				{
					assert(_attr);
					return _attr->SetUINT64(guidKey, unValue);
				}

				STDMETHODIMP SetDouble(__RPC__in REFGUID guidKey, double fValue)
				{
					assert(_attr);
					return _attr->SetDouble(guidKey, fValue);
				}

				STDMETHODIMP SetGUID(__RPC__in REFGUID guidKey, __RPC__in REFGUID guidValue)
				{
					assert(_attr);
					return _attr->SetGUID(guidKey, guidValue);
				}

				STDMETHODIMP SetString(__RPC__in REFGUID guidKey, __RPC__in_string LPCWSTR wszValue)
				{
					assert(_attr);
					return _attr->SetString(guidKey, wszValue);
				}

				STDMETHODIMP SetBlob(__RPC__in REFGUID guidKey, __RPC__in_ecount_full(cbBufSize) const UINT8* pBuf, UINT32 cbBufSize)
				{
					assert(_attr);
					return _attr->SetBlob(guidKey, pBuf, cbBufSize);
				}

				STDMETHODIMP SetUnknown(__RPC__in REFGUID guidKey, __RPC__in_opt IUnknown* pUnknown)
				{
					assert(_attr);
					return _attr->SetUnknown(guidKey, pUnknown);
				}

				STDMETHODIMP LockStore(void)
				{
					assert(_attr);
					return _attr->LockStore();
				}

				STDMETHODIMP UnlockStore(void)
				{
					assert(_attr);
					return _attr->UnlockStore();
				}

				STDMETHODIMP GetCount(__RPC__out UINT32* pcItems)
				{
					assert(_attr);
					return _attr->GetCount(pcItems);
				}

				STDMETHODIMP GetItemByIndex(UINT32 unIndex, __RPC__out GUID* pguidKey, __RPC__inout_opt PROPVARIANT* pValue)
				{
					assert(_attr);
					return _attr->GetItemByIndex(unIndex, pguidKey, pValue);
				}

				STDMETHODIMP CopyAllItems(__RPC__in_opt IMFAttributes* pDest)
				{
					assert(_attr);
					return _attr->CopyAllItems(pDest);
				}

				// Helper functions

				HRESULT SerializeToStream(DWORD dwOptions, IStream* pStm)
					// dwOptions: Flags from MF_ATTRIBUTE_SERIALIZE_OPTIONS
				{
					assert(_attr);
					return MFSerializeAttributesToStream(_attr, dwOptions, pStm);
				}

				HRESULT DeserializeFromStream(DWORD dwOptions, IStream* pStm)
				{
					assert(_attr);
					return MFDeserializeAttributesFromStream(_attr, dwOptions, pStm);
				}

				// SerializeToBlob: Stores the attributes in a byte array.
				//
				// ppBuf: Receives a pointer to the byte array.
				// pcbSize: Receives the size of the byte array.
				//
				// The caller must free the array using CoTaskMemFree.
				HRESULT SerializeToBlob(UINT8** ppBuffer, UINT* pcbSize)
				{
					assert(_attr);

					if (ppBuffer == NULL)
					{
						return E_POINTER;
					}
					if (pcbSize == NULL)
					{
						return E_POINTER;
					}

					HRESULT hr = S_OK;
					UINT32 cbSize = 0;
					BYTE* pBuffer = NULL;

					hr = MFGetAttributesAsBlobSize(_attr, &cbSize);
					if (FAILED(hr))
					{
						cbSize = 0;
						return hr;
					}

					pBuffer = (BYTE*)CoTaskMemAlloc(cbSize);
					if (pBuffer == NULL)
					{
						return E_OUTOFMEMORY;
					}

					hr = MFGetAttributesAsBlob(_attr, pBuffer, cbSize);
					if (FAILED(hr))
					{
						*ppBuffer = NULL;
						pcbSize = 0;
						return hr;
					}

					*ppBuffer = pBuffer;
					*pcbSize = cbSize;

					if (FAILED(hr))
					{
						*ppBuffer = NULL;
						*pcbSize = 0;
						CoTaskMemFree(pBuffer);
					}
					return hr;
				}

				HRESULT DeserializeFromBlob(const UINT8* pBuffer, UINT cbSize)
				{
					assert(_attr);
					return MFInitAttributesFromBlob(_attr, pBuffer, cbSize);
				}

				HRESULT GetRatio(REFGUID guidKey, UINT32* pnNumerator, UINT32* punDenominator)
				{
					assert(_attr);
					return MFGetAttributeRatio(_attr, guidKey, pnNumerator, punDenominator);
				}

				HRESULT SetRatio(REFGUID guidKey, UINT32 unNumerator, UINT32 unDenominator)
				{
					assert(_attr);
					return MFSetAttributeRatio(_attr, guidKey, unNumerator, unDenominator);
				}

				// Gets an attribute whose value represents the size of something (eg a video frame).
				HRESULT GetSize(REFGUID guidKey, UINT32* punWidth, UINT32* punHeight)
				{
					assert(_attr);
					return MFGetAttributeSize(_attr, guidKey, punWidth, punHeight);
				}

				// Sets an attribute whose value represents the size of something (eg a video frame).
				HRESULT SetSize(REFGUID guidKey, UINT32 unWidth, UINT32 unHeight)
				{
					assert(_attr);
					return MFSetAttributeSize(_attr, guidKey, unWidth, unHeight);
				}

			protected:

				IMFAttributes* _attr;
			};
		}
	}
}





#endif
