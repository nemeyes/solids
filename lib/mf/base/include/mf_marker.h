#ifndef _SLD_MF_MARKER_H_
#define _SLD_MF_MARKER_H_

namespace solids
{
	namespace lib
	{
		namespace mf
		{
			MIDL_INTERFACE("FE202287-FFC9-4793-8B75-22C9469033D3")
				IMarker : public IUnknown
			{
				virtual STDMETHODIMP GetMarkerType(MFSTREAMSINK_MARKER_TYPE * pType) = 0;
				virtual STDMETHODIMP GetMarkerValue(PROPVARIANT * pvar) = 0;
				virtual STDMETHODIMP GetContext(PROPVARIANT * pvar) = 0;
			};

			// Holds marker information for IMFStreamSink::PlaceMarker
			class marker
				: base
				, refcount_object
				, public IMarker
			{
			public:

				static HRESULT create(MFSTREAMSINK_MARKER_TYPE mt, const PROPVARIANT * mv, const PROPVARIANT * cv, IMarker ** ppmarker)
				{
					if (ppmarker == NULL)
						return E_POINTER;

					HRESULT hr = S_OK;
					solids::lib::mf::marker * pmarker = new solids::lib::mf::marker(mt);

					if (pmarker == NULL)
						hr = E_OUTOFMEMORY;

					// Copy the marker data.
					if (SUCCEEDED(hr))
					{
						if (mv)
						{
							hr = PropVariantCopy(&pmarker->_mv, mv);
						}
					}

					if (SUCCEEDED(hr))
					{
						if (cv)
						{
							hr = PropVariantCopy(&pmarker->_cv, cv);
						}
					}

					if (SUCCEEDED(hr))
					{
						*ppmarker = pmarker;
						(*ppmarker)->AddRef();
					}

					safe_release(pmarker);
					return hr;
				}

				// IUnknown methods.
				STDMETHODIMP_(ULONG) AddRef(void)
				{
					return refcount_object::AddRef();
				}

				STDMETHODIMP_(ULONG) Release(void)
				{
					return refcount_object::Release();
				}

				STDMETHODIMP QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void ** ppv)
				{
					if (!ppv)
						return E_POINTER;

					if (iid == IID_IUnknown)
						*ppv = static_cast<IUnknown*>(this);
					else if (iid == __uuidof(IMarker))
						*ppv = static_cast<IMarker*>(this);
					else
					{
						*ppv = NULL;
						return E_NOINTERFACE;
					}

					AddRef();
					return S_OK;
				}

				STDMETHODIMP GetMarkerType(MFSTREAMSINK_MARKER_TYPE * mt)
				{
					if (mt == NULL)
						return E_POINTER;

					*mt = _mt;
					return S_OK;
				}

				STDMETHODIMP GetMarkerValue(PROPVARIANT * mv)
				{
					if (mv == NULL)
						return E_POINTER;
					return PropVariantCopy(mv, &_mv);

				}

				STDMETHODIMP GetContext(PROPVARIANT * cv)
				{
					if (cv == NULL)
						return E_POINTER;
					return PropVariantCopy(cv, &_cv);
				}

			protected:
				MFSTREAMSINK_MARKER_TYPE _mt;
				PROPVARIANT _mv;
				PROPVARIANT _cv;

			private:
				marker(MFSTREAMSINK_MARKER_TYPE mt)
					: _mt(mt)
				{
					PropVariantInit(&_mv);
					PropVariantInit(&_cv);
				}

				virtual ~marker(void)
				{
					assert(_ref_count == 0);
					PropVariantClear(&_mv);
					PropVariantClear(&_cv);
				}
			};
		};
	};
};

#endif