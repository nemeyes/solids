#include "mf_rtsp_source_scheme_handler.h"
#include "mf_rtsp_source.h"

sld::lib::mf::source::rtsp::handler::handler(void)
	: _result(NULL)
	, _source(NULL)
{}

sld::lib::mf::source::rtsp::handler::~handler(void)
{
	sld::lib::mf::safe_release(_source);
	sld::lib::mf::safe_release(_result);
}

HRESULT sld::lib::mf::source::rtsp::handler::create_instance(IUnknown * unk, REFIID iid, void ** ppv)
{
	HRESULT hr = S_OK;
	if (!ppv)
		return E_POINTER;
	if (unk)
		return CLASS_E_NOAGGREGATION;

	sld::lib::mf::source::rtsp::handler* handler = NULL;
	do
	{
		handler = new sld::lib::mf::source::rtsp::handler();
		if (!handler)
		{
			hr = E_OUTOFMEMORY;
			break;
		}

		hr = handler->QueryInterface(iid, ppv);
		if (FAILED(hr))
			break;

	} while (FALSE);

	safe_release(handler);
	return hr;
}

// IUnknown
HRESULT sld::lib::mf::source::rtsp::handler::QueryInterface(REFIID iid, void ** ppv)
{
	if (!ppv)
		return E_POINTER;

	if (iid == IID_IUnknown)
		*ppv = static_cast<IUnknown*>(static_cast<IMFSchemeHandler*>(this));
	else if (iid == __uuidof(IMFSchemeHandler))
		*ppv = static_cast<IMFSchemeHandler*>(this);
	else if (iid == __uuidof(IMFAsyncCallback))
		*ppv = static_cast<IMFAsyncCallback*>(this);
	else
	{
		*ppv = NULL;
		return E_NOINTERFACE;
	}
	AddRef();
	return S_OK;
}

ULONG sld::lib::mf::source::rtsp::handler::AddRef(void)
{
	return sld::lib::mf::refcount_object::AddRef();
}

ULONG sld::lib::mf::source::rtsp::handler::Release(void)
{
	return sld::lib::mf::refcount_object::Release();
}

// IMFSchemeHandler
HRESULT sld::lib::mf::source::rtsp::handler::BeginCreateObject(LPCWSTR url, DWORD flags, IPropertyStore *, IUnknown ** unk_cancel_cookie, IMFAsyncCallback * callback, IUnknown * state)
{
	HRESULT hr = S_OK;

	if (!url)
		return E_POINTER;
	if (!callback)
		return E_POINTER;
	if ((flags & MF_RESOLUTION_MEDIASOURCE)==0)
		return E_INVALIDARG;

	IMFAsyncResult * result = NULL;
	sld::lib::mf::source::rtsp::source * source = NULL;
	do
	{
		if (unk_cancel_cookie)
			*unk_cancel_cookie = NULL;

		sld::lib::mf::source::rtsp::source::create_instance(&source);
		hr = source->initialize();
		if (FAILED(hr))
			break;

		hr = MFCreateAsyncResult(NULL, callback, state, &result);
		if (FAILED(hr))
			break;

		source->begin_open(url + 3, this, NULL); //skip rtsp prefix(eartsp > rtsp)

		_result = result;
		_source = source;

		_result->AddRef();
		_source->AddRef();

	} while (FALSE);

	sld::lib::mf::safe_release(result);
	sld::lib::mf::safe_release(source);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::handler::CancelObjectCreation(IUnknown * unk)
{
	return E_NOTIMPL;
}

HRESULT sld::lib::mf::source::rtsp::handler::EndCreateObject(IMFAsyncResult * result, MF_OBJECT_TYPE * ot, IUnknown ** ppv)
{
	HRESULT hr = S_OK;
	if (!result)
		return E_POINTER;
	if (!ot)
		return E_POINTER;
	if (!ppv)
		return E_POINTER;

	
	*ot = MF_OBJECT_INVALID;
	*ppv = NULL;
	do 
	{
		hr = result->GetStatus();
		if (FAILED(hr))
			break;

		*ot = MF_OBJECT_MEDIASOURCE;
		_source->QueryInterface(IID_PPV_ARGS(ppv));
		if (FAILED(hr))
			break;

	} while (FALSE);

	sld::lib::mf::safe_release(_source);
	sld::lib::mf::safe_release(_result);

	return hr;
}

HRESULT sld::lib::mf::source::rtsp::handler::Invoke(IMFAsyncResult * result)
{
	HRESULT hr = S_OK;

	if (_source) 
	{
		hr = _source->end_open(result);
	}
	else 
	{
		hr = E_UNEXPECTED;
	}

	_result->SetStatus(hr);

	hr = MFInvokeCallback(_result);

	return hr;
}