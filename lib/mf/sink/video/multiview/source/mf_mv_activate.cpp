#include "mf_mv_activate.h"
#include "mf_mv_media_sink.h"

HRESULT solids::lib::mf::sink::video::multiview::activate::create_instance(HWND hwnd, IMFActivate ** activate)
{
	if (activate == NULL)
		return E_POINTER;

	solids::lib::mf::sink::video::multiview::activate * act = new solids::lib::mf::sink::video::multiview::activate();
	if (act == NULL)
		return E_OUTOFMEMORY;

	//act->AddRef();

	HRESULT hr = S_OK;
	do
	{
		hr = act->initialize();
		if (FAILED(hr))
			break;

		hr = act->QueryInterface(IID_PPV_ARGS(activate));
		if (FAILED(hr))
			break;

		act->_hwnd = hwnd;
	} while (FALSE);

	solids::lib::mf::safe_release(act);

	return hr;
}

///////////////// IUnknown methods//////////////////////
ULONG solids::lib::mf::sink::video::multiview::activate::AddRef(void)
{
	return solids::lib::mf::refcount_object::AddRef();
}

ULONG solids::lib::mf::sink::video::multiview::activate::Release(void)
{
	return solids::lib::mf::refcount_object::Release();
}

HRESULT solids::lib::mf::sink::video::multiview::activate::QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void** ppv)
{
	if (!ppv)
	{
		return E_POINTER;
	}
	if (iid == IID_IUnknown)
	{
		*ppv = static_cast<IUnknown*>(static_cast<IMFActivate*>(this));
	}
	else if (iid == __uuidof(IMFActivate))
	{
		*ppv = static_cast<IMFActivate*>(this);
	}
	else if (iid == __uuidof(IPersistStream))
	{
		*ppv = static_cast<IPersistStream*>(this);
	}
	else if (iid == __uuidof(IPersist))
	{
		*ppv = static_cast<IPersist*>(this);
	}
	else if (iid == __uuidof(IMFAttributes))
	{
		*ppv = static_cast<IMFAttributes*>(this);
	}
	else
	{
		*ppv = NULL;
		return E_NOINTERFACE;
	}
	AddRef();
	return S_OK;
}
///////////////// end of IUnknown methods//////////////////////

///////////////// IMFActivate methods//////////////////////
HRESULT solids::lib::mf::sink::video::multiview::activate::ActivateObject(__RPC__in REFIID iid, __RPC__deref_out_opt void ** ppv)
{
	HRESULT hr = S_OK;
	IMFGetService * mfGetSvc = NULL;
	IMFVideoDisplayControl * mfVideoDisplayCtrl = NULL;

	do
	{
		if (_media_sink == NULL)
		{
			hr = solids::lib::mf::sink::video::multiview::media::create_instance(IID_PPV_ARGS(&_media_sink));
			if (FAILED(hr)) break;

			hr = _media_sink->QueryInterface(IID_PPV_ARGS(&mfGetSvc));
			if (FAILED(hr)) break;

			hr = mfGetSvc->GetService(MR_VIDEO_RENDER_SERVICE, IID_PPV_ARGS(&mfVideoDisplayCtrl));
			if (FAILED(hr)) break;

			hr = mfVideoDisplayCtrl->SetVideoWindow(_hwnd);
			if (FAILED(hr)) break;
		}

		hr = _media_sink->QueryInterface(iid, ppv);
		if (FAILED(hr)) break;

	} while (FALSE);

	solids::lib::mf::safe_release(mfGetSvc);
	solids::lib::mf::safe_release(mfVideoDisplayCtrl);

	return hr;
}

HRESULT solids::lib::mf::sink::video::multiview::activate::DetachObject(void)
{
	solids::lib::mf::safe_release(_media_sink);
	return S_OK;
}

HRESULT solids::lib::mf::sink::video::multiview::activate::ShutdownObject(void)
{
	if (_media_sink)
	{
		_media_sink->Shutdown();
		solids::lib::mf::safe_release(_media_sink);
	}
	return S_OK;
}
///////////////// end of IMFActivate methods//////////////////////

///////////////// IPersistStream methods//////////////////////
HRESULT solids::lib::mf::sink::video::multiview::activate::GetSizeMax(__RPC__out ULARGE_INTEGER * size)
{
	return E_NOTIMPL;
}

HRESULT solids::lib::mf::sink::video::multiview::activate::IsDirty(void)
{
	return E_NOTIMPL;
}

HRESULT solids::lib::mf::sink::video::multiview::activate::Load(__RPC__in_opt IStream * stream)
{
	return E_NOTIMPL;
}

HRESULT solids::lib::mf::sink::video::multiview::activate::Save(__RPC__in_opt IStream * stream, BOOL clear_dirty)
{
	return E_NOTIMPL;
}
///////////////// end of IPersistStream methods//////////////////////

///////////////// IPersist methods//////////////////////
HRESULT solids::lib::mf::sink::video::multiview::activate::GetClassID(__RPC__out CLSID * clsid)
{
	if (!clsid)
		return E_POINTER;

	*clsid = CLSID_MF_MULTIVIEW_RENDERER_ACTIVATE;
	return S_OK;
}
///////////////// end of IPersist methods//////////////////////

solids::lib::mf::sink::video::multiview::activate::activate(void)
	: _media_sink(NULL)
	, _hwnd(NULL)
{}

solids::lib::mf::sink::video::multiview::activate::~activate(void)
{
	solids::lib::mf::safe_release(_media_sink);
}