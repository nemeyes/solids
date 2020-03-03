#include "mf_merge_transform_activate.h"

HRESULT sld::lib::mf::transform::merge::activate::create_instance(IMFActivate ** activate)
{
	if (activate == NULL)
		return E_POINTER;

	sld::lib::mf::transform::merge::activate * act = new sld::lib::mf::transform::merge::activate();
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

	} while (FALSE);

	safe_release(act);

	return hr;
}

///////////////// IUnknown methods//////////////////////
ULONG sld::lib::mf::transform::merge::activate::AddRef(void)
{
	return sld::lib::mf::refcount_object::AddRef();
}

ULONG sld::lib::mf::transform::merge::activate::Release(void)
{
	return sld::lib::mf::refcount_object::Release();
}

HRESULT sld::lib::mf::transform::merge::activate::QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void ** ppv)
{
	if (!ppv)
		return E_POINTER;
	
	if (iid == IID_IUnknown)
		*ppv = static_cast<IUnknown*>(static_cast<IMFActivate*>(this));
	else if (iid == __uuidof(IMFActivate))
		*ppv = static_cast<IMFActivate*>(this);
	else if (iid == __uuidof(IPersistStream))
		*ppv = static_cast<IPersistStream*>(this);
	else if (iid == __uuidof(IPersist))
		*ppv = static_cast<IPersist*>(this);
	else if (iid == __uuidof(IMFAttributes))
		*ppv = static_cast<IMFAttributes*>(this);
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
HRESULT sld::lib::mf::transform::merge::activate::ActivateObject(__RPC__in REFIID iid, __RPC__deref_out_opt void ** ppv)
{
	HRESULT hr = S_OK;
	
	do
	{
		if (_transform == NULL)
		{
			hr = sld::lib::mf::transform::merge::transform::CreateInstance(__uuidof(_transform), (void**)&_transform);
			if (FAILED(hr))
				break;
		}

		hr = _transform->QueryInterface(iid, ppv);
	} while (FALSE);

	return hr;
}

HRESULT sld::lib::mf::transform::merge::activate::DetachObject(void)
{
	sld::lib::mf::safe_release(_transform);
	return S_OK;
}

HRESULT sld::lib::mf::transform::merge::activate::ShutdownObject(void)
{
	if (_transform != NULL)
		sld::lib::mf::safe_release(_transform);
	return S_OK;
}
///////////////// end of IMFActivate methods//////////////////////

///////////////// IPersistStream methods//////////////////////
HRESULT sld::lib::mf::transform::merge::activate::GetSizeMax(__RPC__out ULARGE_INTEGER * size)
{
	return E_NOTIMPL;
}

HRESULT sld::lib::mf::transform::merge::activate::IsDirty(void)
{
	return E_NOTIMPL;
}

HRESULT sld::lib::mf::transform::merge::activate::Load(__RPC__in_opt IStream * stream)
{
	return E_NOTIMPL;
}

HRESULT sld::lib::mf::transform::merge::activate::Save(__RPC__in_opt IStream * stream, BOOL cleardirty)
{
	return E_NOTIMPL;
}
///////////////// end of IPersistStream methods//////////////////////

///////////////// IPersist methods//////////////////////
HRESULT sld::lib::mf::transform::merge::activate::GetClassID(__RPC__out CLSID * clsid)
{
	if (clsid == NULL)
		return E_POINTER;

	*clsid = CLSID_MF_MERGE_TRANSFORM_ACTIVATE;
	return S_OK;
}
///////////////// end of IPersist methods//////////////////////

sld::lib::mf::transform::merge::activate::activate(void)
	: _transform(NULL)
{}

sld::lib::mf::transform::merge::activate::~activate(void)
{
	sld::lib::mf::safe_release(_transform);
}