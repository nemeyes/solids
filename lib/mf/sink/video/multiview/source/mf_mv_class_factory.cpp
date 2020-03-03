#include "mf_mv_class_factory.h"
#include "mf_mv_media_sink.h"

BOOL sld::lib::mf::sink::video::multiview::factory::is_locked(void)
{
	return (_lock_count == 0) ? FALSE : TRUE;
}

sld::lib::mf::sink::video::multiview::factory::factory(void)
{

}

sld::lib::mf::sink::video::multiview::factory::~factory(void)
{

}

ULONG sld::lib::mf::sink::video::multiview::factory::AddRef(void)
{
	return sld::lib::mf::refcount_object::AddRef();
}

ULONG sld::lib::mf::sink::video::multiview::factory::Release(void)
{
	return sld::lib::mf::refcount_object::Release();
}

HRESULT sld::lib::mf::sink::video::multiview::factory::QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void ** ppv)
{
	if (!ppv)
		return E_POINTER;

	if (iid == IID_IUnknown)
		*ppv = static_cast<IUnknown*>(static_cast<IClassFactory*>(this));
	else if (iid == __uuidof(IClassFactory))
		*ppv = static_cast<IClassFactory*>(this);
	else
	{
		*ppv = NULL;
		return E_NOINTERFACE;
	}
	AddRef();
	return S_OK;
}

HRESULT sld::lib::mf::sink::video::multiview::factory::CreateInstance(_In_opt_ IUnknown * unk, _In_ REFIID iid, _COM_Outptr_ void ** ppv)
{
	if (!ppv)
		return E_POINTER;

	*ppv = NULL;
	if (unk != NULL)
	{
		return CLASS_E_NOAGGREGATION;
	}

	return sld::lib::mf::sink::video::multiview::media::create_instance(iid, ppv);
}

HRESULT sld::lib::mf::sink::video::multiview::factory::LockServer(BOOL lock)
{
	if (lock == FALSE)
		::InterlockedDecrement(&_lock_count);
	else
		::InterlockedIncrement(&_lock_count);
	return S_OK;
}
