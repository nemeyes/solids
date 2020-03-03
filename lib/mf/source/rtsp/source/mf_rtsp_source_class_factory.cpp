#include "mf_rtsp_source_class_factory.h"
#include "mf_rtsp_source_scheme_handler.h"

BOOL sld::lib::mf::source::rtsp::factory::is_locked(void)
{
	return (_lock_count == 0) ? FALSE : TRUE;
}

sld::lib::mf::source::rtsp::factory::factory(void)
	: _ref_count(0)
{}

sld::lib::mf::source::rtsp::factory::~factory(void)
{}


///////////////// IUnknown methods ///////////////////////
ULONG sld::lib::mf::source::rtsp::factory::AddRef(void)
{
	return ::InterlockedIncrement(&_ref_count);
}

ULONG sld::lib::mf::source::rtsp::factory::Release(void)
{
	ULONG count = ::InterlockedDecrement(&_ref_count);
	if (count == 0)
		delete this;
	return count;
}

HRESULT sld::lib::mf::source::rtsp::factory::QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void ** ppv)
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
///////////////// end of IUnknown methods ///////////////////////

///////////////// IClassFactory methods ///////////////////////
HRESULT sld::lib::mf::source::rtsp::factory::CreateInstance(_In_opt_ IUnknown * unk, _In_ REFIID riid, _COM_Outptr_ void ** ppv)
{
	if (ppv == NULL)
	{
		return E_POINTER;
	}

	*ppv = NULL;
	if (unk != NULL)
	{
		return CLASS_E_NOAGGREGATION;
	}

	return sld::lib::mf::source::rtsp::handler::create_instance(unk, riid, ppv);
}

HRESULT sld::lib::mf::source::rtsp::factory::LockServer(BOOL lock)
{
	if (lock == FALSE)
	{
		::InterlockedDecrement(&_lock_count);
	}
	else
	{
		::InterlockedIncrement(&_lock_count);
	}
	return S_OK;
}
///////////////// end of IClassFactory methods ///////////////////////