#include "mf_class_factory.h"
#include "mf_merge_transform.h"

BOOL solids::lib::mf::transform::merge::factory::is_locked(void)
{
	return (_lock_count == 0) ? FALSE : TRUE;
}

solids::lib::mf::transform::merge::factory::factory(void)
	: _ref_count(1)
{

}

solids::lib::mf::transform::merge::factory::~factory(void)
{

}

///////////////// IUnknown methods ///////////////////////
ULONG solids::lib::mf::transform::merge::factory::AddRef(void)
{
	return ::InterlockedIncrement(&_ref_count);
}

ULONG solids::lib::mf::transform::merge::factory::Release(void)
{
	assert(_ref_count >= 0);
	ULONG count = ::InterlockedDecrement(&_ref_count);
	if (count == 0)
		delete this;
	return count;
}

HRESULT solids::lib::mf::transform::merge::factory::QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void ** ppv)
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
HRESULT solids::lib::mf::transform::merge::factory::CreateInstance(_In_opt_ IUnknown * unk, _In_ REFIID riid, _COM_Outptr_ void ** ppv)
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
	return solids::lib::mf::transform::merge::transform::CreateInstance(riid, ppv);
}

HRESULT solids::lib::mf::transform::merge::factory::LockServer(BOOL lock)
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