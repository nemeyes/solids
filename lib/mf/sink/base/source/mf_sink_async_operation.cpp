#include "mf_sink_async_operation.h"

solids::lib::mf::sink::async_operation::async_operation(int32_t op)
	: _op(op)
{}

solids::lib::mf::sink::async_operation::~async_operation(void)
{

}

// IUnknown methods.
ULONG solids::lib::mf::sink::async_operation::AddRef(void)
{
	return refcount_object::AddRef();
}

ULONG solids::lib::mf::sink::async_operation::Release(void)
{
	return refcount_object::Release();
}

HRESULT solids::lib::mf::sink::async_operation::QueryInterface(REFIID iid, void** ppv)
{
	if (!ppv)
		return E_POINTER;
	if (iid == IID_IUnknown)
		*ppv = static_cast<IUnknown*>(this);
	else
	{
		*ppv = NULL;
		return E_NOINTERFACE;
	}
	AddRef();
	return S_OK;
}

int32_t solids::lib::mf::sink::async_operation::async_operation::op(void)
{
	return _op;
}