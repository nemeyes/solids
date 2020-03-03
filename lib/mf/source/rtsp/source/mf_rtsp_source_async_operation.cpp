#include "mf_rtsp_source_async_operation.h"

solids::lib::mf::source::rtsp::async_operation::async_operation(int32_t op)
	: _op(op)
{
	PropVariantInit(&_data);
}

solids::lib::mf::source::rtsp::async_operation::~async_operation(void)
{
	PropVariantClear(&_data);
}

// IUnknown methods.
ULONG solids::lib::mf::source::rtsp::async_operation::AddRef(void)
{
	return refcount_object::AddRef();
}

ULONG solids::lib::mf::source::rtsp::async_operation::Release(void)
{
	return refcount_object::Release();
}

HRESULT solids::lib::mf::source::rtsp::async_operation::QueryInterface(REFIID iid, void ** ppv)
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

int32_t solids::lib::mf::source::rtsp::async_operation::op(void)
{
	return _op;
}

HRESULT solids::lib::mf::source::rtsp::async_operation::data(const PROPVARIANT & var)
{
	return PropVariantCopy(&_data, &var);
}

const PROPVARIANT & solids::lib::mf::source::rtsp::async_operation::data(void)
{
	return _data;
}

solids::lib::mf::source::rtsp::start_async_operation::start_async_operation(IMFPresentationDescriptor * pd)
	: solids::lib::mf::source::rtsp::async_operation(solids::lib::mf::source::rtsp::start_async_operation::type_t::start)
	, _pd(pd)
{
	if (_pd)
		_pd->AddRef();
}

solids::lib::mf::source::rtsp::start_async_operation::~start_async_operation(void)
{
	safe_release(_pd);
}

HRESULT solids::lib::mf::source::rtsp::start_async_operation::presentation_descriptor(IMFPresentationDescriptor ** pd)
{
	if (!pd)
		return E_POINTER;
	if (!_pd)
		return MF_E_INVALIDREQUEST;
	*pd = _pd;
	(*pd)->AddRef();
	return S_OK;
}
