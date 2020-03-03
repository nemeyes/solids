#include "mf_rtsp_source_async_operation.h"

sld::lib::mf::source::rtsp::async_operation::async_operation(int32_t op)
	: _op(op)
{
	PropVariantInit(&_data);
}

sld::lib::mf::source::rtsp::async_operation::~async_operation(void)
{
	PropVariantClear(&_data);
}

// IUnknown methods.
ULONG sld::lib::mf::source::rtsp::async_operation::AddRef(void)
{
	return refcount_object::AddRef();
}

ULONG sld::lib::mf::source::rtsp::async_operation::Release(void)
{
	return refcount_object::Release();
}

HRESULT sld::lib::mf::source::rtsp::async_operation::QueryInterface(REFIID iid, void ** ppv)
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

int32_t sld::lib::mf::source::rtsp::async_operation::op(void)
{
	return _op;
}

HRESULT sld::lib::mf::source::rtsp::async_operation::data(const PROPVARIANT & var)
{
	return PropVariantCopy(&_data, &var);
}

const PROPVARIANT & sld::lib::mf::source::rtsp::async_operation::data(void)
{
	return _data;
}

sld::lib::mf::source::rtsp::start_async_operation::start_async_operation(IMFPresentationDescriptor * pd)
	: sld::lib::mf::source::rtsp::async_operation(sld::lib::mf::source::rtsp::start_async_operation::type_t::start)
	, _pd(pd)
{
	if (_pd)
		_pd->AddRef();
}

sld::lib::mf::source::rtsp::start_async_operation::~start_async_operation(void)
{
	safe_release(_pd);
}

HRESULT sld::lib::mf::source::rtsp::start_async_operation::presentation_descriptor(IMFPresentationDescriptor ** pd)
{
	if (!pd)
		return E_POINTER;
	if (!_pd)
		return MF_E_INVALIDREQUEST;
	*pd = _pd;
	(*pd)->AddRef();
	return S_OK;
}
