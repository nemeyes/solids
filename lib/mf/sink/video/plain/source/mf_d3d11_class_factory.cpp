#include "mf_d3d11_class_factory.h"
#include "mf_d3d11_media.h"

BOOL sld::lib::mf::sink::video::plain::factory::is_locked(void)
{
    return (_lock_count == 0) ? FALSE : TRUE;
}

sld::lib::mf::sink::video::plain::factory::factory(void)
    : _ref_count(0)
{
}

sld::lib::mf::sink::video::plain::factory::~factory(void)
{
}

// IUnknown
ULONG sld::lib::mf::sink::video::plain::factory::AddRef(void)
{
    return ::InterlockedIncrement(&_ref_count);
}

// IUnknown
ULONG sld::lib::mf::sink::video::plain::factory::Release(void)
{
    ULONG lRefCount = ::InterlockedDecrement(&_ref_count);
    if (lRefCount == 0)
        delete this;
    return lRefCount;
}

// IUnknown
HRESULT sld::lib::mf::sink::video::plain::factory::QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void** ppv)
{
    if (!ppv)
    {
        return E_POINTER;
    }
    if (iid == IID_IUnknown)
    {
        *ppv = static_cast<IUnknown*>(static_cast<IClassFactory*>(this));
    }
    else if (iid == __uuidof(IClassFactory))
    {
        *ppv = static_cast<IClassFactory*>(this);
    }
    else
    {
        *ppv = NULL;
        return E_NOINTERFACE;
    }
    AddRef();
    return S_OK;
}

// IClassFactory
HRESULT sld::lib::mf::sink::video::plain::factory::CreateInstance(_In_opt_ IUnknown* unk, _In_ REFIID riid, _COM_Outptr_ void ** ppv)
{
    if (ppv == NULL)
        return E_POINTER;

    *ppv = NULL;

    if (unk != NULL)
        return CLASS_E_NOAGGREGATION;

    return sld::lib::mf::sink::video::plain::media::create_instance(riid, ppv);
}

// IClassFactory
HRESULT sld::lib::mf::sink::video::plain::factory::LockServer(BOOL lock)
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
