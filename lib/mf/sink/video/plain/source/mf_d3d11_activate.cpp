#include "mf_d3d11_activate.h"
#include "mf_d3d11_media.h"

HRESULT sld::lib::mf::sink::video::plain::activate::create_instance(HWND hwnd, IMFActivate ** ppv)
{
    if (ppv == NULL)
        return E_POINTER;

    sld::lib::mf::sink::video::plain::activate * pActivate = new sld::lib::mf::sink::video::plain::activate();
    if (pActivate == NULL)
        return E_OUTOFMEMORY;

    pActivate->AddRef();
    HRESULT hr = S_OK;
    do
    {
        hr = pActivate->initialize();
        if (FAILED(hr))
            break;

        hr = pActivate->QueryInterface(IID_PPV_ARGS(ppv));
        if (FAILED(hr))
            break;

        pActivate->_hwnd = hwnd;
    } while (FALSE);

    sld::lib::mf::safe_release(pActivate);
    return hr;
}

// IUnknown
ULONG sld::lib::mf::sink::video::plain::activate::AddRef(void)
{
    return sld::lib::mf::refcount_object::AddRef();
}

// IUnknown
ULONG sld::lib::mf::sink::video::plain::activate::Release(void)
{
    return sld::lib::mf::refcount_object::Release();
}

// IUnknown
HRESULT sld::lib::mf::sink::video::plain::activate::QueryInterface(REFIID iid, __RPC__deref_out _Result_nullonfailure_ void** ppv)
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


// IMFActivate
HRESULT sld::lib::mf::sink::video::plain::activate::ActivateObject(__RPC__in REFIID riid, __RPC__deref_out_opt void ** ppv)
{
    HRESULT hr = S_OK;
    IMFGetService* pSinkGetService = NULL;
    IMFVideoDisplayControl* pSinkVideoDisplayControl = NULL;

    do
    {
        if (_media_sink == NULL)
        {
            hr = sld::lib::mf::sink::video::plain::media::create_instance(IID_PPV_ARGS(&_media_sink));
            if (FAILED(hr))
            {
                break;
            }

            hr = _media_sink->QueryInterface(IID_PPV_ARGS(&pSinkGetService));
            if (FAILED(hr))
            {
                break;
            }

            hr = pSinkGetService->GetService(MR_VIDEO_RENDER_SERVICE, IID_PPV_ARGS(&pSinkVideoDisplayControl));
            if (FAILED(hr))
            {
                break;
            }

            hr = pSinkVideoDisplayControl->SetVideoWindow(_hwnd);
            if (FAILED(hr))
            {
                break;
            }
        }

        hr = _media_sink->QueryInterface(riid, ppv);
        if (FAILED(hr))
        {
            break;
        }
    } while (FALSE);

    sld::lib::mf::safe_release(pSinkGetService);
    sld::lib::mf::safe_release(pSinkVideoDisplayControl);

    return hr;
}

// IMFActivate
HRESULT sld::lib::mf::sink::video::plain::activate::DetachObject(void)
{
    sld::lib::mf::safe_release(_media_sink);

    return S_OK;
}

// IMFActivate
HRESULT sld::lib::mf::sink::video::plain::activate::ShutdownObject(void)
{
    if (_media_sink != NULL)
    {
        _media_sink->Shutdown();
        sld::lib::mf::safe_release(_media_sink);
    }

    return S_OK;
}

// IPersistStream
HRESULT sld::lib::mf::sink::video::plain::activate::GetSizeMax(__RPC__out ULARGE_INTEGER* pcbSize)
{
    return E_NOTIMPL;
}

// IPersistStream
HRESULT sld::lib::mf::sink::video::plain::activate::IsDirty(void)
{
    return E_NOTIMPL;
}

// IPersistStream
HRESULT sld::lib::mf::sink::video::plain::activate::Load(__RPC__in_opt IStream* pStream)
{
    return E_NOTIMPL;
}

// IPersistStream
HRESULT sld::lib::mf::sink::video::plain::activate::Save(__RPC__in_opt IStream* pStream, BOOL bClearDirty)
{
    return E_NOTIMPL;
}

// IPersist
HRESULT sld::lib::mf::sink::video::plain::activate::GetClassID(__RPC__out CLSID* pClassID)
{
    if (pClassID == NULL)
        return E_POINTER;

    *pClassID = CLSID_MF_D3D11_RENDERER_ACTIVATE;
    return S_OK;
}

// ctor
sld::lib::mf::sink::video::plain::activate::activate(void)
    : _media_sink(NULL)
    , _hwnd(NULL)
{
}

// dtor
sld::lib::mf::sink::video::plain::activate::~activate(void)
{
    sld::lib::mf::safe_release(_media_sink);
}
