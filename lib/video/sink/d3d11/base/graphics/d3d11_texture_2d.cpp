#include "d3d11_texture_2d.h"
#include "d3d11_exception.h"

namespace solids
{
namespace lib
{
namespace video
{
namespace sink
{
namespace d3d11
{
namespace base
{

	RTTI_DEFINITIONS(texture_2d)

	texture_2d::texture_2d(const winrt::com_ptr<ID3D11ShaderResourceView>& shaderResourceView, std::uint32_t width, std::uint32_t height)
		: texture(shaderResourceView)
		, _width(width)
		, _height(height)
	{
	}

	std::shared_ptr<solids::lib::video::sink::d3d11::base::texture_2d> texture_2d::create_texture_2d(gsl::not_null<ID3D11Device*> device, const D3D11_TEXTURE2D_DESC& textureDesc)
	{
		HRESULT hr;
		winrt::com_ptr<ID3D11Texture2D> texture;
		if (FAILED(hr = device->CreateTexture2D(&textureDesc, nullptr, texture.put())))
		{
			throw solids::lib::video::sink::d3d11::base::exception("IDXGIDevice::CreateTexture2D() failed.", hr);
		}

		winrt::com_ptr<ID3D11ShaderResourceView> shaderResourceReview;
		if (FAILED(hr = device->CreateShaderResourceView(texture.get(), nullptr, shaderResourceReview.put())))
		{
			throw solids::lib::video::sink::d3d11::base::exception("IDXGIDevice::CreateShaderResourceView() failed.", hr);
		}

		return std::make_shared<solids::lib::video::sink::d3d11::base::texture_2d>(shaderResourceReview, textureDesc.Width, textureDesc.Height);
	}

	std::shared_ptr<solids::lib::video::sink::d3d11::base::texture_2d> texture_2d::create_texture_2d(gsl::not_null<ID3D11Device*> device, uint32_t width, uint32_t height, uint32_t mipLevels, uint32_t arraySize, DXGI_FORMAT format, DXGI_SAMPLE_DESC sampleDesc, uint32_t bindFlags, uint32_t cpuAccessFlags)
	{
		D3D11_TEXTURE2D_DESC textureDesc;
		ZeroMemory(&textureDesc, sizeof(textureDesc));
		textureDesc.Width = width;
		textureDesc.Height = height;
		textureDesc.MipLevels = mipLevels;
		textureDesc.ArraySize = arraySize;
		textureDesc.Format = format;
		textureDesc.SampleDesc = sampleDesc;
		textureDesc.BindFlags = bindFlags;
		textureDesc.CPUAccessFlags = cpuAccessFlags;

		return create_texture_2d(device, textureDesc);
	}

	uint32_t texture_2d::width(void) const
	{
		return _width;
	}

	uint32_t texture_2d::height(void) const
	{
		return _height;
	}

	solids::lib::video::sink::d3d11::base::rectangle texture_2d::bounds(void) const
	{
		return solids::lib::video::sink::d3d11::base::rectangle(0, 0, gsl::narrow<int32_t>(_width), gsl::narrow<int32_t>(_height));
	}

};
};
};
};
};
};