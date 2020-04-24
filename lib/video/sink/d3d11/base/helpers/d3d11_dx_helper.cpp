#include "d3d11_dx_helper.h"
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

	void create_index_buffer(gsl::not_null<ID3D11Device*> device, const gsl::span<const std::uint16_t>& indices, gsl::not_null<ID3D11Buffer**> indexBuffer)
	{
		D3D11_BUFFER_DESC indexBufferDesc{ 0 };
		indexBufferDesc.ByteWidth = gsl::narrow<uint32_t>(indices.size_bytes());
		indexBufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
		indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;

		D3D11_SUBRESOURCE_DATA indexSubResourceData{ 0 };
		indexSubResourceData.pSysMem = &indices[0];
		solids::lib::video::sink::d3d11::base::throw_if_failed(device->CreateBuffer(&indexBufferDesc, &indexSubResourceData, indexBuffer), "ID3D11Device::CreateBuffer() failed.");
	}

	void create_index_buffer(gsl::not_null<ID3D11Device*> device, const gsl::span<const std::uint32_t>& indices, gsl::not_null<ID3D11Buffer**> indexBuffer)
	{
		D3D11_BUFFER_DESC indexBufferDesc{ 0 };
		indexBufferDesc.ByteWidth = gsl::narrow<uint32_t>(indices.size_bytes());
		indexBufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
		indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;

		D3D11_SUBRESOURCE_DATA indexSubResourceData{ 0 };
		indexSubResourceData.pSysMem = &indices[0];
		solids::lib::video::sink::d3d11::base::throw_if_failed(device->CreateBuffer(&indexBufferDesc, &indexSubResourceData, indexBuffer), "ID3D11Device::CreateBuffer() failed.");
	}

	void create_constant_buffer(gsl::not_null<ID3D11Device*> device, std::size_t byteWidth, gsl::not_null<ID3D11Buffer**> constantBuffer)
	{
		D3D11_BUFFER_DESC constantBufferDesc{ 0 };
		constantBufferDesc.ByteWidth = gsl::narrow<uint32_t>(byteWidth);
		constantBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		solids::lib::video::sink::d3d11::base::throw_if_failed(device->CreateBuffer(&constantBufferDesc, nullptr, constantBuffer), "ID3D11Device::CreateBuffer() failed.");
	}

};
};
};
};
};
};