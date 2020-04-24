#pragma once

#include <gsl/gsl>
#include "d3d11_texture.h"
#include "d3d11_rectangle.h"

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
						class texture_2d final
							: public solids::lib::video::sink::d3d11::base::texture
						{
							RTTI_DECLARATIONS(texture_2d, texture)

						public:
							texture_2d(const winrt::com_ptr<ID3D11ShaderResourceView>& shaderResourceView, std::uint32_t width, std::uint32_t height);
							texture_2d(const solids::lib::video::sink::d3d11::base::texture_2d&) = default;
							texture_2d& operator=(const solids::lib::video::sink::d3d11::base::texture_2d&) = default;
							texture_2d(solids::lib::video::sink::d3d11::base::texture_2d&&) = default;
							texture_2d& operator=(solids::lib::video::sink::d3d11::base::texture_2d&&) = default;
							~texture_2d(void) = default;

							static std::shared_ptr<solids::lib::video::sink::d3d11::base::texture_2d> create_texture_2d(gsl::not_null<ID3D11Device*> device, const D3D11_TEXTURE2D_DESC& textureDesc);
							static std::shared_ptr<solids::lib::video::sink::d3d11::base::texture_2d> create_texture_2d(gsl::not_null<ID3D11Device*> device, std::uint32_t width, std::uint32_t height, std::uint32_t mipLevels = 1, std::uint32_t arraySize = 1, DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC sampleDesc = { 1, 0 }, std::uint32_t bindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE, std::uint32_t cpuAccessFlags = 0);

							std::uint32_t width(void) const;
							std::uint32_t height(void) const;
							solids::lib::video::sink::d3d11::base::rectangle bounds(void) const;

						private:
							std::uint32_t _width;
							std::uint32_t _height;
						};
					};
				};
			};
		};
	};
};