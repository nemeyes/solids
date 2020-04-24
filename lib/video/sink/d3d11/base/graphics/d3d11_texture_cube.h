#pragma once

#include "d3d11_texture.h"

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
						class texture_cube final
							: public solids::lib::video::sink::d3d11::base::texture
						{
							RTTI_DECLARATIONS(texture_cube, texture)

						public:
							texture_cube(const solids::lib::video::sink::d3d11::base::texture_cube&) = default;
							texture_cube& operator=(const solids::lib::video::sink::d3d11::base::texture_cube&) = default;
							texture_cube(solids::lib::video::sink::d3d11::base::texture_cube&&) = default;
							texture_cube& operator=(solids::lib::video::sink::d3d11::base::texture_cube&&) = default;
							~texture_cube(void) = default;

						private:
							friend class texture_cube_reader;

							texture_cube(const winrt::com_ptr<ID3D11ShaderResourceView>& shaderResourceView);
						};
					};
				};
			};
		};
	};
};