#pragma once

#include <sld.h>
#include <DirectXMath.h>
#include "d3d11_rtti.h"

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
						class ray final
							: public solids::lib::video::sink::d3d11::base::rtti
						{
							RTTI_DECLARATIONS(ray, rtti)

						public:
							ray(DirectX::FXMVECTOR position, DirectX::FXMVECTOR direction);
							ray(const DirectX::XMFLOAT3& position, const DirectX::XMFLOAT3& direction);
							ray(const solids::lib::video::sink::d3d11::base::ray&) = default;
							ray& operator=(const solids::lib::video::sink::d3d11::base::ray&) = default;
							ray(solids::lib::video::sink::d3d11::base::ray&&) = default;
							ray& operator=(solids::lib::video::sink::d3d11::base::ray&&) = default;
							~ray(void) = default;

							const DirectX::XMFLOAT3& position(void) const;
							const DirectX::XMFLOAT3& direction(void) const;

							DirectX::XMVECTOR position_vector(void) const;
							DirectX::XMVECTOR direction_vector(void) const;

							virtual void set_position(float x, float y, float z);
							virtual void set_position(DirectX::FXMVECTOR position);
							virtual void set_position(const DirectX::XMFLOAT3& position);

							virtual void set_direction(float x, float y, float z);
							virtual void set_direction(DirectX::FXMVECTOR direction);
							virtual void set_direction(const DirectX::XMFLOAT3& direction);

						private:
							DirectX::XMFLOAT3 _position;
							DirectX::XMFLOAT3 _direction;
						};
					};
				};
			};
		};
	};
};