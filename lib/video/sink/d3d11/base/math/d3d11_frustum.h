#pragma once

#include <sld.h>
#include "d3d11_ray.h"
#include <DirectXMath.h>

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
						typedef struct _frustum_plane_t
						{
							static const int32_t nearby = 0;
							static const int32_t faraway = 1;
							static const int32_t left = 2;
							static const int32_t right = 3;
							static const int32_t top = 4;
							static const int32_t bottom = 5;
						} frustum_plane_t;

						class frustum
						{
						public:
							frustum(DirectX::CXMMATRIX matrix);
							frustum(const solids::lib::video::sink::d3d11::base::frustum&) = default;
							frustum(solids::lib::video::sink::d3d11::base::frustum&&) = default;
							frustum& operator=(const solids::lib::video::sink::d3d11::base::frustum&) = default;
							frustum& operator=(solids::lib::video::sink::d3d11::base::frustum&&) = default;
							~frustum(void) = default;

							const DirectX::XMFLOAT4& nearby(void) const;
							const DirectX::XMFLOAT4& faraway(void) const;
							const DirectX::XMFLOAT4& left(void) const;
							const DirectX::XMFLOAT4& right(void) const;
							const DirectX::XMFLOAT4& top(void) const;
							const DirectX::XMFLOAT4& bottom(void) const;

							DirectX::XMVECTOR near_vector(void) const;
							DirectX::XMVECTOR far_vector(void) const;
							DirectX::XMVECTOR left_vector(void) const;
							DirectX::XMVECTOR right_vector(void) const;
							DirectX::XMVECTOR top_vector(void) const;
							DirectX::XMVECTOR bottom_vector(void) const;

							const std::array<DirectX::XMFLOAT3, 8>& corners(void) const;

							DirectX::XMMATRIX matrix(void) const;
							void set_matrix(DirectX::CXMMATRIX matrix);
							void set_matrix(const DirectX::XMFLOAT4X4& matrix);

						private:
							static solids::lib::video::sink::d3d11::base::ray compute_intersection_line(DirectX::FXMVECTOR p1, DirectX::FXMVECTOR p2);
							static DirectX::XMVECTOR compute_intersection(DirectX::FXMVECTOR& plane, solids::lib::video::sink::d3d11::base::ray& rays);


						private:
							DirectX::XMFLOAT4X4					_matrix;
							std::array<DirectX::XMFLOAT3, 8>	_corners;
							std::array<DirectX::XMFLOAT4, 6>	_planes;
						};
					};
				};
			};
		};
	};
};