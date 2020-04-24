#include "d3d11_frustum.h"

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

	frustum::frustum(DirectX::CXMMATRIX matrix)
	{
		set_matrix(matrix);
	}

	const DirectX::XMFLOAT4& frustum::nearby(void) const
	{
		return _planes[static_cast<int>(solids::lib::video::sink::d3d11::base::frustum_plane_t::nearby)];
	}

	const DirectX::XMFLOAT4& frustum::faraway(void) const
	{
		return _planes[static_cast<int>(solids::lib::video::sink::d3d11::base::frustum_plane_t::faraway)];
	}

	const DirectX::XMFLOAT4& frustum::left(void) const
	{
		return _planes[static_cast<int>(solids::lib::video::sink::d3d11::base::frustum_plane_t::left)];
	}

	const DirectX::XMFLOAT4& frustum::right(void) const
	{
		return _planes[static_cast<int>(solids::lib::video::sink::d3d11::base::frustum_plane_t::right)];
	}

	const DirectX::XMFLOAT4& frustum::top(void) const
	{
		return _planes[static_cast<int>(solids::lib::video::sink::d3d11::base::frustum_plane_t::top)];
	}

	const DirectX::XMFLOAT4& frustum::bottom(void) const
	{
		return _planes[static_cast<int>(solids::lib::video::sink::d3d11::base::frustum_plane_t::bottom)];
	}

	DirectX::XMVECTOR frustum::near_vector(void) const
	{
		return DirectX::XMLoadFloat4(&_planes[static_cast<int>(solids::lib::video::sink::d3d11::base::frustum_plane_t::nearby)]);
	}

	DirectX::XMVECTOR frustum::far_vector(void) const
	{
		return DirectX::XMLoadFloat4(&_planes[static_cast<int>(solids::lib::video::sink::d3d11::base::frustum_plane_t::faraway)]);
	}

	DirectX::XMVECTOR frustum::left_vector(void) const
	{
		return DirectX::XMLoadFloat4(&_planes[static_cast<int>(solids::lib::video::sink::d3d11::base::frustum_plane_t::left)]);
	}

	DirectX::XMVECTOR frustum::right_vector(void) const
	{
		return DirectX::XMLoadFloat4(&_planes[static_cast<int>(solids::lib::video::sink::d3d11::base::frustum_plane_t::right)]);
	}

	DirectX::XMVECTOR frustum::top_vector(void) const
	{
		return DirectX::XMLoadFloat4(&_planes[static_cast<int>(solids::lib::video::sink::d3d11::base::frustum_plane_t::top)]);
	}

	DirectX::XMVECTOR frustum::bottom_vector(void) const
	{
		return DirectX::XMLoadFloat4(&_planes[static_cast<int>(solids::lib::video::sink::d3d11::base::frustum_plane_t::bottom)]);
	}

	const std::array<DirectX::XMFLOAT3, 8>& frustum::corners(void) const
	{
		return _corners;
	}

	DirectX::XMMATRIX frustum::matrix(void) const
	{
		return DirectX::XMLoadFloat4x4(&_matrix);
	}

	void frustum::set_matrix(DirectX::CXMMATRIX matrix)
	{
		DirectX::XMFLOAT4X4 m;
		DirectX::XMStoreFloat4x4(&m, matrix);
		set_matrix(m);
	}

	void frustum::set_matrix(const DirectX::XMFLOAT4X4& matrix)
	{
		_matrix = matrix;

		_planes[0].x = -matrix._13;
		_planes[0].y = -matrix._23;
		_planes[0].z = -matrix._33;
		_planes[0].w = -matrix._43;

		_planes[1].x = -matrix._14 + matrix._13;
		_planes[1].y = -matrix._24 + matrix._23;
		_planes[1].z = -matrix._34 + matrix._33;
		_planes[1].w = -matrix._44 + matrix._43;

		_planes[2].x = -matrix._14 - matrix._11;
		_planes[2].y = -matrix._24 - matrix._21;
		_planes[2].z = -matrix._34 - matrix._31;
		_planes[2].w = -matrix._44 - matrix._41;

		_planes[3].x = -matrix._14 + matrix._11;
		_planes[3].y = -matrix._24 + matrix._21;
		_planes[3].z = -matrix._34 + matrix._31;
		_planes[3].w = -matrix._44 + matrix._41;

		_planes[4].x = -matrix._14 + matrix._12;
		_planes[4].y = -matrix._24 + matrix._22;
		_planes[4].z = -matrix._34 + matrix._32;
		_planes[4].w = -matrix._44 + matrix._42;

		_planes[5].x = -matrix._14 - matrix._12;
		_planes[5].y = -matrix._24 - matrix._22;
		_planes[5].z = -matrix._34 - matrix._32;
		_planes[5].w = -matrix._44 - matrix._42;

		for (auto& plane : _planes)
		{
			DirectX::XMVECTOR vector = DirectX::XMVectorSet(plane.x, plane.y, plane.z, plane.w);
			DirectX::XMVECTOR length = DirectX::XMVector3Length(vector);

			DirectX::XMStoreFloat4(&plane, DirectX::XMVectorDivide(vector, length));
		}

		solids::lib::video::sink::d3d11::base::ray rays = compute_intersection_line(XMLoadFloat4(&_planes[0]), XMLoadFloat4(&_planes[2]));
		DirectX::XMStoreFloat3(&_corners[0], compute_intersection(XMLoadFloat4(&_planes[4]), rays));
		DirectX::XMStoreFloat3(&_corners[3], compute_intersection(XMLoadFloat4(&_planes[5]), rays));

		rays = compute_intersection_line(XMLoadFloat4(&_planes[3]), XMLoadFloat4(&_planes[0]));
		DirectX::XMStoreFloat3(&_corners[1], compute_intersection(XMLoadFloat4(&_planes[4]), rays));
		DirectX::XMStoreFloat3(&_corners[2], compute_intersection(XMLoadFloat4(&_planes[5]), rays));

		rays = compute_intersection_line(XMLoadFloat4(&_planes[2]), XMLoadFloat4(&_planes[1]));
		DirectX::XMStoreFloat3(&_corners[4], compute_intersection(XMLoadFloat4(&_planes[4]), rays));
		DirectX::XMStoreFloat3(&_corners[7], compute_intersection(XMLoadFloat4(&_planes[5]), rays));

		rays = compute_intersection_line(XMLoadFloat4(&_planes[1]), XMLoadFloat4(&_planes[3]));
		DirectX::XMStoreFloat3(&_corners[5], compute_intersection(XMLoadFloat4(&_planes[4]), rays));
		DirectX::XMStoreFloat3(&_corners[6], compute_intersection(XMLoadFloat4(&_planes[5]), rays));
	}

	solids::lib::video::sink::d3d11::base::ray frustum::compute_intersection_line(DirectX::FXMVECTOR p1, DirectX::FXMVECTOR p2)
	{
		DirectX::XMVECTOR direction = DirectX::XMVector3Cross(p1, p2);
		DirectX::XMVECTOR lengthSquared = DirectX::XMVector3LengthSq(direction);
		DirectX::XMVECTOR position = DirectX::XMVector3Cross((-DirectX::XMVectorGetW(p1) * p2) + (DirectX::XMVectorGetW(p2) * p1), direction) / lengthSquared;

		return solids::lib::video::sink::d3d11::base::ray(position, direction);
	}

	DirectX::XMVECTOR frustum::compute_intersection(DirectX::FXMVECTOR& plane, solids::lib::video::sink::d3d11::base::ray& rays)
	{
		float value = (-DirectX::XMVectorGetW(plane) - DirectX::XMVectorGetX(DirectX::XMVector3Dot(plane, rays.position_vector()))) / DirectX::XMVectorGetX(DirectX::XMVector3Dot(plane, rays.direction_vector()));

		return (rays.position_vector() + (rays.direction_vector() * value));
	}

};
};
};
};
};
};