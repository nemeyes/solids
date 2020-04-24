#include "d3d11_ray.h"

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

    RTTI_DEFINITIONS(ray)

    ray::ray(DirectX::FXMVECTOR position, DirectX::FXMVECTOR direction)
    {
        DirectX::XMStoreFloat3(&_position, position);
        DirectX::XMStoreFloat3(&_direction, direction);
    }

    ray::ray(const DirectX::XMFLOAT3& position, const DirectX::XMFLOAT3& direction)
        : _position(position)
        , _direction(direction)
    {
    }

    const DirectX::XMFLOAT3& ray::position(void) const
    {
        return _position;
    }

    const DirectX::XMFLOAT3& ray::direction(void) const
    {
        return _direction;
    }

    DirectX::XMVECTOR ray::position_vector(void) const
    {
        return DirectX::XMLoadFloat3(&_position);
    }

    DirectX::XMVECTOR ray::direction_vector(void) const
    {
        return DirectX::XMLoadFloat3(&_direction);
    }

    void ray::set_position(float x, float y, float z)
    {
        DirectX::XMVECTOR position = DirectX::XMVectorSet(x, y, z, 1.0f);
        set_position(position);
    }

    void ray::set_position(DirectX::FXMVECTOR position)
    {
        DirectX::XMStoreFloat3(&_position, position);
    }

    void ray::set_position(const DirectX::XMFLOAT3& position)
    {
        _position = position;
    }

    void ray::set_direction(float x, float y, float z)
    {
        DirectX::XMVECTOR direction = DirectX::XMVectorSet(x, y, z, 0.0f);
        set_direction(direction);
    }

    void ray::set_direction(DirectX::FXMVECTOR direction)
    {
        DirectX::XMStoreFloat3(&_direction, direction);
    }

    void ray::set_direction(const DirectX::XMFLOAT3& direction)
    {
        _direction = direction;
    }

};
};
};
};
};
};