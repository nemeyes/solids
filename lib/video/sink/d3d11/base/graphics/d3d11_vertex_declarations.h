#pragma once

#include <sld.h>
#include <DirectXMath.h>
#include <gsl/gsl>

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
						class mesh;
						template <typename T>
						struct vertex_declaration_t
						{
							static constexpr uint32_t vertex_size(void) 
							{ 
								return gsl::narrow_cast<uint32_t>(sizeof(T)); 
							}

							static constexpr uint32_t vertex_buffer_bytewidth(size_t vertexCount) 
							{ 
								return gsl::narrow_cast<uint32_t>(sizeof(T) * vertexCount); 
							}

							static void create_vertex_buffer(gsl::not_null<ID3D11Device*> device, const gsl::span<const T>& vertices, gsl::not_null<ID3D11Buffer**> vertexBuffer);
						};

						class vertex_position
							: public solids::lib::video::sink::d3d11::base::vertex_declaration_t<solids::lib::video::sink::d3d11::base::vertex_position>
						{
						private:
							inline static const D3D11_INPUT_ELEMENT_DESC _input_elements[]
							{
								{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
							};

						public:
							vertex_position(void) = default;
							vertex_position(const DirectX::XMFLOAT4& position)
								: position(position) 
							{ }

							DirectX::XMFLOAT4 position;
							inline static const gsl::span<const D3D11_INPUT_ELEMENT_DESC> input_elements{ _input_elements };

							static void CreateVertexBuffer(gsl::not_null<ID3D11Device*> device, const solids::lib::video::sink::d3d11::base::mesh& msh, gsl::not_null<ID3D11Buffer**> vertexBuffer);
							static void CreateVertexBuffer(gsl::not_null<ID3D11Device*> device, const gsl::span<const solids::lib::video::sink::d3d11::base::vertex_position>& vertices, gsl::not_null<ID3D11Buffer**> vertexBuffer)
							{
								vertex_declaration_t::create_vertex_buffer(device, vertices, vertexBuffer);
							}
						};

						class vertex_position_color
							: public solids::lib::video::sink::d3d11::base::vertex_declaration_t<vertex_position_color>
						{
						private:
							inline static const D3D11_INPUT_ELEMENT_DESC _input_elements[]
							{
								{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
								{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
							};

						public:
							vertex_position_color(void) = default;
							vertex_position_color(const DirectX::XMFLOAT4& position, const DirectX::XMFLOAT4& color)
								: position(position)
								, color(color)
							{ }

							DirectX::XMFLOAT4 position;
							DirectX::XMFLOAT4 color;

							inline static const gsl::span<const D3D11_INPUT_ELEMENT_DESC> input_elements{ _input_elements };

							static void create_vertex_buffer(gsl::not_null<ID3D11Device*> device, const solids::lib::video::sink::d3d11::base::mesh& msh, gsl::not_null<ID3D11Buffer**> vertexBuffer);
							static void create_vertex_buffer(gsl::not_null<ID3D11Device*> device, const gsl::span<const solids::lib::video::sink::d3d11::base::vertex_position_color>& vertices, gsl::not_null<ID3D11Buffer**> vertexBuffer)
							{
								vertex_declaration_t::create_vertex_buffer(device, vertices, vertexBuffer);
							}
						};

						class vertex_position_texture
							: public solids::lib::video::sink::d3d11::base::vertex_declaration_t<vertex_position_texture>
						{
						private:
							inline static const D3D11_INPUT_ELEMENT_DESC _input_elements[]
							{
								{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
								{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
							};
						public:
							vertex_position_texture(void) = default;
							vertex_position_texture(const DirectX::XMFLOAT4& position, const DirectX::XMFLOAT2& textureCoordinates)
								: position(position)
								, texture_coordinates(textureCoordinates)
							{ }

							DirectX::XMFLOAT4 position;
							DirectX::XMFLOAT2 texture_coordinates;

							inline static const gsl::span<const D3D11_INPUT_ELEMENT_DESC> input_elements{ _input_elements };

							static void create_vertex_buffer(gsl::not_null<ID3D11Device*> device, const solids::lib::video::sink::d3d11::base::mesh& msh, gsl::not_null<ID3D11Buffer**> vertexBuffer);
							static void create_vertex_buffer(gsl::not_null<ID3D11Device*> device, const gsl::span<const solids::lib::video::sink::d3d11::base::vertex_position_texture>& vertices, gsl::not_null<ID3D11Buffer**> vertexBuffer)
							{
								vertex_declaration_t::create_vertex_buffer(device, vertices, vertexBuffer);
							}
						};

						class vertex_position_size
							: public solids::lib::video::sink::d3d11::base::vertex_declaration_t<vertex_position_size>
						{
						private:
							inline static const D3D11_INPUT_ELEMENT_DESC _input_elements[]
							{
								{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
								{ "SIZE", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
							};

						public:
							vertex_position_size(void) = default;
							vertex_position_size(const DirectX::XMFLOAT4& position, const DirectX::XMFLOAT2& size)
								: position(position)
								, size(size)
							{ }

							DirectX::XMFLOAT4 position;
							DirectX::XMFLOAT2 size;

							inline static const gsl::span<const D3D11_INPUT_ELEMENT_DESC> input_elements{ _input_elements };
							static void create_vertex_buffer(gsl::not_null<ID3D11Device*> device, const gsl::span<const vertex_position_size>& vertices, gsl::not_null<ID3D11Buffer**> vertexBuffer)
							{
								vertex_declaration_t::create_vertex_buffer(device, vertices, vertexBuffer);
							}
						};

						class vertex_position_normal
							: public solids::lib::video::sink::d3d11::base::vertex_declaration_t<vertex_position_normal>
						{
						private:
							inline static const D3D11_INPUT_ELEMENT_DESC _input_elements[]
							{
								{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
								{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
							};

						public:
							vertex_position_normal(void) = default;
							vertex_position_normal(const DirectX::XMFLOAT4& position, const DirectX::XMFLOAT3& normal)
								: position(position)
								, normal(normal)
							{ }

							DirectX::XMFLOAT4 position;
							DirectX::XMFLOAT3 normal;

							inline static const gsl::span<const D3D11_INPUT_ELEMENT_DESC> input_elements{ _input_elements };

							static void create_vertex_buffer(gsl::not_null<ID3D11Device*> device, const solids::lib::video::sink::d3d11::base::mesh& msh, gsl::not_null<ID3D11Buffer**> vertexBuffer);
							static void create_vertex_buffer(gsl::not_null<ID3D11Device*> device, const gsl::span<const vertex_position_normal>& vertices, gsl::not_null<ID3D11Buffer**> vertexBuffer)
							{
								vertex_declaration_t::create_vertex_buffer(device, vertices, vertexBuffer);
							}
						};

						class vertex_position_texture_normal
							: public solids::lib::video::sink::d3d11::base::vertex_declaration_t<vertex_position_texture_normal>
						{
						private:
							inline static const D3D11_INPUT_ELEMENT_DESC _input_elements[]
							{
								{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
								{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
								{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
							};

						public:
							vertex_position_texture_normal(void) = default;
							vertex_position_texture_normal(const DirectX::XMFLOAT4& position, const DirectX::XMFLOAT2& textureCoordinates, const DirectX::XMFLOAT3& normal)
								: position(position)
								, texture_coordinates(textureCoordinates)
								, normal(normal) { }

							DirectX::XMFLOAT4 position;
							DirectX::XMFLOAT2 texture_coordinates;
							DirectX::XMFLOAT3 normal;

							inline static const gsl::span<const D3D11_INPUT_ELEMENT_DESC> input_elements{ _input_elements };

							static void create_vertex_buffer(gsl::not_null<ID3D11Device*> device, const solids::lib::video::sink::d3d11::base::mesh& msh, gsl::not_null<ID3D11Buffer**> vertexBuffer);
							static void create_vertex_buffer(gsl::not_null<ID3D11Device*> device, const gsl::span<const vertex_position_texture_normal>& vertices, gsl::not_null<ID3D11Buffer**> vertexBuffer)
							{
								vertex_declaration_t::create_vertex_buffer(device, vertices, vertexBuffer);
							}
						};

						class vertex_position_texture_normal_tangent
							: public solids::lib::video::sink::d3d11::base::vertex_declaration_t<vertex_position_texture_normal_tangent>
						{
						private:
							inline static const D3D11_INPUT_ELEMENT_DESC _input_elements[]
							{
								{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
								{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
								{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
								{ "TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
							};

						public:
							vertex_position_texture_normal_tangent(void) = default;
							vertex_position_texture_normal_tangent(const DirectX::XMFLOAT4& position, const DirectX::XMFLOAT2& textureCoordinates, const DirectX::XMFLOAT3& normal, const DirectX::XMFLOAT3& tangent)
								: position(position)
								, texture_coordinates(textureCoordinates)
								, normal(normal)
								, tangent(tangent)
							{ }

							DirectX::XMFLOAT4 position;
							DirectX::XMFLOAT2 texture_coordinates;
							DirectX::XMFLOAT3 normal;
							DirectX::XMFLOAT3 tangent;

							inline static const gsl::span<const D3D11_INPUT_ELEMENT_DESC> input_elements{ _input_elements };

							static void create_vertex_buffer(gsl::not_null<ID3D11Device*> device, const solids::lib::video::sink::d3d11::base::mesh& msh, gsl::not_null<ID3D11Buffer**> vertexBuffer);
							static void create_vertex_buffer(gsl::not_null<ID3D11Device*> device, const gsl::span<const vertex_position_texture_normal_tangent>& vertices, gsl::not_null<ID3D11Buffer**> vertexBuffer)
							{
								vertex_declaration_t::create_vertex_buffer(device, vertices, vertexBuffer);
							}
						};

						class vertex_skinned_position_texture_normal
							: public solids::lib::video::sink::d3d11::base::vertex_declaration_t<vertex_skinned_position_texture_normal>
						{
						private:
							inline static const D3D11_INPUT_ELEMENT_DESC _input_elements[]
							{
								{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
								{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
								{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
								{ "BONEINDICES", 0, DXGI_FORMAT_R32G32B32A32_UINT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
								{ "BONEWEIGHTS", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 }
							};

						public:
							vertex_skinned_position_texture_normal(void) = default;
							vertex_skinned_position_texture_normal(const DirectX::XMFLOAT4& position, const DirectX::XMFLOAT2& textureCoordinates, const DirectX::XMFLOAT3& normal, const DirectX::XMUINT4& boneIndices, const DirectX::XMFLOAT4& boneWeights)
								: position(position)
								, texture_coordinates(textureCoordinates)
								, normal(normal)
								, bone_indices(boneIndices)
								, bone_weights(boneWeights) { }

							DirectX::XMFLOAT4	position;
							DirectX::XMFLOAT2	texture_coordinates;
							DirectX::XMFLOAT3	normal;
							DirectX::XMUINT4	bone_indices;
							DirectX::XMFLOAT4	bone_weights;

							inline static const gsl::span<const D3D11_INPUT_ELEMENT_DESC> input_elements{ _input_elements };

							static void create_vertex_buffer(gsl::not_null<ID3D11Device*> device, const solids::lib::video::sink::d3d11::base::mesh& msh, gsl::not_null<ID3D11Buffer**> vertexBuffer);
							static void create_vertex_buffer(gsl::not_null<ID3D11Device*> device, const gsl::span<const vertex_skinned_position_texture_normal>& vertices, gsl::not_null<ID3D11Buffer**> vertexBuffer)
							{
								vertex_declaration_t::create_vertex_buffer(device, vertices, vertexBuffer);
							}
						};
					};
				};
			};
		};
	};
};

#include "d3d11_vertex_declarations.inl"