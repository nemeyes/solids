#include "d3d11_vertex_declarations.h"
#include "d3d11_exception.h"
#include "d3d11_mesh.h"

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

	void vertex_position::create_vertex_buffer(gsl::not_null<ID3D11Device*> device, const solids::lib::video::sink::d3d11::base::mesh& msh, gsl::not_null<ID3D11Buffer**> vertexBuffer)
	{
		const std::vector<DirectX::XMFLOAT3>& sourceVertices = mesh.Vertices();

		std::vector<vertex_position> vertices;
		vertices.reserve(sourceVertices.size());

		for (size_t i = 0; i < sourceVertices.size(); i++)
		{
			const DirectX::XMFLOAT3& position = sourceVertices.at(i);
			vertices.emplace_back(DirectX::XMFLOAT4(position.x, position.y, position.z, 1.0f));
		}

		VertexDeclaration::create_vertex_buffer(device, vertices, vertexBuffer);
	}

	void vertex_position_color::create_vertex_buffer(not_null<ID3D11Device*> device, const solids::lib::video::sink::d3d11::base::mesh& msh, not_null<ID3D11Buffer**> vertexBuffer)
	{
		const std::vector<DirectX::XMFLOAT3>& sourceVertices = mesh.Vertices();

		std::vector<vertex_position_color> vertices;
		vertices.reserve(sourceVertices.size());

		assert(mesh.VertexColors().size() > 0);
		const std::vector<DirectX::XMFLOAT4>& vertexColors = mesh.VertexColors().at(0);
		assert(vertexColors.size() == sourceVertices.size());

		for (size_t i = 0; i < sourceVertices.size(); i++)
		{
			const XMFLOAT3& position = sourceVertices.at(i);
			const XMFLOAT4& color = vertexColors.at(i);
			vertices.emplace_back(XMFLOAT4(position.x, position.y, position.z, 1.0f), color);
		}

		VertexDeclaration::create_vertex_buffer(device, vertices, vertexBuffer);
	}

	void vertex_positionTexture::create_vertex_buffer(not_null<ID3D11Device*> device, const solids::lib::video::sink::d3d11::base::mesh& msh, not_null<ID3D11Buffer**> vertexBuffer)
	{
		const std::vector<XMFLOAT3>& sourceVertices = mesh.Vertices();
		const std::vector<XMFLOAT3>& textureCoordinates = mesh.TextureCoordinates().at(0);
		assert(textureCoordinates.size() == sourceVertices.size());

		std::vector<vertex_positionTexture> vertices;
		vertices.reserve(sourceVertices.size());
		for (size_t i = 0; i < sourceVertices.size(); i++)
		{
			const XMFLOAT3& position = sourceVertices.at(i);
			const XMFLOAT3& uv = textureCoordinates.at(i);
			vertices.emplace_back(XMFLOAT4(position.x, position.y, position.z, 1.0f), XMFLOAT2(uv.x, uv.y));
		}

		VertexDeclaration::create_vertex_buffer(device, vertices, vertexBuffer);
	}

	void vertex_positionNormal::create_vertex_buffer(not_null<ID3D11Device*> device, const solids::lib::video::sink::d3d11::base::mesh& msh, not_null<ID3D11Buffer**> vertexBuffer)
	{
		const std::vector<XMFLOAT3>& sourceVertices = mesh.Vertices();
		const std::vector<XMFLOAT3>& sourceNormals = mesh.Normals();
		assert(sourceNormals.size() == sourceVertices.size());

		std::vector<vertex_positionNormal> vertices;
		vertices.reserve(sourceVertices.size());
		for (size_t i = 0; i < sourceVertices.size(); i++)
		{
			const XMFLOAT3& position = sourceVertices.at(i);
			const XMFLOAT3& normal = sourceNormals.at(i);
			vertices.emplace_back(XMFLOAT4(position.x, position.y, position.z, 1.0f), normal);
		}

		VertexDeclaration::create_vertex_buffer(device, vertices, vertexBuffer);
	}

	void vertex_positionTextureNormal::create_vertex_buffer(not_null<ID3D11Device*> device, const solids::lib::video::sink::d3d11::base::mesh& msh, not_null<ID3D11Buffer**> vertexBuffer)
	{
		const std::vector<XMFLOAT3>& sourceVertices = mesh.Vertices();
		const auto& sourceUVs = mesh.TextureCoordinates().at(0);
		assert(sourceUVs.size() == sourceVertices.size());
		const auto& sourceNormals = mesh.Normals();
		assert(sourceNormals.size() == sourceVertices.size());

		std::vector<vertex_positionTextureNormal> vertices;
		vertices.reserve(sourceVertices.size());
		for (size_t i = 0; i < sourceVertices.size(); i++)
		{
			const XMFLOAT3& position = sourceVertices.at(i);
			const XMFLOAT3& uv = sourceUVs.at(i);
			const XMFLOAT3& normal = sourceNormals.at(i);
			vertices.emplace_back(XMFLOAT4(position.x, position.y, position.z, 1.0f), XMFLOAT2(uv.x, uv.y), normal);
		}

		VertexDeclaration::create_vertex_buffer(device, vertices, vertexBuffer);
	}

	void vertex_positionTextureNormalTangent::create_vertex_buffer(not_null<ID3D11Device*> device, const solids::lib::video::sink::d3d11::base::mesh& msh, not_null<ID3D11Buffer**> vertexBuffer)
	{
		const std::vector<XMFLOAT3>& sourceVertices = mesh.Vertices();
		const auto& sourceUVs = mesh.TextureCoordinates().at(0);
		assert(sourceUVs.size() == sourceVertices.size());
		const auto& sourceNormals = mesh.Normals();
		assert(sourceNormals.size() == sourceVertices.size());
		const auto& sourceTangents = mesh.Tangents();
		assert(sourceTangents.size() == sourceVertices.size());

		std::vector<vertex_positionTextureNormalTangent> vertices;
		vertices.reserve(sourceVertices.size());
		for (size_t i = 0; i < sourceVertices.size(); i++)
		{
			const XMFLOAT3& position = sourceVertices.at(i);
			const XMFLOAT3& uv = sourceUVs.at(i);
			const XMFLOAT3& normal = sourceNormals.at(i);
			const XMFLOAT3& tangent = sourceTangents.at(i);
			vertices.emplace_back(XMFLOAT4(position.x, position.y, position.z, 1.0f), XMFLOAT2(uv.x, uv.y), normal, tangent);
		}

		VertexDeclaration::create_vertex_buffer(device, vertices, vertexBuffer);
	}

	void VertexSkinnedPositionTextureNormal::create_vertex_buffer(not_null<ID3D11Device*> device, const solids::lib::video::sink::d3d11::base::mesh& msh, not_null<ID3D11Buffer**> vertexBuffer)
	{
		const std::vector<XMFLOAT3>& sourceVertices = mesh.Vertices();
		const auto& sourceUVs = mesh.TextureCoordinates().at(0);
		assert(sourceUVs.size() == sourceVertices.size());
		const auto& sourceNormals = mesh.Normals();
		assert(sourceNormals.size() == sourceVertices.size());
		const auto& boneWeights = mesh.BoneWeights();
		assert(boneWeights.size() == sourceVertices.size());

		std::vector<VertexSkinnedPositionTextureNormal> vertices;
		vertices.reserve(sourceVertices.size());
		for (size_t i = 0; i < sourceVertices.size(); i++)
		{
			const XMFLOAT3& position = sourceVertices.at(i);
			const XMFLOAT3& uv = sourceUVs.at(i);
			const XMFLOAT3& normal = sourceNormals.at(i);
			const BoneVertexWeights& vertexWeights = boneWeights.at(i);

			float weights[BoneVertexWeights::MaxBoneWeightsPerVertex];
			uint32_t indices[BoneVertexWeights::MaxBoneWeightsPerVertex];
			ZeroMemory(weights, sizeof(float) * size(weights));
			ZeroMemory(indices, sizeof(uint32_t) * size(indices));
			for (size_t j = 0; j < vertexWeights.Weights().size(); j++)
			{
				const BoneVertexWeights::VertexWeight& vertexWeight = vertexWeights.Weights().at(j);
				weights[j] = vertexWeight.Weight;
				indices[j] = vertexWeight.BoneIndex;
			}

			vertices.emplace_back(XMFLOAT4(position.x, position.y, position.z, 1.0f), XMFLOAT2(uv.x, uv.y), normal, XMUINT4(indices), XMFLOAT4(weights));
		}

		VertexDeclaration::create_vertex_buffer(device, vertices, vertexBuffer);
	}

};
};
};
};
};
};