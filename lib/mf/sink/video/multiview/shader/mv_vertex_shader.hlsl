cbuffer WVPMatrix
{
	float4x4 WVP;
};
struct VertexOut
{
	float4 pos : SV_POSITION;
	float2 tex : TEXCOORD;
};
VertexOut VS( float4 pos : POSITION , float2 tex : TEXCOORD)
{
	VertexOut vout;
	vout.pos = pos;
	vout.pos = mul(pos, WVP);
	
	vout.tex = tex;

	return vout;
}