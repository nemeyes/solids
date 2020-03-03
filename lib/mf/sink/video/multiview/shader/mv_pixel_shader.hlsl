Texture2D src;
SamplerState sampleState;

struct VertexOut
{
	float4 pos : SV_POSITION;
	float2 tex : TEXCOORD;
};
//SamplerState filter
//{
//	Filter = ANISOTROPIC;
//	MaxAnisotropy = 4;
//};
float4 PS(VertexOut vout) : SV_TARGET
{
	if(vout.tex.x >= 0)
		return src.Sample(sampleState, vout.tex); 
	else
		return float4(1.0f, 112.0f / 255, 0.0f, 1.0f);
	//return float4(0.0f, 1.0f, 0.0f, 1.0f);
}