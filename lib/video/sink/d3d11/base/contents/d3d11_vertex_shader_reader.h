#pragma once

#include "d3d11_content_type_reader.h"
#include "d3d11_vertex_shader.h"

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
						class vertex_shader_reader
							: public solids::lib::video::sink::d3d11::base::content_type_reader<vertex_shader>
						{
							RTTI_DECLARATIONS(vertex_shader_reader, base_content_type_reader)

						public:
							vertex_shader_reader(solids::lib::video::sink::d3d11::base::engine& core);
							vertex_shader_reader(const vertex_shader_reader&) = default;
							vertex_shader_reader& operator=(const vertex_shader_reader&) = default;
							vertex_shader_reader(vertex_shader_reader&&) = default;
							vertex_shader_reader& operator=(vertex_shader_reader&&) = default;
							~vertex_shader_reader(void) = default;

						protected:
							virtual std::shared_ptr<solids::lib::video::sink::d3d11::base::vertex_shader> _read(const std::wstring& name) override;

						};
					};
				};
			};
		};
	};
};