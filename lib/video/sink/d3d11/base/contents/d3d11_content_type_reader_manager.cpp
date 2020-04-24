#include "d3d11_content_type_reader_manager.h"

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

	std::map<uint64_t, std::shared_ptr<solids::lib::video::sink::d3d11::base::base_content_type_reader>> content_type_reader_manager::_content_type_readers;
	BOOL content_type_reader_manager::_initialized;

	const std::map<uint64_t, std::shared_ptr<solids::lib::video::sink::d3d11::base::base_content_type_reader>>& content_type_reader_manager::content_type_readers(void)
	{
		return _content_type_readers;
	}

	BOOL content_type_reader_manager::add(std::shared_ptr<solids::lib::video::sink::d3d11::base::base_content_type_reader> reader)
	{
		return _content_type_readers.emplace(reader->target_type_id(), move(reader)).second;
	}

	void content_type_reader_manager::initialize(solids::lib::video::sink::d3d11::base::engine& core)
	{
		if (_initialized == FALSE)
		{
			/*
			add(std::make_shared<Texture2DReader>(core));
			add(std::make_shared<TextureCubeReader>(core));
			add(std::make_shared<VertexShaderReader>(core));
			add(std::make_shared<PixelShaderReader>(core));
			add(std::make_shared<HullShaderReader>(core));
			add(std::make_shared<DomainShaderReader>(core));
			add(std::make_shared<GeometryShaderReader>(core));
			add(std::make_shared<ComputeShaderReader>(core));
			add(std::make_shared<ModelReader>(core));
			*/
			_initialized = TRUE;
		}
	}

	void content_type_reader_manager::release(void)
	{
		_content_type_readers.clear();
		_initialized = FALSE;
	}

};
};
};
};
};
};