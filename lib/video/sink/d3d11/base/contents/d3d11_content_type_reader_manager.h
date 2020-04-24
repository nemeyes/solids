#pragma once

#include "d3d11_content_type_reader.h"

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
						class content_type_reader_manager final
						{
						public:
							content_type_reader_manager(void) = delete;
							content_type_reader_manager(const content_type_reader_manager&) = delete;
							content_type_reader_manager& operator=(const content_type_reader_manager&) = delete;
							content_type_reader_manager(content_type_reader_manager&&) = delete;
							content_type_reader_manager& operator=(content_type_reader_manager&&) = delete;
							~content_type_reader_manager(void) = default;

							static const std::map<std::uint64_t, std::shared_ptr<solids::lib::video::sink::d3d11::base::base_content_type_reader>>& content_type_readers(void);
							static BOOL add(std::shared_ptr<solids::lib::video::sink::d3d11::base::base_content_type_reader> reader);

							static void initialize(solids::lib::video::sink::d3d11::base::engine & core);
							static void release(void);

						private:
							static std::map<std::uint64_t, std::shared_ptr<solids::lib::video::sink::d3d11::base::base_content_type_reader>> _content_type_readers;
							static BOOL _initialized;
						};
					};
				};
			};
		};
	};
};