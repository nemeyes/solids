#pragma once

#include <sld.h>
#include <gsl/gsl>
#include "d3d11_rtti.h"

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
						class engine;
						class base_content_type_reader
							: public solids::lib::video::sink::d3d11::base::rtti
						{
							RTTI_DECLARATIONS(base_content_type_reader, rtti)
						public:
							base_content_type_reader(const base_content_type_reader&) = default;
							base_content_type_reader& operator=(const base_content_type_reader&) = default;
							base_content_type_reader(base_content_type_reader&&) = default;
							base_content_type_reader& operator=(base_content_type_reader&&) = default;
							virtual ~base_content_type_reader(void) = default;

							std::uint64_t target_type_id(void) const;
							virtual std::shared_ptr<solids::lib::video::sink::d3d11::base::rtti> read(const std::wstring& name) = 0;

						protected:
							base_content_type_reader(solids::lib::video::sink::d3d11::base::engine& core, const std::uint64_t target_type_id);


						protected:
							gsl::not_null<solids::lib::video::sink::d3d11::base::engine*> _engine;
							const std::uint64_t _target_type_id;
						};

						template <typename T>
						class content_type_reader : public solids::lib::video::sink::d3d11::base::base_content_type_reader
						{
						public:
							content_type_reader(const content_type_reader&) = default;
							content_type_reader& operator=(const content_type_reader&) = default;
							content_type_reader(content_type_reader&&) = default;
							content_type_reader& operator=(content_type_reader&&) = default;
							virtual ~content_type_reader(void) = default;

							virtual std::shared_ptr<solids::lib::video::sink::d3d11::base::rtti> read(const std::wstring& name) override;

						protected:
							content_type_reader(solids::lib::video::sink::d3d11::base::engine& core, const std::uint64_t target_type_id);
							virtual std::shared_ptr<T> _read(const std::wstring& name) = 0;
						};
					};
				};
			};
		};
	};
};

#include "d3d11_content_type_reader.inl"