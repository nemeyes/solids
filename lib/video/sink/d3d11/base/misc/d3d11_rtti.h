#pragma once

#include <sld.h>

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
						class rtti
						{
						public:
							using id_type = std::uint64_t;
							virtual ~rtti() = default;

							virtual std::uint64_t type_id_instance(void) const = 0;

							virtual rtti * QueryInterface(const id_type)
							{
								return nullptr;
							}

							virtual BOOL is(id_type) const
							{
								return FALSE;
							}

							virtual BOOL is(const std::string &) const
							{
								return FALSE;
							}

							template <typename T>
							T* as() const
							{
								return (is(T::type_id_class()) ? reinterpret_cast<T*>(const_cast<rtti*>(this)) : nullptr);
							}

							virtual std::string to_string(void) const
							{
								return "RTTI";
							}

							virtual BOOL equals(const rtti * rhs) const
							{
								return (this == rhs) ? TRUE : FALSE;
							}
						};

#define RTTI_DECLARATIONS(Type, ParentType)																				 \
		public:																											 \
			static std::string type_name(void) { return std::string(#Type); }												 \
			static id_type type_id_class(void) { return _runtime_type_id; }												 \
			virtual id_type type_id_instance(void) const override { return Type::type_id_class(); }						 \
			virtual solids::lib::video::sink::d3d11::base::rtti * QueryInterface(const id_type id) override										 \
            {																											 \
				return (id == _runtime_type_id ? reinterpret_cast<solids::lib::video::sink::d3d11::base::rtti*>(this) : ParentType::QueryInterface(id)); \
            }																											 \
			virtual BOOL is(id_type id) const override															 \
			{																											 \
				return (id == _runtime_type_id ? TRUE : ParentType::is(id));												 \
			}																											 \
			virtual BOOL is(const std::string & name) const override														 \
			{																											 \
				return (name == type_name() ? TRUE : ParentType::is(name));												 \
			}																											 \
			private:																									 \
				static id_type _runtime_type_id;

#define RTTI_DEFINITIONS(Type) rtti::id_type Type::_runtime_type_id = reinterpret_cast<solids::lib::video::sink::d3d11::base::rtti::id_type>(&Type::_runtime_type_id);
					};
				};
			};
		};
	};
};
