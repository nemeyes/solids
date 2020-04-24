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
						class service final
						{
						public:
							void	add(std::uint64_t type_id, void* service);
							void	remove(std::uint64_t type_id);
							void*	get(std::uint64_t type_id) const;

							std::map<std::uint64_t, void*> _services;
						};
					};
				};
			};
		};
	};
};