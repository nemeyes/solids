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
						class utility final
						{
						public:
							static void										get_file_name(const std::string& path, std::string& name);
							static void										get_directory(const std::string& path, std::string& dir);
							static std::tuple<std::string, std::string>		get_filename_and_directory(const std::string& path);
							static void										load_binary_file(const std::wstring& filename, std::vector<char>& data);
							static void										to_wide_string(const std::string& src, std::wstring& dst);
							static std::wstring								to_wide_string(const std::string& src);
							static void										tostring(const std::wstring& src, std::string& dst);
							static std::string								tostring(const std::wstring& src);

							utility(void) = delete;
							utility(const utility&) = delete;
							utility& operator=(const utility&) = delete;
							utility(utility&&) = delete;
							utility& operator=(utility&&) = delete;
							~utility(void) = default;
						};

						template <typename T>
						void update_value(std::function<bool()> increasePredicate, std::function<bool()> decreasePredicate, T& value, const T& delta, std::function<void(const T&)> updateFunc = nullptr, const T& minValue = std::numeric_limits<T>::min(), const T& maxValue = std::numeric_limits<T>::max())
						{
							bool update = false;
							if (increasePredicate() && value < maxValue)
							{
								value += delta;
								value = std::min(value, maxValue);
								update = true;
							}
							else if (decreasePredicate() && value > minValue)
							{
								value -= delta;
								value = std::max(value, minValue);
								update = true;
							}

							if (update && updateFunc != nullptr)
							{
								updateFunc(value);
							}
						}
					};
				};
			};
		};
	};
};
