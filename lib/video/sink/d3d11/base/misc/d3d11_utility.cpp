#include "d3d11_utility.h"

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

	void utility::get_file_name(const std::string& path, std::string& name)
	{
		std::string fullPath(path);
		std::replace(fullPath.begin(), fullPath.end(), '\\', '/');

		std::string::size_type lastSlashIndex = fullPath.find_last_of('/');

		if (lastSlashIndex == std::string::npos)
		{
			name = fullPath;
		}
		else
		{
			name = fullPath.substr(lastSlashIndex + 1, fullPath.size() - lastSlashIndex - 1);
		}
	}

	void utility::get_directory(const std::string& path, std::string& dir)
	{
		std::string fullPath(path);
		std::replace(fullPath.begin(), fullPath.end(), '\\', '/');

		std::string::size_type lastSlashIndex = fullPath.find_last_of('/');
		if (lastSlashIndex == std::string::npos)
		{
			dir = "";
		}
		else
		{
			dir = fullPath.substr(0, lastSlashIndex);
		}
	}

	std::tuple<std::string, std::string> utility::get_filename_and_directory(const std::string& path)
	{
		std::string fullPath(path);
		std::replace(fullPath.begin(), fullPath.end(), '\\', '/');

		std::string::size_type lastSlashIndex = fullPath.find_last_of('/');

		std::string directory;
		std::string filename;

		if (lastSlashIndex == std::string::npos)
		{
			directory = "";
			filename = fullPath;
		}
		else
		{
			directory = fullPath.substr(0, lastSlashIndex);
			filename = fullPath.substr(lastSlashIndex + 1, fullPath.size() - lastSlashIndex - 1);
		}

		return std::make_tuple(filename, directory);
	}

	void  utility::load_binary_file(const std::wstring& filename, std::vector<char>& data)
	{
		std::ifstream file(filename.c_str(), std::ios::binary);
		if (!file.good())
		{
			throw std::exception("Could not open file.");
		}

		file.seekg(0, std::ios::end);
		uint32_t size = (uint32_t)file.tellg();

		if (size > 0)
		{
			data.resize(size);
			file.seekg(0, std::ios::beg);
			file.read(&data.front(), size);
		}

		file.close();
	}

#pragma warning(push)
#pragma warning(disable: 4996)
	void utility::to_wide_string(const std::string& src, std::wstring& dst)
	{
		dst = std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(src);
	}

	std::wstring  utility::to_wide_string(const std::string& src)
	{
		return std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(src);
	}

	void utility::tostring(const std::wstring& src, std::string& dst)
	{
		dst = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(src);
	}

	std::string utility::tostring(const std::wstring& src)
	{
		return std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(src);
	}
#pragma warning(pop)
	
};
};
};
};
};
};
