#include "sld_ff_demuxer.h"
#include "ff_demuxer.h"


namespace solids
{
namespace lib
{
namespace container
{
namespace ff
{

	demuxer::demuxer(void)
	{
		_core = new solids::lib::container::ff::demuxer::core(this);
	}

	demuxer::~demuxer(void)
	{
		if (_core)
			delete _core;
		_core = NULL;
	}

	BOOL demuxer::is_running(void)
	{
		return _core->is_running();
	}

	BOOL demuxer::is_paused(void)
	{
		return _core->is_paused();
	}

	int32_t demuxer::play(const char* container, int32_t type)
	{
		return _core->play(container, type);
	}

	int32_t demuxer::resume(void)
	{
		return _core->resume();
	}

	int32_t demuxer::pause(void)
	{
		return _core->pause();
	}

	int32_t demuxer::stop(void)
	{
		return _core->stop();
	}

	int32_t demuxer::next(void)
	{
		return _core->next();
	}

};
};
};
};
